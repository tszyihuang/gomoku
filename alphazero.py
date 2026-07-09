"""
AlphaZero 算法复现 —— 8×8 五子棋 (Gomoku)
============================================
轻量、单文件、详尽中文注释。专为教学与快速实验设计。

核心思想：
  不使用任何人类棋谱或手工特征，仅通过神经网络引导的蒙特卡洛树搜索
  (MCTS) 进行自我对弈，再从对弈数据中训练网络，循环迭代逐步变强。

模块概览：
  0. 工具函数       —— 棋盘格式转换、合法动作、D4 对称增强
  1. PolicyValueNet —— 双头残差卷积网络（策略头 + 价值头）
  2. MCTS           —— 纯神经网络引导的树搜索（无随机走子）
  3. SelfPlay       —— 自我对弈 + 数据收集 + D4 对称增强
  4. Trainer        —— 经验回放缓冲区 + 损失函数 + 优化

运行方式：
  python alphazero.py               # 从零开始训练
  python alphazero.py --resume XX   # 从第 XX 轮 checkpoint 恢复训练

参考：
  Silver et al. "Mastering the Game of Go without Human Knowledge" (2017)
  Silver et al. "A general reinforcement learning algorithm..." (2018)
"""

from __future__ import annotations

import math
import os
import random
import multiprocessing as mp
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gomoku import GomokuEnv, check_winner_from

# ==============================================================================
#  0. 工具函数
# ==============================================================================

# D4 对称群：棋盘的正方形对称变换共有 8 种（旋转 0°/90°/180°/270° × 翻转）
# 对于 8×8 棋盘，每个位置 (r, c) 在变换后会映射到新位置 (r', c')
# 我们用一维动作索引 (0~63) 来表示位置，预计算每个变换下的索引映射

def _build_d4_transforms(board_size: int = 8):
    """
    预计算 D4 群（8 种对称变换）下：
      - state_perm: 展平后 64 个位置的排列（用于变换策略向量 π）
      - action_map: action → 变换后的 action 的映射

    变换生成方式：先旋转 k*90°，再对旋转结果做左右翻转。共 2×4=8 种。
    变换矩阵（以标准棋盘坐标，左上角为原点）：
      旋转 90°:  (r, c) → (c, size-1-r)
      左右翻转:  (r, c) → (r, size-1-c)
    """
    transforms = []
    for rot in range(4):          # 0, 1, 2, 3 次 90° 旋转
        for flip in (False, True):
            mapping = []
            for r in range(board_size):
                for c in range(board_size):
                    rr, cc = r, c
                    # 应用旋转
                    for _ in range(rot):
                        rr, cc = cc, board_size - 1 - rr
                    # 应用翻转
                    if flip:
                        cc = board_size - 1 - cc
                    # 新位置的一维索引
                    mapping.append(rr * board_size + cc)
            transforms.append(np.array(mapping, dtype=np.int64))
    return transforms


D4_ACTION_MAPS = _build_d4_transforms(8)  # 8 个变换，每个是长度为 64 的排列


def board_to_tensor(board: np.ndarray, current_player: int) -> torch.Tensor:
    """
    将环境棋盘转换为神经网络的 3 通道输入张量。
    始终以"当前落子玩家"的视角构建特征，保证网络输出的一致性。

    参数:
        board: (8, 8) NumPy 数组，0=空, 1=黑棋(先手), 2=白棋(后手)
        current_player: 1 或 2，表示当前该谁落子

    返回:
        (3, 8, 8) 的 torch.float32 张量
          Channel 0: 当前玩家的棋子位置（1 表示有棋子）
          Channel 1: 对手的棋子位置（1 表示有棋子）
          Channel 2: 先手/后手标识（全 1 = 先手黑棋走，全 0 = 后手白棋走）
    """
    # 通道 0：当前玩家的棋子
    ch0 = (board == current_player).astype(np.float32)
    # 通道 1：对手的棋子
    ch1 = ((board != 0) & (board != current_player)).astype(np.float32)
    # 通道 2：当前玩家是否为黑棋（先手）
    # 这个通道让网络知道自己是先手还是后手，因为五子棋中先手有天然优势
    ch2 = np.ones((8, 8), dtype=np.float32) if current_player == 1 else np.zeros((8, 8), dtype=np.float32)

    stacked = np.stack([ch0, ch1, ch2], axis=0)  # (3, 8, 8)
    return torch.from_numpy(stacked)


def get_legal_actions(board: np.ndarray) -> np.ndarray:
    """返回所有合法动作（空格位置）的一维索引数组。"""
    return np.flatnonzero(board.ravel() == 0)


def augment_d4(states: list, policies: list, outcomes: list) -> list[tuple]:
    """
    利用五子棋棋盘的 D4 对称性进行数据增强。
    每条对局数据扩展为 8 条（旋转 + 镜像），大幅提升数据利用率。

    参数:
        states:   列表，每个元素为 (3, 8, 8) 的 torch.Tensor
        policies: 列表，每个元素为 (64,) 的 numpy 数组（MCTS 后验概率 π）
        outcomes: 列表，每个元素为标量 z（最终胜负，从当前玩家视角）

    返回:
        扩展后的 [(state_tensor, policy_array, z), ...] 列表，长度为输入的 8 倍
    """
    augmented = []
    for state, pi, z in zip(states, policies, outcomes):
        for perm in D4_ACTION_MAPS:
            # 变换状态：对最后两个维度 (8, 8) 应用相同的空间变换
            # 将展平索引转为 2D 行列，应用变换后再转回
            state_2d = state.numpy()  # (3, 8, 8)
            # 使用 perm 的逆来变换策略/状态
            # perm[a] = a 位置上的棋子在新棋盘上的位置
            # 即原位置 a 的棋子被搬到了 perm[a]
            # 对于状态通道，我们需要逆变换：新位置 perm[a] 的值来自旧位置 a
            inv_perm = np.argsort(perm)

            # 变换状态的每个通道
            flat_state = state_2d.reshape(3, 64)  # (3, 64)
            new_flat = flat_state[:, inv_perm]     # 重排
            new_state = torch.from_numpy(new_flat.reshape(3, 8, 8).copy())

            # 变换策略：π'[perm[a]] = π[a]，即新位置的先验概率来自旧位置
            new_pi = pi[inv_perm].copy()

            augmented.append((new_state, new_pi, z))
    return augmented


# ==============================================================================
#  0.5  BatchNorm 融合（推理加速）
# ==============================================================================

def fuse_batchnorm_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    将 BatchNorm2d 的参数融合到前置 Conv2d 中，消除推理时的 BN 计算。

    融合公式（conv 后接 bn）：
      y = γ * (W*x + b - μ) / √(σ² + ε) + β
        = (γ / √(σ² + ε)) * W * x  +  (γ * (b - μ) / √(σ² + ε) + β)

    融合后只需一次卷积即可得到等价结果，节省约 30% 推理时间。
    """
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)
    scale = gamma / std                                          # (out_channels,)

    # weight: (out_channels, in_channels, kH, kW)
    fused_weight = conv.weight * scale.view(-1, 1, 1, 1)

    if conv.bias is not None:
        fused_bias = scale * (conv.bias - mean) + beta
    else:
        fused_bias = beta - scale * mean

    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=conv.groups,
        bias=True,
    )
    fused.weight.data = fused_weight
    fused.bias.data = fused_bias
    return fused


def fuse_all_batchnorms(net: 'PolicyValueNet') -> 'PolicyValueNet':
    """
    将 PolicyValueNet 中所有 Conv→BN 序列融合。
    融合后 BN 层被替换为 nn.Identity，消除全部 BN 计算开销。
    """
    net.conv1 = fuse_batchnorm_conv(net.conv1, net.bn1)
    net.bn1 = nn.Identity()

    net.conv2 = fuse_batchnorm_conv(net.conv2, net.bn2)
    net.bn2 = nn.Identity()

    net.conv3 = fuse_batchnorm_conv(net.conv3, net.bn3)
    net.bn3 = nn.Identity()

    for blk in net.res_blocks:
        blk.conv1 = fuse_batchnorm_conv(blk.conv1, blk.bn1)
        blk.bn1 = nn.Identity()
        blk.conv2 = fuse_batchnorm_conv(blk.conv2, blk.bn2)
        blk.bn2 = nn.Identity()

    net.policy_conv = fuse_batchnorm_conv(net.policy_conv, net.policy_bn)
    net.policy_bn = nn.Identity()

    net.value_conv = fuse_batchnorm_conv(net.value_conv, net.value_bn)
    net.value_bn = nn.Identity()

    return net


# ==============================================================================
#  1. 双头神经网络 (PolicyValueNet)
# ==============================================================================

class ResidualBlock(nn.Module):
    """
    残差块 (Residual Block)。
    结构：Conv → BN → ReLU → Conv → BN → 加法跳跃连接 → ReLU
    残差连接让梯度可以"跳过"卷积层直接传播，有效缓解深层网络的退化问题，
    是 AlphaGo Zero / AlphaZero 架构的核心组件。
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x                           # 保存输入，用于跳跃连接
        out = F.relu(self.bn1(self.conv1(x)))  # 第一层：卷积 + BN + ReLU
        out = self.bn2(self.conv2(out))        # 第二层：卷积 + BN（先不加 ReLU）
        out = out + residual                   # 跳跃连接：F(x) + x
        return F.relu(out)                     # 最后统一做 ReLU 激活


class PolicyValueNet(nn.Module):
    """
    双头神经网络 —— AlphaZero 的唯一"大脑"。
    输入 3 通道棋盘特征，同时输出：
      - 策略 (Policy)：64 维动作概率分布（表示每个落子位置的优劣）
      - 价值 (Value)：一个标量 ∈ [-1, 1]（预测当前玩家的胜率）

    网络结构（可配置通道数和残差块数）：
      输入 (3, 8, 8)
        → 3 层共享卷积（num_channels 通道，默认 128）
        → num_res_blocks 个残差块（默认 4）
        ──→ 策略头：Conv(2) → FC(64) → Logits(64)
        ──→ 价值头：Conv(1) → FC(num_channels) → FC(1) → Tanh
    """

    def __init__(self, in_channels: int = 3, board_size: int = 8,
                 num_channels: int = 128, num_res_blocks: int = 4):
        super().__init__()
        self.board_size = board_size
        self.num_actions = board_size * board_size  # 64

        # ---- 主干网络 (共享层) ----
        # 三个卷积层逐步提取特征
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_channels)

        # 可配置数量的残差块，加深网络并保持梯度流动
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # ---- 策略头 (Policy Head) ----
        # 用 1×1 卷积降维到 2 通道，再展平接全连接层
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.num_actions)

        # ---- 价值头 (Value Head) ----
        # 用 1×1 卷积降维到 1 通道，再接两个全连接层，隐藏层大小与主干通道数一致
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, num_channels)
        self.value_fc2 = nn.Linear(num_channels, 1)

        # 权重初始化：使用 Kaiming 初始化有助于训练初期更快收敛
        self._init_weights()

    def _init_weights(self):
        """初始化所有权重为 Kaiming Normal 分布，偏置为 0。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        参数:
            x: (batch_size, 3, 8, 8) 的输入张量

        返回:
            policy_logits: (batch_size, 64) 各动作的原始 logits
            value:         (batch_size, 1)  胜率预测 ∈ [-1, 1]
        """
        batch_size = x.shape[0]

        # ---- 主干 ----
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        for res_block in self.res_blocks:
            x = res_block(x)

        # ---- 策略头 ----
        p = F.relu(self.policy_bn(self.policy_conv(x)))        # (B, 2, 8, 8)
        p = p.reshape(batch_size, -1)                          # (B, 128)
        policy_logits = self.policy_fc(p)                      # (B, 64)

        # ---- 价值头 ----
        v = F.relu(self.value_bn(self.value_conv(x)))          # (B, 1, 8, 8)
        v = v.reshape(batch_size, -1)                          # (B, 64)
        v = F.relu(self.value_fc1(v))                          # (B, 64)
        value = torch.tanh(self.value_fc2(v))                  # (B, 1), 范围 [-1, 1]

        return policy_logits, value

    def predict(self, state_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        """
        推理接口：输入单个状态张量，返回 (策略概率, 价值)。
        专为 MCTS 搜索设计，不计算梯度以提高效率。

        参数:
            state_tensor: (3, 8, 8) 或 (1, 3, 8, 8) 的张量

        返回:
            probs: (64,) 的 numpy 数组，经过 softmax 的动作概率
            value: float 标量，胜率预测
        """
        self.eval()
        with torch.no_grad():
            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)  # 添加 batch 维度
            # 将输入张量搬到网络参数所在设备（CPU→GPU）
            device = next(self.parameters()).device
            state_tensor = state_tensor.to(device)
            logits, value = self.forward(state_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            val = value.item()
        return probs, val


# ==============================================================================
#  2. 蒙特卡洛树搜索 (MCTS)
# ==============================================================================

class _TreeNode:
    """
    MCTS 树节点。

    每个节点代表一个棋盘状态和"将要落子"的玩家视角。
    存储该状态下的搜索统计信息：访问次数 N、累计价值 Q、先验概率 P。

    注意：Q 始终从"该节点的玩家"视角存储。也就是说：
      - 如果该节点玩家处于优势 → Q 为正
      - 如果该节点玩家处于劣势 → Q 为负
    回溯时通过取负号（-V）来翻转视角，因为这是零和博弈。
    """

    __slots__ = ('parent', 'action', 'children', 'N', 'Q', 'P')

    def __init__(self, parent: _TreeNode | None = None, action: int | None = None,
                 prior: float = 0.0):
        self.parent = parent          # 父节点
        self.action = action          # 从父节点到达此节点所走的动作（落子位置 0~63）
        self.children: dict[int, _TreeNode] = {}  # 子节点字典 {动作: 节点}
        self.N = 0                    # 访问次数
        self.Q = 0.0                  # 累计价值（从当前节点玩家视角）
        self.P = prior                # 先验概率 P(s, a)，由神经网络策略头给出


class MCTS:
    """
    纯神经网络引导的蒙特卡洛树搜索（无随机走子）。

    与传统的 MCTS 不同，AlphaZero 的 MCTS 完全不需要随机模拟 (Rollout)。
    到达叶节点时直接调用神经网络进行评估，获得先验概率 P 和局面价值 V，
    然后沿搜索路径回溯。

    支持两种模拟模式：
      - 顺序模式 (batch_size=1)：逐条模拟，适用于 CPU 推理
      - 批量模式 (batch_size>1)：多线程同时选择 + GPU 批量推理，配合虚拟损失
        (virtual loss) 确保搜索多样性。GPU 批量模式可将推理吞吐从 ~800/s
        提升至 ~20,000+/s。

    超参数说明：
      n_sim:          每次搜索的总模拟次数
      cpuct:          探索-利用平衡系数。值越大越倾向探索未访问的走法
      dirichlet_alpha: 狄利克雷噪声的浓度参数。α 越小噪声越集中（越"尖锐"）
      dirichlet_eps:   噪声混合权重。ε=0.25 表示 25% 来自噪声，75% 来自网络
      batch_size:      GPU 批量推理大小（1=顺序模式，>1=批量+虚拟损失模式）
      virtual_loss:    虚拟损失值，让同一批次内的并行模拟避开彼此已选路径
    """

    def __init__(self, net: PolicyValueNet, n_sim: int = 400, cpuct: float = 3.0,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25,
                 batch_size: int = 1, virtual_loss: float = 3.0):
        self.net = net
        self.n_sim = n_sim
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.batch_size = batch_size
        self.virtual_loss = virtual_loss
        self._last_root: _TreeNode | None = None  # 上一步搜索根节点，用于子树复用

    def get_action_probs(self, board: np.ndarray, current_player: int,
                         temperature: float = 1.0,
                         reuse_root: _TreeNode | None = None) -> np.ndarray:
        """
        对给定棋盘状态执行 MCTS 搜索，返回每个动作的概率分布 π。

        参数:
            board:          (8, 8) 棋盘，0=空, 1/2=棋子
            current_player: 当前该谁落子 (1 或 2)
            temperature:    温度参数 τ：
                            τ=1: 按访问次数比例的 1 次方采样（探索模式，前 30 步使用）
                            τ→0: 退化为贪心选择（最强手模式，30 步后使用）

        返回:
            probs: (64,) numpy 数组，每个合法动作的采样概率
        """
        legal_actions = get_legal_actions(board)
        if len(legal_actions) == 0:
            return np.zeros(64, dtype=np.float32)

        if len(legal_actions) == 1:
            probs = np.zeros(64, dtype=np.float32)
            probs[legal_actions[0]] = 1.0
            return probs

        # ---- 创建或复用根节点 ----
        # AlphaGo Zero 论文做法：上一步落子对应的子节点成为新根节点，
        # 其子树及全部统计量被保留，树的其余部分丢弃。
        if reuse_root is not None:
            root = reuse_root
            # 清理已不合法的子节点（棋盘上新落子导致该位置不可用）
            legal_set = set(legal_actions)
            stale = [a for a in root.children if a not in legal_set]
            for a in stale:
                del root.children[a]
            # 用新鲜 NN 评估更新先验概率，并添加新子节点（如有）
            state_tensor = board_to_tensor(board, current_player)
            nn_probs, _ = self.net.predict(state_tensor)
            for action in legal_actions:
                if action in root.children:
                    root.children[action].P = float(nn_probs[action])
                else:
                    root.children[action] = _TreeNode(
                        parent=root, action=action, prior=float(nn_probs[action]))
        else:
            root = _TreeNode()
            state_tensor = board_to_tensor(board, current_player)
            nn_probs, _ = self.net.predict(state_tensor)

            legal_probs = np.array([nn_probs[a] for a in legal_actions], dtype=np.float64)
            legal_probs = legal_probs / legal_probs.sum()

            for i, action in enumerate(legal_actions):
                root.children[action] = _TreeNode(parent=root, action=action, prior=float(legal_probs[i]))

        # ---- 狄利克雷噪声（仅在 alpha > 0 时添加，对弈模式可设为 0 跳过） ----
        if self.dirichlet_alpha > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for i, action in enumerate(legal_actions):
                child = root.children[action]
                child.P = float((1 - self.dirichlet_eps) * child.P + self.dirichlet_eps * noise[i])

        # ---- 执行 n_sim 次模拟 ----
        if self.batch_size <= 1:
            # 顺序模式：逐条模拟（原始 AlphaZero 方式，适合 CPU）
            for _ in range(self.n_sim):
                self._simulate(root, board.copy(), current_player)
        else:
            # GPU 批量模式：多线程模拟 + 批量 NN 推理 + 虚拟损失
            for batch_start in range(0, self.n_sim, self.batch_size):
                batch_n = min(self.batch_size, self.n_sim - batch_start)
                self._simulate_batch(root, board, current_player, batch_n)

        # ---- 从根节点访问次数计算动作概率 ----
        visits = np.zeros(64, dtype=np.float64)
        for action, child in root.children.items():
            visits[action] = child.N

        if temperature < 1e-8:
            best_action = int(np.argmax(visits))
            probs = np.zeros(64, dtype=np.float32)
            probs[best_action] = 1.0
        else:
            visits_pow = visits ** (1.0 / temperature)
            probs = (visits_pow / visits_pow.sum()).astype(np.float32)

        self._last_root = root  # 保存根节点，供下一时间步子树复用（论文做法）
        return probs

    # ==========================================================================
    #  顺序模拟 (batch_size=1)：原始的逐条 MCTS 模拟
    # ==========================================================================

    def _simulate(self, root: _TreeNode, board: np.ndarray, player: int):
        """
        执行一次完整的 MCTS 模拟：
          1. Selection  → 沿 PUCT 走到叶节点
          2. Expansion  → 用神经网络评估并展开
          3. Backup     → 将评估值沿路径向上传播

        优化：终局检测只在新叶子节点进行，内部节点已认证为非终局。
        """
        node = root
        sim_board = board
        sim_player = player
        path = [node]

        while node.children:
            node = self._select_child(node)
            r, c = divmod(node.action, 8)
            sim_board[r, c] = sim_player
            path.append(node)
            sim_player = 3 - sim_player

        # 到达叶节点：检查上一步落子是否导致终局
        if node is not root:
            last_player = 3 - sim_player
            r, c = divmod(node.action, 8)
            if check_winner_from(sim_board, last_player, r, c, GomokuEnv.CONNECT):
                self._backup(path, -1.0)
                return

        if np.all(sim_board != 0):
            value = 0.0
        else:
            state_tensor = board_to_tensor(sim_board, sim_player)
            nn_probs, value = self.net.predict(state_tensor)
            legal_actions = get_legal_actions(sim_board)
            for action in legal_actions:
                node.children[action] = _TreeNode(
                    parent=node, action=action, prior=float(nn_probs[action])
                )

        self._backup(path, value)

    # ==========================================================================
    #  GPU 批量模拟 (batch_size>1)：虚拟损失 + 批量 NN 推理
    # ==========================================================================

    def _simulate_batch(self, root: _TreeNode, board: np.ndarray, player: int,
                        batch_size: int):
        """
        执行 batch_size 条并行 MCTS 模拟。

        每条模拟独立地从根选择到叶节点，沿途施加虚拟损失 (virtual loss)
        以防止同一批次内的模拟都走同一条路径。所有叶节点一次性批量送入
        GPU 做 NN 推理，最后统一回溯并清除虚拟损失。

        虚拟损失原理（仅修改 N，不改 Q）：
          施加:  node.N += VL → U = cpuct·P·√N_parent/(1+N) 变小
          清除:  node.N -= VL → 恢复真实 N
          效果:  N 大的节点探索项被压制，批次内后续模拟自然分散到其他路径

        ⚠️ 关键：绝不修改 Q。因为零和博弈下 Q 的视角翻转会使得
           "child 的损失"变成"parent 的收益"，反直觉地让节点更受青睐。
        """
        # ---- Phase 1: 并行选择 batch_size 条路径到叶节点 ----
        leaf_data = []  # [(leaf_node, sim_board, sim_player, path), ...]
        for _ in range(batch_size):
            result = self._select_to_leaf(root, board.copy(), player)
            if result is not None:
                leaf_data.append(result)

        if not leaf_data:
            return  # 所有模拟都因终局/平局提前结束

        # ---- Phase 2: GPU 批量推理 ----
        # 将所有叶节点的棋盘状态打包成一个 batch，一次 forward 完成全部评估
        states_list = [board_to_tensor(b, p) for _, b, p, _ in leaf_data]
        batch_states = torch.stack(states_list)  # (B, 3, 8, 8)

        device = next(self.net.parameters()).device
        with torch.no_grad():
            logits, values = self.net.forward(batch_states.to(device))

        probs_batch = F.softmax(logits, dim=1).cpu().numpy()  # (B, 64)
        vals_batch = values.squeeze(-1).cpu().numpy()          # (B,)

        # ---- Phase 3: 展开 + 回溯（清除虚拟损失，写入真实值） ----
        for i, (node, sim_board, sim_player, path) in enumerate(leaf_data):
            if np.all(sim_board != 0):
                # 平局
                self._backup(path, 0.0, virtual=self.virtual_loss)
            else:
                # 展开子节点
                legal = get_legal_actions(sim_board)
                for a in legal:
                    node.children[a] = _TreeNode(
                        parent=node, action=a, prior=float(probs_batch[i][a])
                    )
                # 回溯真实价值（先清除虚拟损失）
                self._backup(path, float(vals_batch[i]), virtual=self.virtual_loss)

    def _select_to_leaf(self, root: _TreeNode, board: np.ndarray, player: int
                        ) -> tuple[_TreeNode, np.ndarray, int, list[_TreeNode]] | None:
        """
        从根节点选择到叶节点，沿途施加虚拟损失（仅增加 N，不改 Q）。

        返回 (leaf_node, leaf_board, leaf_player, path) 供后续批量推理使用。
        如果遇到终局或平局则直接回溯并返回 None。

        优化：终局检测只在新叶子节点进行。内部节点（已有 children）
        在之前模拟中已验证非终局，无需重复检测。
        """
        node = root
        sim_board = board
        sim_player = player
        path = [node]

        # 对根节点也施加虚拟损失（仅 N++，Q 不动）
        node.N += self.virtual_loss

        while node.children:
            node = self._select_child(node)
            r, c = divmod(node.action, 8)
            sim_board[r, c] = sim_player

            # 对选中节点施加虚拟损失（仅 N++，Q 不动）
            # U = cpuct * P * sqrt(N_parent) / (1 + N) → N 变大 → U 变小
            node.N += self.virtual_loss
            path.append(node)
            sim_player = 3 - sim_player

        # 到达叶节点：检查上一步落子是否导致终局
        # node.action 的落子方是 3-sim_player（翻转前的玩家）
        if node is not root:
            last_player = 3 - sim_player
            r, c = divmod(node.action, 8)
            if check_winner_from(sim_board, last_player, r, c, GomokuEnv.CONNECT):
                self._backup(path, -1.0, virtual=self.virtual_loss)
                return None

        if np.all(sim_board != 0):
            self._backup(path, 0.0, virtual=self.virtual_loss)
            return None

        return (node, sim_board, sim_player, path)

    # ==========================================================================
    #  共享方法：PUCT 选择 + 回溯
    # ==========================================================================

    def _select_child(self, node: _TreeNode) -> _TreeNode:
        """
        PUCT 选择：在节点的子节点中选出得分最高的。

        PUCT(s, a) = Q̄(s, a) + cpuct × P(s, a) × √(ΣN) / (1 + N(s, a))

        当使用虚拟损失时，child.Q 和 child.N 已包含虚拟损失的贡献，
        PUCT 公式会自动将"被虚拟损失惩罚"的节点排在后面。
        """
        # 预计算常数部分，避免每次迭代重复
        c = self.cpuct * math.sqrt(node.N + 1e-8)
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            n = child.N
            q = -child.Q / n if n > 0 else 0.0
            u = c * child.P / (1.0 + n)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _backup(self, path: list[_TreeNode], value: float, virtual: float = 0.0):
        """
        沿搜索路径回溯传播价值。

        参数:
            path:    从根到叶的节点列表
            value:   叶节点的评估价值（从叶节点玩家视角）
            virtual: 需要清除的虚拟 N 膨胀量（0=无虚拟损失）

        每次向上走一级时价值取负号（零和博弈的视角翻转）。
        """
        for node in reversed(path):
            if virtual > 0:
                node.N -= virtual       # 还原：扣除虚拟访问次数
            node.N += 1                  # 真实访问 +1
            node.Q += value              # 累加真实价值
            value = -value               # 视角翻转


# ==============================================================================
#  3. 自我对弈数据收集 (Self-Play)
# ==============================================================================

@dataclass
class SelfPlayConfig:
    """自我对弈的配置参数。"""
    temperature_threshold: int = 30  # 前 30 步使用温度 τ=1（探索），之后 τ→0（最强手）
    dirichlet_alpha: float = 0.3    # 狄利克雷噪声浓度参数
    dirichlet_eps: float = 0.25     # 噪声混合权重
    n_mcts_sim: int = 400           # 每步 MCTS 搜索模拟次数
    cpuct: float = 3.0              # PUCT 探索系数
    mcts_batch_size: int = 1        # MCTS 批量推理大小（1=顺序，>1=GPU批量+虚拟损失）
    virtual_loss: float = 3.0       # 虚拟损失值（仅在 batch_size>1 时生效）


def play_one_game(net: PolicyValueNet, config: SelfPlayConfig | None = None
                  ) -> list[tuple[torch.Tensor, np.ndarray, float]]:
    """
    让模型与自己下一局棋，收集训练数据。

    流程：
      1. 初始化棋盘
      2. 每步用 MCTS 计算动作概率 π
      3. 按 π 采样落子（前 N 步有温度，之后贪心）
      4. 终局后为每步分配价值 z（赢 +1，输 -1，平 0）
      5. 用 D4 对称性将数据扩展为 8 倍

    返回:
        增强后的训练数据列表，每个元素为 (state_tensor, policy_π, outcome_z)
    """
    if config is None:
        config = SelfPlayConfig()

    mcts = MCTS(net, n_sim=config.n_mcts_sim, cpuct=config.cpuct,
                dirichlet_alpha=config.dirichlet_alpha, dirichlet_eps=config.dirichlet_eps,
                batch_size=config.mcts_batch_size, virtual_loss=config.virtual_loss)

    env = GomokuEnv()
    board, info = env.reset()
    current_player: int = int(info['current_player'])  # 1 或 2

    # 缓冲区：记录每一步的（状态张量, 策略概率, 当前玩家）
    states: list[torch.Tensor] = []
    policies: list[np.ndarray] = []
    players: list[int] = []  # 记录每步是谁走的

    # 子树复用（AlphaGo Zero 论文做法：上一步落子对应的子节点成为新根节点）
    reuse_root: _TreeNode | None = None

    step = 0
    while True:
        # 构建神经网络输入
        state_tensor = board_to_tensor(board, current_player)

        # 温度控制：前 temperature_threshold 步 τ=1 鼓励探索，之后 τ→0 走最强手
        temperature = 1.0 if step < config.temperature_threshold else 0.0

        # MCTS 搜索
        action_probs = mcts.get_action_probs(board, current_player, temperature,
                                              reuse_root=reuse_root)

        # 记录数据
        states.append(state_tensor)
        policies.append(action_probs)
        players.append(current_player)

        # 根据概率采样动作
        if temperature < 1e-8:
            action = int(np.argmax(action_probs))
        else:
            # np.random.choice 按概率采样
            action = int(np.random.choice(64, p=action_probs))

        # 为下一步准备子树复用："对应于所下动作的子节点成为新的根节点"
        if mcts._last_root is not None and action in mcts._last_root.children:
            reuse_root = mcts._last_root.children[action]
            reuse_root.parent = None  # 脱离旧树
        else:
            reuse_root = None

        # 执行动作
        next_board, reward, terminated, truncated, info = env.step(action)

        if terminated:
            # ---- 游戏结束，为每一步分配最终价值 z ----
            # reward > 0.5 表示刚刚落子的玩家获胜
            # info['current_player'] 在终局时保持不变（仍是获胜方）
            winner: int | None = None
            if reward > 0.5:
                winner = int(info['current_player'])  # 赢家
            # else: 平局或非法走子惩罚

            data = []
            for s, pi, p in zip(states, policies, players):
                if winner is None:
                    z = 0.0  # 平局
                elif p == winner:
                    z = 1.0  # 该步玩家最终获胜
                else:
                    z = -1.0  # 该步玩家最终落败

                data.append((s, pi, z))

            # D4 对称增强：1 局 → 8 局数据
            return augment_d4(
                [d[0] for d in data],
                [d[1] for d in data],
                [d[2] for d in data],
            )

        board = next_board
        current_player = int(info['current_player'])
        step += 1


# ==============================================================================
#  多进程自对弈 —— 同时榨干 CPU + GPU
# ==============================================================================
#
#  策略：
#    GPU 并行模式 (默认)：spawn N 个 worker，每个持有独立 GPU 网络副本。
#      - MCTS 树操作 (PUCT/backup) → CPU 核心并行
#      - NN 推理 (forward)         → GPU 批量推理 (batch=32 per worker)
#      - 多个 worker 同时跑 → GPU 时间片轮转 → GPU 利用率 60%+
#      - worker 数 ≈ GPU 可容纳的网络副本数 (~8 个, 每个 ~6MB)
#
#    CPU 并行模式 (--cpu-selfplay)：forkserver Pool, 纯 CPU 推理。
#      - 适用于无 GPU 或 GPU 被其他任务占用时。
#
#  权重传递：写入临时文件，worker 读取。规避 mp.Queue pipe 64KB 缓冲区限制。

# 模块级全局：供 CPU forkserver Pool 使用
_cpu_worker_net: PolicyValueNet | None = None
_cpu_worker_config: SelfPlayConfig | None = None


def _cpu_worker_init(weight_path: str, num_channels: int, num_res_blocks: int,
                     config: SelfPlayConfig):
    """CPU Pool worker 初始化。"""
    global _cpu_worker_net, _cpu_worker_config
    _cpu_worker_net = PolicyValueNet(num_channels=num_channels, num_res_blocks=num_res_blocks)
    if weight_path and os.path.exists(weight_path):
        _cpu_worker_net.load_state_dict(
            torch.load(weight_path, map_location='cpu', weights_only=True))
    _cpu_worker_net.eval()
    _cpu_worker_config = config


def _cpu_worker_play_game(seed: int) -> list[tuple]:
    """CPU worker：运行一局自对弈。"""
    global _cpu_worker_net, _cpu_worker_config
    random.seed(seed)
    np.random.seed(seed)
    return play_one_game(_cpu_worker_net, _cpu_worker_config)


# ---- GPU 多进程 worker ----

def _gpu_worker(weight_path: str, num_channels: int, num_res_blocks: int,
                config: SelfPlayConfig, seeds: list[int],
                result_queue: mp.Queue, worker_id: int, gpu_id: int = 0):
    """
    GPU worker 进程：持有独立 GPU 网络副本，批量 MCTS 推理。

    每个 worker：
      1. 从文件加载权重 → 创建 GPU 网络 → torch.compile 加速
      2. 顺序运行分配的 games（每个 game 内 MCTS 批量推理 batch=32）
      3. 多个 worker 同时跑 → GPU 利用率大幅提升
    """
    # 每个 worker 绑定到指定 GPU，独立初始化 CUDA
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(worker_id * 10000 + seeds[0] if seeds else 0)

    # 创建 GPU 网络并加载权重
    net = PolicyValueNet(num_channels=num_channels, num_res_blocks=num_res_blocks)
    net = net.to(device)
    if weight_path and os.path.exists(weight_path):
        sd = torch.load(weight_path, map_location=device, weights_only=True)
        worker_sd = net.state_dict()
        filtered = {k: v for k, v in sd.items() if k in worker_sd}
        net.load_state_dict(filtered, strict=False)
    net.eval()

    # torch.compile (CUDA graph 在 spawn 子进程中独立创建，线程安全)
    if hasattr(torch, 'compile'):
        try:
            import torch._inductor.config as _ic
            _ic.triton.cudagraph_dynamic_shape_warn_limit = None
            net = torch.compile(net, mode='reduce-overhead')
            # 预热
            net(torch.zeros(1, 3, 8, 8, device=device))
        except Exception:
            pass

    # 运行分配的对局（结果转 numpy，规避 torch tensor 跨进程序列化问题）
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        game_data = play_one_game(net, config)
        # 转换: torch.Tensor → numpy array (mp.Queue 安全)
        serialized = [(s.cpu().numpy(), p, z) for s, p, z in game_data]
        result_queue.put(serialized)


# ---- 主调度函数 ----

def _run_selfplay(net: PolicyValueNet, config: SelfPlayConfig,
                   num_games: int, device: torch.device,
                   num_workers: int = 0,
                   num_channels: int = 128, num_res_blocks: int = 4,
                   force_cpu: bool = False, force_gpu: bool = False,
                   ) -> list[list[tuple]]:
    """
    运行 self-play 对局。自动选择最优模式：

      - GPU 并行 (默认): spawn N 个 GPU worker，CPU+GPU 同时满载
      - CPU 并行: forkserver Pool，纯 CPU
      - GPU 顺序: 单进程，用于 CPU 核心 < 8 且无 --cpu-selfplay 时

    返回: 每局游戏的增强数据列表 [(state, π, z), ...]
    """
    # ---- 模式决策 ----
    if force_cpu:
        use_gpu_parallel = False
        use_gpu_sequential = False
    elif force_gpu or (device.type == 'cuda' and (os.cpu_count() or 4) < 8):
        use_gpu_parallel = False
        use_gpu_sequential = True
    elif device.type == 'cuda':
        use_gpu_parallel = True   # 默认：GPU 多进程并行
        use_gpu_sequential = False
    else:
        use_gpu_parallel = False
        use_gpu_sequential = False

    # =====================================================================
    #  模式 1: GPU 多进程并行（默认，同时用 CPU + GPU）
    # =====================================================================
    if use_gpu_parallel:
        if num_workers <= 0:
            # GPU worker 数：每个占 ~100MB 显存（网络 6MB + CUDA ctx + compile graph）
            # RTX 5090 32GB → 理论上限 ~300 workers，实际受限于游戏数
            # 默认 game 数即 worker 数（一 worker 一局，最大化并行）
            num_workers = min(num_games, 32)

        if num_workers <= 1 or num_games <= 1:
            # 回退：单进程 GPU
            results = []
            for i in range(num_games):
                seed = random.randint(0, 2 ** 31 - 1)
                random.seed(seed); np.random.seed(seed)
                results.append(play_one_game(net, config))
            return results

        # 写入权重文件
        import tempfile as _tmp
        clean_sd = _unwrap_state_dict(
            {k: v.cpu().clone() for k, v in net.state_dict().items()})
        weight_path = os.path.join(_tmp.gettempdir(),
                                    f'az_gpu_weights_{os.getpid()}.pt')
        torch.save(clean_sd, weight_path)

        # 分配种子给各 worker（每人 num_games/num_workers 局）
        all_seeds = [random.randint(0, 2 ** 31 - 1) for _ in range(num_games)]
        seed_chunks = []
        for i in range(num_workers):
            start = i * num_games // num_workers
            end = (i + 1) * num_games // num_workers
            seed_chunks.append(all_seeds[start:end])

        # 启动 GPU worker 进程（轮询分配到多张 GPU）
        num_gpus = torch.cuda.device_count()
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        workers = []
        for i in range(num_workers):
            if not seed_chunks[i]:
                continue
            gpu_id = i % num_gpus  # 轮询分配：worker 0→GPU0, 1→GPU1, 2→GPU2, 3→GPU0, ...
            p = ctx.Process(
                target=_gpu_worker,
                args=(weight_path, num_channels, num_res_blocks,
                      config, seed_chunks[i], result_queue, i, gpu_id),
                daemon=True,
            )
            p.start()
            workers.append(p)

        # 收集结果（numpy → tensor 还原）
        results = []
        t_start = time.perf_counter()
        for _ in range(num_games):
            raw = result_queue.get()
            results.append([(torch.from_numpy(s), p, z) for s, p, z in raw])

        elapsed = time.perf_counter() - t_start
        total_moves = sum(len(g) // 8 for g in results)
        print(f'    [GPU×{num_workers}@{num_gpus}] {num_games} 局完成: '
              f'{elapsed:.1f}s ({elapsed/num_games:.1f}s/局, '
              f'~{total_moves} 原始步)')

        # 等待所有 worker 退出
        for p in workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

        # 清理
        try:
            os.remove(weight_path)
        except OSError:
            pass

        return results

    # =====================================================================
    #  模式 2: GPU 顺序（CPU 核心少时的回退）
    # =====================================================================
    if use_gpu_sequential:
        results = []
        report_every = max(1, num_games // 5) if num_games >= 5 else 1
        t_start = time.perf_counter()
        for i in range(num_games):
            seed = random.randint(0, 2 ** 31 - 1)
            random.seed(seed); np.random.seed(seed)
            game_data = play_one_game(net, config)
            results.append(game_data)
            if (i + 1) % report_every == 0 or i == num_games - 1:
                avg = (time.perf_counter() - t_start) / (i + 1)
                moves = len(game_data) // 8
                print(f'    [GPU] {i+1}/{num_games} 局 (均 {avg:.1f}s/局, '
                      f'本局 ~{moves} 步)')
        return results

    # =====================================================================
    #  模式 3: CPU 多进程并行
    # =====================================================================
    if num_workers <= 0:
        num_workers = max(1, min((os.cpu_count() or 4) - 1, num_games, 64))

    if num_workers <= 1 or num_games <= 1:
        results = []
        for _ in range(num_games):
            seed = random.randint(0, 2 ** 31 - 1)
            random.seed(seed); np.random.seed(seed)
            results.append(play_one_game(net, config))
        return results

    import tempfile as _tmp
    clean_sd = _unwrap_state_dict(
        {k: v.cpu().clone() for k, v in net.state_dict().items()})
    weight_path = os.path.join(_tmp.gettempdir(),
                                f'az_cpu_weights_{os.getpid()}.pt')
    torch.save(clean_sd, weight_path)

    seeds = [random.randint(0, 2 ** 31 - 1) for _ in range(num_games)]
    effective_workers = min(num_workers, num_games)
    ctx = mp.get_context('forkserver')
    with ctx.Pool(processes=effective_workers,
                  initializer=_cpu_worker_init,
                  initargs=(weight_path, num_channels, num_res_blocks, config)) as pool:
        results = pool.map(_cpu_worker_play_game, seeds)

    try:
        os.remove(weight_path)
    except OSError:
        pass

    return results


def _shutdown_selfplay_pool():
    """兼容旧接口。GPU worker 每次用完即清理，无需显式关闭。"""
    pass


# ==============================================================================
#  4. 训练器 (Trainer)
# ==============================================================================

class ReplayBuffer:
    """
    经验回放缓冲区。
    用 deque 存储训练样本，当超过最大容量时自动丢弃最旧的数据，
    确保训练数据始终反映最新的对弈水平。
    """

    def __init__(self, max_size: int = 100000):
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, data: list[tuple]):
        """添加多条数据到缓冲区。"""
        self.buffer.extend(data)

    def sample(self, batch_size: int) -> list[tuple]:
        """随机采样一个批次。"""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def compute_loss(
    net: PolicyValueNet,
    states: torch.Tensor,       # (B, 3, 8, 8)
    target_policies: torch.Tensor,  # (B, 64)  目标策略 π
    target_values: torch.Tensor,    # (B, 1)   目标价值 z
    l2_weight: float = 1e-4,
) -> tuple[torch.Tensor, dict]:
    """
    计算 AlphaZero 的总损失。

    总损失 = 价值误差 + 策略误差 + L2 正则化

    L = (z - v)²                          ← 价值损失（MSE，让网络学会判断局面好坏）
        - πᵀ log(p)                       ← 策略损失（交叉熵，让网络模仿 MCTS 的搜索策略）
        + c ||θ||²                         ← L2 正则化（防止过拟合）

    参数:
        net:             神经网络
        states:          批次状态 (B, 3, 8, 8)
        target_policies: MCTS 搜索出的目标策略概率 π (B, 64)
        target_values:   对局结果 z ∈ {-1, 0, +1} (B, 1)
        l2_weight:       L2 正则化系数 c

    返回:
        total_loss: 标量损失
        components: 各项损失值的字典（用于日志输出）
    """
    logits, values = net(states)

    # ---- 价值损失：均方误差 (MSE) ----
    # 让网络的价值预测 v 尽可能接近真实对局结果 z
    value_loss = F.mse_loss(values, target_values)

    # ---- 策略损失：交叉熵 (Cross-Entropy) ----
    # 让网络的策略输出 p 尽可能模仿 MCTS 搜索出的后验概率 π
    # F.cross_entropy 内部会先做 log_softmax，所以我们传入原始 logits
    policy_loss = F.cross_entropy(logits, target_policies)

    # ---- L2 正则化 ----
    # 对所有参数施加 L2 惩罚，防止过拟合
    l2_loss = 0.0
    for param in net.parameters():
        l2_loss += torch.sum(param ** 2)

    total_loss = value_loss + policy_loss + l2_weight * l2_loss

    return total_loss, {
        'total': total_loss.item(),
        'value': value_loss.item(),
        'policy': policy_loss.item(),
        'l2': (l2_weight * l2_loss).item(),
    }


def _unwrap_state_dict(state_dict: dict) -> dict:
    """移除 torch.compile 添加的 _orig_mod. 前缀，得到干净的参数名。"""
    clean = {}
    for k, v in state_dict.items():
        clean[k.replace('_orig_mod.', '')] = v
    return clean


def save_checkpoint(net: PolicyValueNet, optimizer: torch.optim.Optimizer,
                    iteration: int, save_dir: str = 'checkpoints'):
    """保存模型检查点（始终保存干净参数名，不受 torch.compile 影响）。"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'alphazero_iter_{iteration:04d}.pth')
    torch.save({
        'iteration': iteration,
        'model_state_dict': _unwrap_state_dict(net.state_dict()),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f'  [Checkpoint] 已保存: {path}')


def load_checkpoint(net: PolicyValueNet, optimizer: torch.optim.Optimizer | None,
                    iteration: int, save_dir: str = 'checkpoints') -> int:
    """加载模型检查点。返回加载的迭代数。兼容编译/未编译网络。"""
    path = os.path.join(save_dir, f'alphazero_iter_{iteration:04d}.pth')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    # 加载时也做一次 unwrap，以兼容各种保存格式
    saved_sd = _unwrap_state_dict(checkpoint['model_state_dict'])
    current_sd = _unwrap_state_dict(net.state_dict())
    # 只加载存在的 key（兼容不同网络结构）
    filtered_sd = {k: v for k, v in saved_sd.items() if k in current_sd}
    net.load_state_dict(filtered_sd, strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'  [Checkpoint] 已加载: {path}')
    return checkpoint['iteration']


# ==============================================================================
#  5. 主训练循环
# ==============================================================================

def main():
    """
    AlphaZero 训练主函数入口。

    训练流程（每个迭代）：
      1. 自我对弈：用当前网络生成 N 局对弈数据
      2. 数据增强：利用 D4 棋盘对称性将数据扩展 8 倍
      3. 存入缓冲区：保留最近 M 万条数据
      4. 训练：从缓冲区采样，执行 K 次梯度更新（启用 AMP 混合精度）
      5. 每 C 轮保存一次检查点
    """
    import argparse
    parser = argparse.ArgumentParser(description='AlphaZero 五子棋训练')
    parser.add_argument('--resume', type=int, default=None, help='从指定轮数恢复训练')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--selfplay-games', type=int, default=30,
                        help='每轮自我对弈局数（GPU 推荐 30-50，CPU 推荐 15-30）')
    parser.add_argument('--epochs', type=int, default=10,
                        help='每轮训练 epoch 数')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='批次大小')
    parser.add_argument('--iterations', type=int, default=2000,
                        help='总迭代轮数')
    parser.add_argument('--mcts-sims', type=int, default=600,
                        help='MCTS 模拟次数（8×8 推荐 400-800，冷启动阶段越多越好）')
    parser.add_argument('--buffer-size', type=int, default=200000,
                        help='回放缓冲区大小（太大=早期噪声数据滞留）')
    parser.add_argument('--l2-weight', type=float, default=1e-4,
                        help='L2 正则化系数')
    parser.add_argument('--checkpoint-freq', type=int, default=10,
                        help='检查点保存频率（轮）')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num-channels', type=int, default=128,
                        help='网络主干通道数（GPU 推荐 128，CPU 推荐 64）')
    parser.add_argument('--num-res-blocks', type=int, default=4,
                        help='残差块数量（GPU 推荐 4，CPU 推荐 2）')
    parser.add_argument('--no-amp', action='store_true',
                        help='禁用 AMP 混合精度训练')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='并行 self-play 进程数（0=自动检测 CPU 核心数）')
    parser.add_argument('--no-compile', action='store_true',
                        help='禁用 torch.compile 加速')
    parser.add_argument('--cpu-selfplay', action='store_true',
                        help='强制使用 CPU 多进程自对弈（即使有 GPU）')
    parser.add_argument('--gpu-selfplay', action='store_true',
                        help='强制使用 GPU 顺序自对弈（仅推荐 CPU<8 核时）')
    parser.add_argument('--mcts-batch-size', type=int, default=0,
                        help='MCTS GPU 批量推理大小（0=自动: GPU 默认32, CPU 默认1）')
    args = parser.parse_args()

    # ---- 设置随机种子 ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 自动检测设备 ----
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # 启用 TF32 tensor core 加速（RTX 30 系列及以上支持）
    # 对 8×8 的小网络影响不大，但大 batch 训练时有帮助
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        # 抑制 torch.compile CUDA graph 动态 shape 警告
        # 我们的 batch size 变化（1/32/2048）是预期行为，内存开销可忽略
        import torch._inductor.config as _inductor_config
        _inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None

    use_amp = (device.type == 'cuda') and (not args.no_amp)

    # ---- 自对弈模式 ----
    # 默认: GPU 多进程并行 (同时用 CPU+GPU)
    # --cpu-selfplay: 强制纯 CPU
    # --gpu-selfplay: 强制 GPU 顺序（单进程）
    cpu_cores = os.cpu_count() or 4
    if args.cpu_selfplay:
        selfplay_mode = 'CPU'
    elif args.gpu_selfplay:
        selfplay_mode = 'GPU_seq'
    elif device.type == 'cuda':
        selfplay_mode = 'GPU_parallel'
    else:
        selfplay_mode = 'CPU'

    # 自动检测 worker 数量
    num_workers = args.num_workers
    if num_workers == 0 and selfplay_mode == 'CPU':
        num_workers = max(1, min(cpu_cores - 1, args.selfplay_games, 64))

    # GPU 并行时默认 worker 数
    gpu_workers = (args.num_workers if args.num_workers > 0
                   else min(args.selfplay_games, 32)) if selfplay_mode == 'GPU_parallel' else 0

    # ---- 打印系统信息 ----
    print('=' * 60)
    print('  系统信息')
    print('=' * 60)
    print(f'  计算设备:     {device}')
    if device.type == 'cuda':
        print(f'  GPU 数量:     {torch.cuda.device_count()}')
        print(f'  GPU 0 型号:   {torch.cuda.get_device_name(0)}')
        print(f'  CUDA 版本:    {torch.version.cuda}')
        print(f'  显存总量:     {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')
        print(f'  AMP 混合精度: {"启用" if use_amp else "禁用"}')
    print(f'  自对弈模式:   {selfplay_mode}'
          + (f' ({gpu_workers} GPU workers)' if selfplay_mode == 'GPU_parallel'
             else f' ({num_workers} CPU workers)' if selfplay_mode == 'CPU' else ''))
    print(f'  CPU 核心数:   {cpu_cores}')
    print(f'  网络通道数:   {args.num_channels}')
    print(f'  残差块数:     {args.num_res_blocks}')
    print(f'  随机种子:     {args.seed}')
    print(f'超参数: {args}')
    print('=' * 60)

    # ---- 初始化网络和优化器 ----
    net = PolicyValueNet(
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
    ).to(device)

    # torch.compile 编译网络：首次调用时花几秒编译，之后推理速度提升 2-3×
    # 对于 MCTS 中数千次的单样本 forward，编译收益巨大
    if hasattr(torch, 'compile') and (not args.no_compile):
        try:
            print('  正在编译网络 (torch.compile)...', end=' ', flush=True)
            t0 = time.perf_counter()
            net = torch.compile(net, mode='reduce-overhead')
            # 预热：运行一次虚拟 forward 触发编译
            dummy = torch.zeros(1, 3, 8, 8, device=device)
            net(dummy)
            elapsed = time.perf_counter() - t0
            print(f'完成 ({elapsed:.1f}s)')
        except Exception as e:
            print(f'编译失败 ({e})，回退到普通模式')

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    # 使用 AdamW 的内置 weight_decay 替代手动 L2，效率更高

    # AMP 梯度缩放器（仅在 CUDA 时使用）
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    total_params = sum(p.numel() for p in net.parameters())
    print(f'  网络参数量:   {total_params:,}')
    print('=' * 60)

    # ---- 恢复训练 ----
    start_iteration = 0
    if args.resume is not None:
        start_iteration = load_checkpoint(net, optimizer, args.resume, args.save_dir) + 1

    # ---- 初始化组件 ----
    buffer = ReplayBuffer(max_size=args.buffer_size)

    # 自动确定 MCTS 批量大小
    if args.mcts_batch_size > 0:
        mcts_batch_size = args.mcts_batch_size
    elif selfplay_mode in ('GPU_parallel', 'GPU_seq'):
        mcts_batch_size = 64   # GPU 模式：批量推理 + 虚拟损失
    else:
        mcts_batch_size = 1    # CPU 模式：顺序模拟

    selfplay_config = SelfPlayConfig(
        n_mcts_sim=args.mcts_sims,
        mcts_batch_size=mcts_batch_size,
    )

    print()
    print(f'  每轮自我对弈: {args.selfplay_games} 局')
    print(f'  每轮训练:     {args.epochs} epochs × batch={args.batch_size}')
    print(f'  MCTS 模拟:    {args.mcts_sims} 次/步')
    if mcts_batch_size > 1:
        print(f'  MCTS 批量:    {mcts_batch_size} (GPU 虚拟损失模式)')
    print(f'  缓冲区上限:   {args.buffer_size:,} 条')
    if start_iteration > 0:
        print(f'  从第 {start_iteration} 轮恢复')
    print('=' * 60 + '\n')

    for iteration in range(start_iteration, start_iteration + args.iterations):
        print(f'--- 第 {iteration + 1} 轮 / {start_iteration + args.iterations} ---')

        # ---- 第一阶段：自对弈 ----
        print(f'  自对弈 [{selfplay_mode}] ({args.selfplay_games} 局)...')
        t0 = time.perf_counter()
        all_game_data = _run_selfplay(
            net, selfplay_config, args.selfplay_games, device,
            num_workers=num_workers,
            num_channels=args.num_channels,
            num_res_blocks=args.num_res_blocks,
            force_cpu=(selfplay_mode == 'CPU'),
            force_gpu=(selfplay_mode == 'GPU_seq'),
        )
        total_new_data = 0
        for game_data in all_game_data:
            buffer.add(game_data)
            total_new_data += len(game_data)
        elapsed = time.perf_counter() - t0
        print(f'  完成！用时 {elapsed:.1f}s, 新增 {total_new_data:,} 条数据, '
              f'缓冲区: {len(buffer):,} 条')

        # ---- 第二阶段：训练 ----
        print(f'  训练 ({args.epochs} epochs)...')
        net.train()
        for epoch in range(args.epochs):
            if len(buffer) < args.batch_size:
                print(f'    缓冲区数据不足 ({len(buffer)} < {args.batch_size})，跳过训练')
                break

            batch = buffer.sample(args.batch_size)

            # 组装 batch tensor
            batch_states = torch.stack([item[0] for item in batch]).to(device)
            batch_policies = torch.from_numpy(
                np.array([item[1] for item in batch], dtype=np.float32)
            ).to(device)
            batch_values = torch.from_numpy(
                np.array([[item[2]] for item in batch], dtype=np.float32)
            ).to(device)

            # 前向 + 损失计算（AMP 混合精度）
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    total_loss, components = compute_loss(
                        net, batch_states, batch_policies, batch_values,
                        l2_weight=0.0,  # L2 已由 AdamW weight_decay 处理
                    )
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss, components = compute_loss(
                    net, batch_states, batch_policies, batch_values,
                    l2_weight=args.l2_weight,
                )
                total_loss.backward()
                optimizer.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'    Epoch {epoch + 1}/{args.epochs} | '
                      f'Total: {components["total"]:.4f} | '
                      f'Value: {components["value"]:.4f} | '
                      f'Policy: {components["policy"]:.4f} | '
                      f'L2: {components["l2"]:.6f}')

        # ---- 保存检查点 ----
        if (iteration + 1) % args.checkpoint_freq == 0:
            save_checkpoint(net, optimizer, iteration + 1, args.save_dir)

        # ---- 打印显存使用 ----
        if device.type == 'cuda':
            allocated = torch.cuda.max_memory_allocated(0) / (1024**3)
            print(f'  GPU 显存峰值: {allocated:.2f} GB')

    _shutdown_selfplay_pool()
    print('\n训练完成！')


if __name__ == '__main__':
    main()
