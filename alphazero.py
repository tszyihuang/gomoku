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

    超参数说明：
      n_sim:          每次搜索的总模拟次数（推荐 400）
      cpuct:          探索-利用平衡系数。值越大越倾向探索未访问的走法
      dirichlet_alpha: 狄利克雷噪声的浓度参数。α 越小噪声越集中（越"尖锐"）
      dirichlet_eps:   噪声混合权重。ε=0.25 表示 25% 来自噪声，75% 来自网络
    """

    def __init__(self, net: PolicyValueNet, n_sim: int = 400, cpuct: float = 3.0,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25):
        self.net = net
        self.n_sim = n_sim
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def get_action_probs(self, board: np.ndarray, current_player: int,
                         temperature: float = 1.0) -> np.ndarray:
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
            return np.zeros(64, dtype=np.float32)  # 无合法动作（不应出现）

        # 如果只剩一个合法动作，直接返回 one-hot
        if len(legal_actions) == 1:
            probs = np.zeros(64, dtype=np.float32)
            probs[legal_actions[0]] = 1.0
            return probs

        # ---- 创建根节点并预展开 ----
        root = _TreeNode()
        state_tensor = board_to_tensor(board, current_player)

        # 用神经网络评估根状态，获取先验概率
        nn_probs, _ = self.net.predict(state_tensor)

        # 只保留合法动作的先验概率
        legal_probs = np.array([nn_probs[a] for a in legal_actions], dtype=np.float64)
        # 对合法动作的概率重新归一化
        legal_probs = legal_probs / legal_probs.sum()

        # 为合法动作创建子节点
        for i, action in enumerate(legal_actions):
            root.children[action] = _TreeNode(parent=root, action=action, prior=float(legal_probs[i]))

        # ---- 为根节点的先验概率添加狄利克雷噪声（仅自我对弈时使用） ----
        # 噪声的目的是强制探索：即使网络认为某一步很差，MCTS 也会给它一些访问机会
        # 这防止网络过早陷入局部最优策略
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
        for i, action in enumerate(legal_actions):
            child = root.children[action]
            # ε × 噪声 + (1-ε) × 原始先验
            child.P = float((1 - self.dirichlet_eps) * child.P + self.dirichlet_eps * noise[i])

        # ---- 执行 n_sim 次模拟 ----
        for _ in range(self.n_sim):
            self._simulate(root, board.copy(), current_player)

        # ---- 从根节点访问次数计算动作概率 ----
        visits = np.zeros(64, dtype=np.float64)
        for action, child in root.children.items():
            visits[action] = child.N

        if temperature < 1e-8:
            # 温度 ≈ 0：选择访问次数最多的动作（贪心）
            best_action = int(np.argmax(visits))
            probs = np.zeros(64, dtype=np.float32)
            probs[best_action] = 1.0
        else:
            # 温度 > 0：按 N^(1/τ) 的比例采样
            # τ=1 时按访问次数比例采样；τ 越小越倾向选择访问次数多的动作
            visits_pow = visits ** (1.0 / temperature)
            probs = (visits_pow / visits_pow.sum()).astype(np.float32)

        return probs

    def _simulate(self, root: _TreeNode, board: np.ndarray, player: int):
        """
        执行一次完整的 MCTS 模拟：
          1. Selection  (选择)：沿 PUCT 公式向下走到叶节点
          2. Expansion  (扩展)：用神经网络评估叶节点并展开
          3. Backup     (回溯)：将评估值沿路径向上传播

        参数:
            root:   根节点
            board:  当前棋盘（会在函数内被修改，传入的是副本）
            player: 当前玩家
        """
        node = root
        sim_board = board
        sim_player = player
        path = [node]         # 记录从根到叶的路径，用于回溯

        # ---- Phase 1: Selection (选择) ----
        # PUCT 公式：选择使 Q(s,a) + cpuct * P(s,a) * √(ΣN) / (1 + N(s,a)) 最大的子节点
        # 其中 Q(s,a) = -child.Q / child.N  （因为 child.Q 是对手视角，需要翻转）
        # 当 child.N == 0 时，Q 项取 0，只靠先验概率 U 来选择（保证每个子节点至少被访问一次）
        while node.children:
            node = self._select_child(node)
            # 在模拟棋盘上执行该动作
            r, c = divmod(node.action, 8)
            sim_board[r, c] = sim_player
            path.append(node)

            # 检查这个动作是否直接获胜（五连）
            if check_winner_from(sim_board, sim_player, r, c, GomokuEnv.CONNECT):
                # 该动作使当前玩家获胜 → 对手（子节点玩家）输了
                # 子节点的价值 = -1（从子节点玩家视角，它输了）
                value = -1.0
                # 直接进入回溯阶段，不需要神经网络评估
                self._backup(path, value)
                return

            # 切换到对手
            sim_player = 3 - sim_player  # 1→2 or 2→1

        # ---- Phase 2: Expansion & Evaluation (扩展与评估) ----
        # 到达未展开的叶节点，检查是否为平局
        if np.all(sim_board != 0):
            value = 0.0  # 棋盘满且无胜者 → 平局
        else:
            # 用神经网络评估该状态
            state_tensor = board_to_tensor(sim_board, sim_player)
            nn_probs, value = self.net.predict(state_tensor)

            # 展开节点：为所有合法动作创建子节点
            legal_actions = get_legal_actions(sim_board)
            for action in legal_actions:
                node.children[action] = _TreeNode(
                    parent=node, action=action, prior=float(nn_probs[action])
                )

        # ---- Phase 3: Backup (回溯) ----
        # 沿路径向上回传价值。每次向上走一级时，价值取负号：
        #   因为这是零和博弈——对当前玩家有利的局面，对上一级（对手）就是不利的。
        #   公式：Q_new = Q_old + V, N_new = N_old + 1
        #   向上传递时：V_parent = -V_child（视角翻转）
        self._backup(path, value)

    def _select_child(self, node: _TreeNode) -> _TreeNode:
        """
        PUCT 选择：在节点的子节点中选出得分最高的。

        PUCT(s, a) = Q̄(s, a) + cpuct × P(s, a) × √(Σ_b N(s, b)) / (1 + N(s, a))

        其中：
          Q̄(s, a) = -child.Q / child.N   是"从父节点视角"的平均动作价值
                 （child.Q 从子节点（对手）视角存储，取负号转回父节点视角）
          P(s, a) = child.P              是先验概率（神经网络策略头输出）
          N(s, a) = child.N              是该动作的访问次数
          Σ_b N(s, b) = node.N           是父节点的总访问次数

        直观理解：
          - Q̄ 项：选择历史上平均效果好的走法（利用）
          - U 项：选择访问次数少但先验概率高的走法（探索）
          - cpuct 系数控制探索与利用的平衡
        """
        sqrt_parent_N = math.sqrt(node.N + 1e-8)  # 父节点总访问次数的平方根

        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            # ---- 计算 Q 项（利用） ----
            # child.Q 从子节点（对手）视角存储，取负号转回父节点视角
            if child.N > 0:
                q_value = -child.Q / child.N
            else:
                q_value = 0.0  # 未曾访问过的子节点，Q 视为 0

            # ---- 计算 U 项（探索） ----
            # 分子：cpuct × P × √(ΣN)   — 先验概率越高、父节点访问越多 → U 越大
            # 分母：1 + N(s,a)           — 该子节点被访问越多 → U 越小
            u_value = self.cpuct * child.P * sqrt_parent_N / (1.0 + child.N)

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _backup(self, path: list[_TreeNode], value: float):
        """
        回溯：将叶节点评估值沿搜索路径向上传播。

        关键操作——每次向上传时价值取负号 (-V)：
          因为五子棋是零和交替博弈，如果一个局面对当前玩家有利 (V>0)，
          那么对上一级玩家（对手）就等量不利 (-V)。

        示例（三层路径）：
          叶节点（玩家 2 视角）: V = +0.8  → Q_leaf += 0.8
          中间节点（玩家 1 视角）: V = -0.8 → Q_mid  += -0.8
          根节点  （玩家 1 视角）: V = +0.8 → Q_root += +0.8
        """
        for node in reversed(path):
            node.N += 1          # 访问次数 +1
            node.Q += value      # 累加价值（从该节点玩家视角）
            value = -value       # 翻转视角，供上级节点使用


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
                dirichlet_alpha=config.dirichlet_alpha, dirichlet_eps=config.dirichlet_eps)

    env = GomokuEnv()
    board, info = env.reset()
    current_player: int = int(info['current_player'])  # 1 或 2

    # 缓冲区：记录每一步的（状态张量, 策略概率, 当前玩家）
    states: list[torch.Tensor] = []
    policies: list[np.ndarray] = []
    players: list[int] = []  # 记录每步是谁走的

    step = 0
    while True:
        # 构建神经网络输入
        state_tensor = board_to_tensor(board, current_player)

        # 温度控制：前 temperature_threshold 步 τ=1 鼓励探索，之后 τ→0 走最强手
        temperature = 1.0 if step < config.temperature_threshold else 0.0

        # MCTS 搜索
        action_probs = mcts.get_action_probs(board, current_player, temperature)

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


def save_checkpoint(net: PolicyValueNet, optimizer: torch.optim.Optimizer,
                    iteration: int, save_dir: str = 'checkpoints'):
    """保存模型检查点。"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'alphazero_iter_{iteration:04d}.pth')
    torch.save({
        'iteration': iteration,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f'  [Checkpoint] 已保存: {path}')


def load_checkpoint(net: PolicyValueNet, optimizer: torch.optim.Optimizer | None,
                    iteration: int, save_dir: str = 'checkpoints') -> int:
    """加载模型检查点。返回加载的迭代数。"""
    path = os.path.join(save_dir, f'alphazero_iter_{iteration:04d}.pth')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'])
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
    parser.add_argument('--selfplay-games', type=int, default=50,
                        help='每轮自我对弈局数')
    parser.add_argument('--epochs', type=int, default=10,
                        help='每轮训练 epoch 数')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='批次大小')
    parser.add_argument('--iterations', type=int, default=200,
                        help='总迭代轮数')
    parser.add_argument('--mcts-sims', type=int, default=800,
                        help='MCTS 模拟次数')
    parser.add_argument('--buffer-size', type=int, default=500000,
                        help='回放缓冲区大小')
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

    use_amp = (device.type == 'cuda') and (not args.no_amp)

    # ---- 打印 GPU 信息 ----
    print('=' * 60)
    print('  系统信息')
    print('=' * 60)
    print(f'  计算设备:     {device}')
    if device.type == 'cuda':
        print(f'  GPU 型号:     {torch.cuda.get_device_name(0)}')
        print(f'  CUDA 版本:    {torch.version.cuda}')
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'  显存总量:     {total_vram:.1f} GB')
        print(f'  AMP 混合精度: {"启用" if use_amp else "禁用"}')
    else:
        print(f'  AMP 混合精度: 不可用（需要 CUDA）')
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
    selfplay_config = SelfPlayConfig(n_mcts_sim=args.mcts_sims)

    print()
    print(f'  每轮自我对弈: {args.selfplay_games} 局')
    print(f'  每轮训练:     {args.epochs} epochs × batch={args.batch_size}')
    print(f'  MCTS 模拟:    {args.mcts_sims} 次/步')
    print(f'  缓冲区上限:   {args.buffer_size:,} 条')
    if start_iteration > 0:
        print(f'  从第 {start_iteration} 轮恢复')
    print('=' * 60 + '\n')

    for iteration in range(start_iteration, start_iteration + args.iterations):
        print(f'--- 第 {iteration + 1} 轮 / {start_iteration + args.iterations} ---')

        # ---- 第一阶段：自我对弈 ----
        print(f'  自我对弈 ({args.selfplay_games} 局)...')
        total_new_data = 0
        for game_idx in range(args.selfplay_games):
            game_data = play_one_game(net, selfplay_config)
            buffer.add(game_data)
            total_new_data += len(game_data)
            if (game_idx + 1) % 10 == 0 or game_idx == 0:
                print(f'    已完成 {game_idx + 1}/{args.selfplay_games} 局, '
                      f'缓冲区大小: {len(buffer):,}')

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

    print('\n训练完成！')


if __name__ == '__main__':
    main()
