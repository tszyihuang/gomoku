# AlphaZero 五子棋 (Gomoku)

基于 AlphaZero 算法的 8×8 五子棋强化学习项目。从零开始，不依赖任何人类棋谱或手工特征，仅通过**自我对弈 + 神经网络引导的蒙特卡洛树搜索 (MCTS)** 循环迭代，逐步学会下棋。

## 项目结构

```
gomoku/
├── alphazero.py          # ★ AlphaZero 核心实现（单文件：网络 + MCTS + 自对弈 + 训练）
├── gomoku.py             # 五子棋 Gymnasium 环境 (8×8)
├── arena.py              # 模型试炼场：不同 AI 互相对弈，统计胜率
├── play.py               # Flask Web 对弈界面（人机对战）
├── checkpoints/          # 模型检查点保存目录
├── 古典算法/
│   ├── minimax.py        # Minimax + Alpha-Beta 剪枝（纯搜索，无学习）
│   └── mcts.py           # 启发式 MCTS（纯搜索 + 手工评估函数，无神经网络）
└── pyproject.toml
```

## AlphaZero 算法概览

AlphaZero 的核心思想是一个自我强化的正反馈循环：

```
  ┌──────────┐    自我对弈     ┌──────────┐
  │  神经网络  │ ────────────→ │  训练数据  │
  │  f(θ)    │               │  (s, π, z) │
  └──────────┘               └──────────┘
       ↑                          │
       │      梯度更新             │ 监督学习
       │                          ↓
       │                    ┌──────────┐
       └────────────────────│  训练器    │
                            └──────────┘
```

1. **自我对弈 (Self-Play)**：当前网络通过 MCTS 搜索与自己对弈，产生训练数据 `(状态 s, MCTS 策略 π, 最终结果 z)`
2. **训练 (Train)**：用这些数据训练网络，让它学会预测 MCTS 的策略 π 和最终胜负 z
3. **网络变强 → MCTS 搜索更准 → 数据质量更高 → 网络更强**，形成良性循环

---

## 核心组件详解

### 1. 双头神经网络 (PolicyValueNet)

网络是 AlphaZero 唯一的"大脑"，同时输出两个预测：

| 头部 | 输出 | 含义 |
|------|------|------|
| **策略头 (Policy Head)** | 64 维向量 | 每个落子位置的优劣概率 p(s, a) |
| **价值头 (Value Head)** | 标量 ∈ [-1, 1] | 当前玩家的预估胜率 v(s) |

#### 网络结构

```
输入 (3, 8, 8)
  │
  ├─ Channel 0: 己方棋子位置
  ├─ Channel 1: 对手棋子位置        ← 始终以"当前落子方"视角构建
  └─ Channel 2: 是否先手（全1/全0）

  ↓
Conv(3→128, 3×3) + BN + ReLU
Conv(128→128, 3×3) + BN + ReLU
Conv(128→128, 3×3) + BN + ReLU       ← 三层共享卷积提取特征
  ↓
ResBlock × 4                          ← 残差块加深网络，保持梯度流动
  ↓
  ├─── 策略头 ───→ Conv(128→2, 1×1) + BN + ReLU → FC(128→64)
  └─── 价值头 ───→ Conv(128→1, 1×1) + BN + ReLU → FC(64→128) → FC(128→1) → Tanh
```

**设计要点：**

- **视角归一化**：输入始终以"当前玩家"视角构建。Channel 2 告知网络自己是先手（全1）还是后手（全0），因为五子棋中先手有天然优势。
- **残差连接**：`out = F(x) + x`，让梯度可以跳过卷积层直接传播，缓解深层网络的退化问题。
- **参数规模**：默认 128 通道 + 4 残差块，总参数量约 60 万，足够学习 8×8 五子棋的复杂模式。

#### BatchNorm 融合

推理时将 BatchNorm 参数合并到前置卷积层中，消除 BN 计算开销，推理速度提升约 30%。

融合公式（conv 后接 bn）：

```
y = γ · (W*x + b - μ) / √(σ² + ε) + β
  = (γ/√(σ²+ε)) · W · x  +  (γ(b-μ)/√(σ²+ε) + β)
                        ↑                           ↑
                    fused_weight               fused_bias
```

### 2. 蒙特卡洛树搜索 (MCTS)

AlphaZero 的 MCTS **完全不需要随机模拟 (Rollout)**。到达叶节点时直接调用神经网络评估，获得先验概率 P 和局面价值 V，然后沿路径回溯。

#### 一次模拟的四阶段

```
① Selection        ② Expansion          ③ Evaluation        ④ Backup
  沿 PUCT 走         到达叶节点，          神经网络评估          将 V 沿路径
  到叶节点           创建子节点            (P, V)               向上传播
                                                                  
     ○                  ○                                      ○ ← +V
    / \                /|\                                     /|\
   ○   ○     →       ○ ○ ○          →         →              ○ ○ ○
  /                   /|                                   +V→/|
 ○                   ○ ○                                     ○ ○
```

#### PUCT 选择公式

$$
\text{PUCT}(s, a) = \bar{Q}(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}
$$

- **第一项 (Q̄)**：平均价值，倾向于选择历史表现好的走法（利用）
- **第二项 (U)**：探索奖励，倾向于选择先验概率高但访问次数少的走法（探索）
- **c_puct**：平衡系数，默认 3.0

#### GPU 批量推理 + 虚拟损失

在 GPU 模式下，使用**虚拟损失 (Virtual Loss)** 实现并行 MCTS 模拟：

```
Phase 1: 并行选择           Phase 2: 批量推理          Phase 3: 回溯
线程1: root→A→B→leaf1  ┐                             展开 leaf1, 回溯 +V₁
线程2: root→C→D→leaf2  ├─→ GPU batch forward ─→      展开 leaf2, 回溯 +V₂
线程3: root→A→E→leaf3  ┘  (一次推理评估所有叶节点)     展开 leaf3, 回溯 +V₃
```

**虚拟损失原理**（仅修改 N，不改 Q）：
- 施加: `node.N += VL` → U 项变小 → 批次内其他模拟自然分散到不同路径
- 清除: 回溯时 `node.N -= VL` → 恢复真实访问次数

⚠️ **关键**：绝不修改 Q。因为零和博弈下 Q 的视角翻转会使 "child 的损失" 变成 "parent 的收益"，反直觉地让该节点更受青睐。

GPU 批量模式可将推理吞吐从 ~800/s 提升至 ~20,000+/s。

#### 狄利克雷噪声

在根节点的先验概率上添加狄利克雷噪声，确保探索多样性：

$$P'(s, a) = (1 - \varepsilon) \cdot P(s, a) + \varepsilon \cdot \text{Dir}(\alpha)$$

- α = 0.3：噪声浓度参数，越小噪声越"尖锐"
- ε = 0.25：噪声混合权重

### 3. 自我对弈 (Self-Play)

每局自我对弈的流程：

```
1. 初始化空棋盘
2. 每一步：
   a. 用 MCTS 搜索计算动作概率 π（前30步有温度探索，之后贪心）
   b. 按 π 采样落子
   c. 记录 (状态, π, 当前玩家)
3. 终局后为每一步分配价值 z：
   - 该步玩家最终获胜 → z = +1
   - 该步玩家最终落败 → z = -1
   - 平局 → z = 0
4. D4 对称增强：每条数据扩展为 8 条（旋转 × 镜像）
```

#### D4 对称增强

五子棋棋盘具有 D4 对称性（旋转 0°/90°/180°/270° × 翻转，共 8 种变换）。利用这一性质，每局数据扩展 8 倍：

- 1 局 60 步 → 480 条训练样本
- 30 局 → 14,400 条样本

每个变换同时应用于状态张量和策略向量，保证数据一致性。

#### 多进程并行自对弈

为充分利用 CPU + GPU：

- **GPU 并行模式**（默认）：spawn N 个 worker 进程，每个持有独立 GPU 网络副本，MCTS 树操作在 CPU 核心并行，NN 推理在 GPU 批量执行
- **CPU 并行模式**：forkserver Pool，纯 CPU 推理
- 权重通过临时文件传递，规避 mp.Queue pipe 的 64KB 缓冲区限制

### 4. 训练器 (Trainer)

#### 损失函数

总损失由三部分组成：

$$
L = \underbrace{(z - v)^2}_{\text{价值损失 (MSE)}} - \underbrace{\pi^T \log p}_{\text{策略损失 (交叉熵)}} + \underbrace{c \|\theta\|^2}_{\text{L2 正则化}}
$$

- **价值损失**：让网络学会判断局面好坏，预测最终胜负
- **策略损失**：让网络模仿 MCTS 的搜索策略，学会"直觉"落子
- **L2 正则化**：防止过拟合（AdamW 内置 weight_decay 替代手动计算）

#### 经验回放缓冲区

用固定容量的 deque 存储训练样本，自动丢弃最旧数据，确保训练数据始终反映最新对弈水平。

#### 训练技巧

| 技巧 | 说明 |
|------|------|
| **AMP 混合精度** | CUDA 下自动启用，加速训练并节省显存 |
| **torch.compile** | JIT 编译网络，首次编译后推理速度提升 2-3× |
| **AdamW** | 内置 weight_decay 的 Adam 变体，解耦权重衰减与学习率 |
| **TF32** | RTX 30 系列及以上自动启用 tensor core 加速 |

---

## 快速开始

### 安装依赖

```bash
pip install torch numpy gymnasium
```

### 训练模型

```bash
# 从零开始训练（推荐默认参数）
python alphazero.py

# 从 checkpoint 恢复训练
python alphazero.py --resume 200

# 自定义训练参数
python alphazero.py \
    --selfplay-games 50 \    # 每轮自对弈局数
    --mcts-sims 800 \        # MCTS 模拟次数
    --epochs 10 \            # 每轮训练轮数
    --batch-size 2048 \      # 批次大小
    --iterations 200 \       # 总迭代轮数
    --lr 0.001               # 学习率
```

### 模型对战

```bash
# 交互式菜单
python arena.py

# 命令行：AlphaZero vs 启发式 MCTS
python arena.py --a az_s400 --b mcts_t1.0 --episodes 200

# 并行对战（加速）
python arena.py --a az_s400 --b minimax_d3 --episodes 200 --workers 8
```

### Web 人机对弈

```bash
python play.py
# 浏览器访问 http://localhost:8080
```

---

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MCTS 模拟次数 | 600 | 越大越强但越慢（8×8 推荐 400-800） |
| cpuct | 3.0 | PUCT 探索系数 |
| dirichlet_alpha | 0.3 | 狄利克雷噪声浓度 |
| dirichlet_eps | 0.25 | 噪声混合权重 |
| 温度阈值 | 30 步 | 前 N 步 τ=1 (探索)，之后 τ→0 (最强手) |
| 缓冲区大小 | 200,000 | 太大会滞留早期噪声数据 |
| 学习率 | 0.001 | AdamW 初始学习率 |
| L2 正则化 | 1e-4 | 权重衰减系数 |
| 网络通道数 | 128 | GPU 推荐 128，CPU 推荐 64 |
| 残差块数 | 4 | GPU 推荐 4，CPU 推荐 2 |

---

## 算法对比

| 算法 | 类型 | 搜索 | 评估 | 特点 |
|------|------|------|------|------|
| **AlphaZero** | 强化学习 | MCTS + 神经网络引导 | 神经网络 | 从零自学，越训越强 |
| Minimax | 经典搜索 | Alpha-Beta 剪枝 | 手工评估函数 | 固定强度，不学习 |
| 启发式 MCTS | 经典搜索 | UCB1 树搜索 | 手工评估函数 | 比 Minimax 强，不学习 |
| Random | 基线 | 无 | 无 | 随机落子 |

---

## 参考

- Silver et al. ["Mastering the Game of Go without Human Knowledge"](https://www.nature.com/articles/nature24270) (Nature, 2017)
- Silver et al. ["A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play"](https://www.science.org/doi/10.1126/science.aar6404) (Science, 2018)
