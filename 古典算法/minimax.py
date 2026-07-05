"""
Minimax 剪枝搜索策略，用于 8×8 五子棋（自由落子规则）。

优化点：
  - 候选落子过滤：只在已有棋子附近搜索（曼哈顿距离 ≤ 2）
  - 落子排序：优先试探能直接获胜、能堵住对手五连的位置，加速剪枝
  - 迭代加深 + 时限控制
  - Negamax 框架：统一最大化/最小化视角，消除镜像重复代码
  - 静态搜索：深度耗尽时延申一层捕获威胁，避免地平线效应
"""

import time
import numpy as np
from gomoku import GomokuEnv, check_winner_from

# ---------- 棋盘与方向常量 ----------

N_CELLS = GomokuEnv.ROWS * GomokuEnv.COLS          # 棋盘格子总数（8×8 = 64）
_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]    # 四个搜索方向：水平、垂直、主对角线、副对角线

# 预计算每个格子的行 / 列索引，加速一维索引 → 二维坐标转换
_ROW = np.array([i // GomokuEnv.COLS for i in range(N_CELLS)], dtype=np.int8)
_COL = np.array([i % GomokuEnv.COLS for i in range(N_CELLS)], dtype=np.int8)

# ---------- 评分权重表 ----------
# 键为连子长度 run_len（1~4），值为 { 开放端数: 分数 }
# 开放端数：2 = 两端都空，1 = 一端被堵，0 = 两端都被堵
# 注：run_len=5 在搜索中由 check_winner_from 截获，不会走到评分逻辑
_WEIGHTS = {
    4: {2: 10_000_000, 1: 500_000, 0: 10},
    3: {2: 100_000,   1: 10_000,  0: 50},
    2: {2: 500,       1: 100,     0: 10},
    1: {2: 10,        1: 3,       0: 1},
}

# 获胜 / 阻挡的排序标记，保证这些位置被优先搜索
_WIN_SCORE = 1_000_000           # 直接获胜的评估分数（远高于任何位置评分）
_ORDER_WIN = 100_000             # 落子排序权重：自己可直接五连
_ORDER_BLOCK = 50_000            # 落子排序权重：用于堵住对手的五连
_QUIESCENCE_THRESHOLD = _WEIGHTS[4][1]  # 静态搜索阈值：评分超过此值则延申搜索

# 预计算每个位置距离棋盘中心的曼哈顿距离权重
# 中心最高（30），四角最低（0），用于落子排序中的"贴近中心优先"
_CX, _CY = (GomokuEnv.COLS - 1) / 2, (GomokuEnv.ROWS - 1) / 2
_CENTER_WEIGHT = np.array([
    30 - int(abs(r - _CY) + abs(c - _CX))
    for r in range(GomokuEnv.ROWS) for c in range(GomokuEnv.COLS)
], dtype=np.int32)

# 预计算"邻近"掩码：_NEAR_MASK[i, j] 表示格子 i 与 j 的曼哈顿距离 ≤ 2
_NEAR_MASK = np.zeros((N_CELLS, N_CELLS), dtype=bool)
for i in range(N_CELLS):
    ri, ci = _ROW[i], _COL[i]
    for j in range(N_CELLS):
        rj, cj = _ROW[j], _COL[j]
        if abs(ri - rj) + abs(ci - cj) <= 2:
            _NEAR_MASK[i, j] = True


def _trivial_move(board):
    """
    处理特殊情况：棋盘上 ≤ 1 个空格时直接返回，跳过搜索。
    返回 None 表示需要正常搜索。
    """
    avail = np.flatnonzero(board.ravel() == 0)
    n = len(avail)
    if n == 0:
        return None
    if n == 1:
        return int(avail[0])
    return None


def _score_player(board, player):
    """
    计算 `player` 在棋盘上的总评分。
    遍历该玩家的每一颗棋子，沿四个方向统计连子长度和开放端数，
    每条连线只从方向起点计数一次（避免重复）。
    """
    total = 0
    for idx in np.flatnonzero(board.ravel() == player):
        r, c = int(_ROW[idx]), int(_COL[idx])
        for dr, dc in _DIRECTIONS:
            # 前一个格子同色 → 此线已从更早的起点统计过，跳过
            br, bc = r - dr, c - dc
            if 0 <= br < GomokuEnv.ROWS and 0 <= bc < GomokuEnv.COLS and board[br, bc] == player:
                continue

            # 沿方向统计连子长度
            run_len = 1
            nr, nc = r + dr, c + dc
            while 0 <= nr < GomokuEnv.ROWS and 0 <= nc < GomokuEnv.COLS and board[nr, nc] == player:
                run_len += 1
                nr += dr
                nc += dc

            # 统计两端开放数
            open_ends = 0
            if 0 <= br < GomokuEnv.ROWS and 0 <= bc < GomokuEnv.COLS and board[br, bc] == 0:
                open_ends += 1
            er, ec = r + run_len * dr, c + run_len * dc
            if 0 <= er < GomokuEnv.ROWS and 0 <= ec < GomokuEnv.COLS and board[er, ec] == 0:
                open_ends += 1

            w = _WEIGHTS.get(run_len, _WEIGHTS[4])
            total += w.get(open_ends, 0)
    return total


def evaluate(board, player):
    """
    从 `player` 视角评估局面：己方得分 − 对方得分。正值表示优势。
    """
    opp = 3 - player
    return _score_player(board, player) - _score_player(board, opp)


def _relevant_moves(board, avail):
    """
    从空位中筛选候选落子：只返回与已有棋子曼哈顿距离 ≤ 2 的位置。
    大幅减少搜索分支，是搜索加速的关键。
    """
    flat = board.ravel()
    occupied = np.flatnonzero(flat != 0)
    if len(occupied) == 0:
        return list(avail)
    near = _NEAR_MASK[occupied].any(axis=0)
    result = [a for a in avail if near[a]]
    return result if result else list(avail)


def _order_moves(board, avail, player):
    """
    对候选落子排序，使最有希望的分支先被搜索，加速 alpha-beta 剪枝。
    优先级：己方能直接五连 ＞ 堵住对方五连 ＞ 靠近中心。
    """
    opp = 3 - player
    scored = []
    for action in avail:
        r, c = _ROW[action], _COL[action]
        w = _CENTER_WEIGHT[action]

        # 己方在此落子能否五连？
        board[r, c] = player
        if check_winner_from(board, player, r, c, GomokuEnv.CONNECT):
            board[r, c] = 0
            scored.append((_ORDER_WIN + w, action))
            continue
        board[r, c] = 0

        # 对方在此落子能否五连？（需要堵住）
        board[r, c] = opp
        if check_winner_from(board, opp, r, c, GomokuEnv.CONNECT):
            board[r, c] = 0
            scored.append((_ORDER_BLOCK + w, action))
            continue
        board[r, c] = 0

        scored.append((w, action))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [action for _, action in scored]


def _can_win_now(board, player):
    """
    检查 `player` 是否只要再走一步就能形成五连。
    用于静态搜索中的对手反击检测。
    """
    avail = np.flatnonzero(board.ravel() == 0)
    for action in _relevant_moves(board, avail):
        r, c = _ROW[action], _COL[action]
        board[r, c] = player
        win = check_winner_from(board, player, r, c, GomokuEnv.CONNECT)
        board[r, c] = 0
        if win:
            return True
    return False


def _negamax(board, depth, alpha, beta, player):
    """
    Negamax + Alpha-Beta 剪枝搜索核心。

    统一从 `player`（当前落子方）视角评估，score 始终以 `player` 的
    视角计算——正值对 `player` 有利。递归时取负值翻转视角。

    深度耗尽时自动进入静态搜索：若局面存在显著威胁则延申一层，
    避免在危险局面提前截断（地平线效应）。

    返回: (score, action)
    """
    avail = np.flatnonzero(board.ravel() == 0)
    if len(avail) == 0:
        return 0, None

    static_score = evaluate(board, player)
    in_quiescence = depth <= 0

    # 静态搜索终止：深度耗尽且局面平静 → 直接返回评估值
    if in_quiescence and (depth < 0 or abs(static_score) < _QUIESCENCE_THRESHOLD):
        return static_score, None

    opp = 3 - player
    search = _relevant_moves(board, avail)
    best_score = static_score if in_quiescence else -float('inf')
    best_action = search[0]

    for action in _order_moves(board, search, player):
        r, c = _ROW[action], _COL[action]
        board[r, c] = player

        # 直接获胜 → 立即返回
        if check_winner_from(board, player, r, c, GomokuEnv.CONNECT):
            board[r, c] = 0
            return _WIN_SCORE, action

        # 静态搜索中：跳过会被对手立刻反击取胜的落子
        if in_quiescence and _can_win_now(board, opp):
            board[r, c] = 0
            continue

        # Negamax：递归搜索对手视角，取负值转回己方视角
        next_depth = -1 if in_quiescence else depth - 1
        val, _ = _negamax(board, next_depth, -beta, -alpha, opp)
        val = -val
        board[r, c] = 0

        if val > best_score:
            best_score = val
            best_action = action

        alpha = max(alpha, val)
        if beta <= alpha:
            break

    return best_score, best_action


def _iterative_deepening(board, player, max_depth, time_limit):
    """
    迭代加深搜索：从深度 1 开始逐步加深，受 time_limit 时间限制。

    每层保留结果，超时时退回到上一层。根据上一层耗时预估本层是否
    能在剩余时间内完成，提前终止。

    注：调用方（_search / minimax_opponent）已处理 _trivial_move，此处不重复检查。
    """
    t0 = time.perf_counter()
    avail = np.flatnonzero(board.ravel() == 0)
    best_action = int(avail[0])
    last_elapsed = 0

    for d in range(1, max_depth + 1):
        # 若上一层耗时 × 6 已超出剩余时间，预估本层无法完成
        if d > 1 and last_elapsed * 6 > time_limit - (time.perf_counter() - t0):
            break

        _, action = _negamax(board, d, -float('inf'), float('inf'), player)
        if action is not None:
            best_action = int(action)

        last_elapsed = time.perf_counter() - t0
        if last_elapsed > time_limit * 0.5:
            break

    return best_action


def minimax_opponent(board, player, depth=2):
    """
    环境对手回调函数，符合 (board, player) -> action 接口。
    供 gym 环境直接调用，使用固定深度搜索。
    """
    trivial = _trivial_move(board)
    if trivial is not None:
        return trivial
    _, action = _negamax(board, depth, -float('inf'), float('inf'), player)
    return int(action) if action is not None else 0


class MinimaxPlayer:
    """
    Minimax AI 玩家类，兼容 arena.Player 接口。

    用法：
      MinimaxPlayer(depth=3)              → 固定 3 层搜索
      MinimaxPlayer(depth=4, time_limit=1.0) → 最多搜 4 层，总时限 1 秒
    """

    def __init__(self, depth=3, time_limit=None):
        self._depth = depth
        self._time_limit = time_limit
        self._name = f"minimax_d{depth}"

    @property
    def name(self):
        return self._name

    def reset(self):
        """无状态 AI，无需重置。"""
        pass

    def predict_first(self, obs128):
        """
        处理首次观测，返回先手落子。
        obs128 是长度为 128 的扁平数组（8×8×2），取前 64 个值解码棋盘。
        """
        n_cells = GomokuEnv.ROWS * GomokuEnv.COLS
        board = (obs128[:n_cells] * 2.0).astype(np.int8).reshape(GomokuEnv.ROWS, GomokuEnv.COLS)
        return self._search(board, GomokuEnv.PLAYER_1)

    def opponent_callback(self, board, player):
        """对手落子后的回调，返回 AI 的最佳回应。"""
        return self._search(board, player)

    def _search(self, board, player):
        """
        搜索入口：统一处理 _trivial_move 快速路径，然后根据
        是否设置 time_limit 选择迭代加深或固定深度搜索。
        """
        trivial = _trivial_move(board)
        if trivial is not None:
            return trivial

        if self._time_limit is not None:
            return _iterative_deepening(board, player, self._depth, self._time_limit)

        _, action = _negamax(board, self._depth, -float('inf'), float('inf'), player)
        return int(action) if action is not None else 0
