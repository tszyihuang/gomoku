"""
MCTS (Monte Carlo Tree Search) 策略，用于 15×15 五子棋。

纯启发式叶节点评估 + MCTS 树搜索。用 minimax 评估函数代替随机 rollout，
以极快的速度（数千迭代/秒）搜索更深层战术威胁。
"""

from __future__ import annotations

import math
import time

import numpy as np
from gomoku import GomokuEnv, check_winner_from
from minimax import evaluate as _heuristic_eval, _ROW, _COL, _NEAR_MASK, _trivial_move

# 启发式评估缩放
_EVAL_SCALE = 500_000.0


def _relevant_moves(board):
    """返回所有靠近已有棋子的空位列表。"""
    flat = board.ravel()
    occupied = np.flatnonzero(flat != 0)
    if len(occupied) == 0:
        return [(GomokuEnv.ROWS // 2) * GomokuEnv.COLS + (GomokuEnv.COLS // 2)]
    near = _NEAR_MASK[occupied].any(axis=0)
    avail = np.flatnonzero(flat == 0)
    result = [int(a) for a in avail if near[a]]
    return result if result else [int(a) for a in avail]


def _quick_win_check(board, player, action):
    """检查 `player` 在 `action` 落子后是否立即五连（不修改棋盘）。"""
    r, c = _ROW[action], _COL[action]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            for i in range(1, GomokuEnv.CONNECT):
                nr, nc = r + sign * i * dr, c + sign * i * dc
                if 0 <= nr < GomokuEnv.ROWS and 0 <= nc < GomokuEnv.COLS and board[nr, nc] == player:
                    count += 1
                else:
                    break
        if count >= GomokuEnv.CONNECT:
            return True
    return False


def _init_untried(board, player, visits=0):
    """
    为一个节点生成候选动作列表。
    优先级：直接获胜 > 堵住对方获胜 > 局部模式评分排序。
    加入渐进式展开：访问数越多，候选动作越多，防止树过宽。
    """
    moves = _relevant_moves(board)
    opp = 3 - player

    wins = [a for a in moves if _quick_win_check(board, player, a)]
    if wins:
        return wins

    blocks = [a for a in moves if _quick_win_check(board, opp, a)]
    if blocks:
        return blocks

    # 局部模式评分排序
    scored = [(_pattern_score(board, player, a) + _pattern_score(board, opp, a), a)
              for a in moves]
    scored.sort(reverse=True, key=lambda x: x[0])

    # 渐进式展开：限制候选数，迫使树在关键分支上深入
    limit = min(len(scored), int(math.sqrt(visits + 1)) * 3 + 6)
    return [a for _, a in scored[:limit]]


def _pattern_score(board, player, action):
    """
    估算 `player` 在 `action` 落子后在该位置形成的局部模式价值。
    只沿 4 个方向从 (r,c) 向两端搜索，不扫描全盘。
    """
    r, c = _ROW[action], _COL[action]
    total = 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        # 分别统计正/反方向的连续子数
        cnt_p = 0
        for i in range(1, 5):
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < GomokuEnv.ROWS and 0 <= nc < GomokuEnv.COLS and board[nr, nc] == player:
                cnt_p += 1
            else:
                break
        cnt_m = 0
        for i in range(1, 5):
            nr, nc = r - i * dr, c - i * dc
            if 0 <= nr < GomokuEnv.ROWS and 0 <= nc < GomokuEnv.COLS and board[nr, nc] == player:
                cnt_m += 1
            else:
                break
        run = 1 + cnt_p + cnt_m

        # 两端开放数
        opens = 0
        nr, nc = r + (cnt_p + 1) * dr, c + (cnt_p + 1) * dc
        if 0 <= nr < GomokuEnv.ROWS and 0 <= nc < GomokuEnv.COLS and board[nr, nc] == 0:
            opens += 1
        nr, nc = r - (cnt_m + 1) * dr, c - (cnt_m + 1) * dc
        if 0 <= nr < GomokuEnv.ROWS and 0 <= nc < GomokuEnv.COLS and board[nr, nc] == 0:
            opens += 1

        if run >= 5:
            total += 1_000_000
        elif run == 4:
            total += 100_000 if opens >= 1 else 5_000
        elif run == 3:
            total += 5_000 if opens == 2 else 500
        elif run == 2:
            total += 100 if opens == 2 else 10
    return total


# ---------- MCTS Node ----------

class _Node:
    __slots__ = ('parent', 'action', 'player', 'visits', 'value',
                 'children', 'untried')

    def __init__(self, player, action=None, parent=None):
        self.parent = parent
        self.action = action
        self.player = player
        self.visits = 0
        self.value = 0.0
        self.children: dict[int, _Node] = {}
        self.untried: list[int] | None = None

    @property
    def is_expanded(self):
        return self.untried is not None and len(self.untried) == 0


# ---------- Leaf Evaluation ----------

def _leaf_value(board, player):
    """叶节点评估：用 minimax 启发式函数打分并映射到 [-1, 1]。"""
    raw = _heuristic_eval(board, player)
    return math.tanh(raw / _EVAL_SCALE)


# ---------- MCTS Search ----------

_MAX_TREE_DEPTH = 7


def _resolve_result(board, node, won=False):
    """统一叶节点结果：获胜 → -1, 平局 → 0, 否则 → 启发式评估。"""
    if won:
        return -1.0
    if np.all(board.ravel() != 0):
        return 0.0
    return _leaf_value(board, node.player)


def mcts_search(board, player, time_limit=1.0, exploration=1.414):
    """
    从 `board` / `player` 出发运行 MCTS，时限 `time_limit` 秒，
    返回最佳动作（落子索引）。
    """
    trivial = _trivial_move(board)
    if trivial is not None:
        return trivial

    root = _Node(player)
    root.untried = _init_untried(board, player)

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < time_limit:
        # 1. Selection
        node = root
        b = board.copy()
        path = [node]

        while node.is_expanded and node.children:
            node = _select_child(node, exploration)
            r, c = _ROW[node.action], _COL[node.action]
            b[r, c] = node.parent.player
            path.append(node)

        # 2. Expansion（超过最大深度则直接评估，不展开）
        if len(path) >= _MAX_TREE_DEPTH:
            result = _leaf_value(b, node.player)
        else:
            if node.untried is None:
                node.untried = _init_untried(b, node.player, node.visits)

            if node.untried:
                action = node.untried.pop()
                r, c = _ROW[action], _COL[action]
                b[r, c] = node.player
                won = check_winner_from(b, node.player, r, c, GomokuEnv.CONNECT)
                child = _Node(3 - node.player, action=action, parent=node)
                node.children[action] = child
                node = child
                path.append(node)
                result = _resolve_result(b, node, won)
            else:
                result = _resolve_result(b, node)

        # 3. Backpropagation
        for n in reversed(path):
            n.visits += 1
            n.value += result
            result = -result

    # 返回访问次数最多的动作
    if not root.children:
        return root.untried[0] if root.untried else 0

    best = max(root.children.items(), key=lambda x: x[1].visits)
    return int(best[0])


def _select_child(node, exploration):
    """UCB1 选择。child.value 从 child.player（对手）视角，取负号转回。"""
    best_child = None
    best_ucb = -float('inf')
    log_n = math.log(node.visits + 1)

    for child in node.children.values():
        if child.visits == 0:
            ucb = float('inf')
        else:
            exploit = -child.value / child.visits
            explore = exploration * math.sqrt(log_n / child.visits)
            ucb = exploit + explore
        if ucb > best_ucb:
            best_ucb = ucb
            best_child = child

    return best_child


# ---------- 兼容接口 ----------

def mcts_opponent(board, player, time_limit=1.0):
    """供 play.py 使用的回调函数，兼容 (board, player) -> action 签名。"""
    return mcts_search(board, player, time_limit=time_limit)


class MCTSPlayer:
    """MCTS AI 玩家，兼容 arena.Player 接口。"""

    def __init__(self, time_limit=1.0):
        self._time_limit = time_limit
        self._name = f"mcts_t{time_limit:.1f}"

    @property
    def name(self):
        return self._name

    def reset(self):
        pass

    def select_action(self, board, player):
        return mcts_search(board, player, time_limit=self._time_limit)
