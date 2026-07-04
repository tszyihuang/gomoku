"""
五子棋 Gymnasium 环境 (15×15)
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

def check_winner_from(board: np.ndarray, player: int, r: int, c: int,
                      connect: int) -> bool:
    """检查 (r, c) 落子后 player 是否连成 connect 子"""
    rows, cols = board.shape
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        count = 1
        for sign in (-1, 1):
            for i in range(1, connect):
                nr, nc = r + sign * i * dr, c + sign * i * dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr, nc] == player:
                    count += 1
                else:
                    break
        if count >= connect:
            return True
    return False

class GomokuEnv(gym.Env):
    """五子棋 Gymnasium 环境。每步走一子，自动交替执棋。"""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    ROWS = 15
    COLS = 15
    CONNECT = 5

    EMPTY = 0
    PLAYER_1 = 1
    PLAYER_2 = 2

    _CHARS = {0: " ", 1: "X", 2: "O"}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.ROWS, self.COLS), dtype=np.int8,
        )
        self.action_space = spaces.Discrete(self.ROWS * self.COLS)
        self.render_mode = render_mode

        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = self.PLAYER_1
        self._empty_count = self.ROWS * self.COLS
        self._terminated = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = self.PLAYER_1
        self._empty_count = self.ROWS * self.COLS
        self._terminated = False
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("step() called after episode termination")
        if not (0 <= action < self.ROWS * self.COLS):
            raise ValueError(f"action {action} out of range [0, {self.ROWS * self.COLS})")
        row, col = divmod(action, self.COLS)
        # 非法走子：随机选空位并惩罚
        penalty = 0.0
        if self.board[row, col] != self.EMPTY:
            avail = np.flatnonzero(self.board.ravel() == self.EMPTY)
            action = int(self.np_random.choice(avail))
            row, col = divmod(action, self.COLS)
            penalty = -0.1
        player = self.current_player
        self.board[row, col] = player
        self._empty_count -= 1

        # 胜负判定
        if check_winner_from(self.board, player, row, col, self.CONNECT):
            self._terminated = True
            return self._get_obs(), 1.0 + penalty, True, False, self._get_info()

        # 平局判定
        if self._empty_count == 0:
            self._terminated = True
            return self._get_obs(), penalty, True, False, self._get_info()

        # 切换执棋方
        self.current_player = self.PLAYER_2 if player == self.PLAYER_1 else self.PLAYER_1
        return self._get_obs(), penalty, False, False, self._get_info()

    def render(self):
        col_header = "     " + "  ".join(f"{c:<2d}" for c in range(self.COLS))
        lines = [col_header]
        lines.append("   ┌" + "───┬" * (self.COLS - 1) + "───┐")
        for r in range(self.ROWS):
            row = f"{r:2d} │ " + " │ ".join(self._CHARS[c] for c in self.board[r]) + " │"
            lines.append(row)
            if r < self.ROWS - 1:
                lines.append("   ├" + "───┼" * (self.COLS - 1) + "───┤")
        lines.append("   └" + "───┴" * (self.COLS - 1) + "───┘")
        out = "\n".join(lines)
        if self.render_mode == "human":
            print(out)
        return out

    def close(self):
        pass

    def _get_obs(self):
        return self.board.copy()

    def _get_info(self):
        return {"current_player": self.current_player}