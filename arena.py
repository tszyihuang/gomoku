"""
Gomoku 模型试炼场 —— 选择双方模型自动对弈，统计胜率。

运行方式：
  python arena.py                    交互菜单
  python arena.py --list             列出可用模型
  python arena.py --a <model> --b <model> --episodes 200
  python arena.py --a <model> --b <model> --episodes 200 --workers 8
"""

from __future__ import annotations

import glob
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from gomoku import GomokuEnv
from minimax import MinimaxPlayer
from mcts import MCTSPlayer
from alphazero import PolicyValueNet, MCTS as AlphaZeroMCTS, _unwrap_state_dict


# =============================================================================
# Model discovery
# =============================================================================

def discover_models() -> list[str]:
    """Return sorted list of available model names."""
    models = ["random",
              "minimax_d2", "minimax_d3", "minimax_d4",
              "mcts_t0.5", "mcts_t1.0", "mcts_t2.0"]

    # ---- AlphaZero 模型 ----
    # 自动发现 checkpoints 目录下的模型文件
    az_checkpoints = sorted(glob.glob("checkpoints/alphazero_iter_*.pth"))
    if az_checkpoints:
        # 最新 checkpoint + 多档 MCTS 模拟次数
        for sims in [100, 200, 400, 800]:
            models.append(f"az_s{sims}")

    return models


# =============================================================================
# Player interface & implementations
# =============================================================================

class Player:
    """Unified interface for game-playing agents."""

    def reset(self):
        pass

    def select_action(self, board: np.ndarray, player: int) -> int:
        """Given the current board and player number, return an action."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class RandomPlayer(Player):
    name = "random"

    def select_action(self, board: np.ndarray, player: int) -> int:
        avail = np.flatnonzero(board.ravel() == 0)
        return int(random.choice(avail)) if len(avail) > 0 else 0


class _MinimaxAdapter(Player):
    """Thin adapter so MinimaxPlayer satisfies the arena Player interface."""

    def __init__(self, mp: MinimaxPlayer):
        self._mp = mp

    @property
    def name(self) -> str:
        return self._mp.name

    def reset(self):
        self._mp.reset()

    def select_action(self, board: np.ndarray, player: int) -> int:
        return self._mp.opponent_callback(board, player)


class AlphaZeroPlayer(Player):
    """AlphaZero 神经网络 + MCTS 玩家。

    加载训练好的 PolicyValueNet 权重，使用纯神经网络引导的
    MCTS 搜索来选择落子。

    优化策略：
      - BatchNorm 融合：消除推理时的 BN 计算，提速 ~30%
      - 批量 MCTS：CPU 下用 batch_size=16 + 虚拟损失，提速 ~5×
      - 默认使用 CPU 推理以保证多进程兼容
    """

    def __init__(self, checkpoint_path: str, mcts_sims: int = 400,
                 device: str = 'cpu', num_channels: int = 128,
                 num_res_blocks: int = 4):
        self._mcts_sims = mcts_sims
        self._device = torch.device(device)

        # 从 checkpoint 文件名提取迭代数
        basename = os.path.basename(checkpoint_path)
        if basename.startswith("alphazero_iter_") and basename.endswith(".pth"):
            iter_str = basename[len("alphazero_iter_"):-len(".pth")]
            self._iteration = int(iter_str)
        else:
            self._iteration = 0

        self._name = f"az_i{self._iteration:04d}_s{self._mcts_sims}"

        # 加载网络
        self._net = PolicyValueNet(
            num_channels=num_channels, num_res_blocks=num_res_blocks,
        ).to(self._device)
        self._net.eval()

        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        saved_sd = _unwrap_state_dict(ckpt['model_state_dict'])
        current_sd = _unwrap_state_dict(self._net.state_dict())
        filtered = {k: v for k, v in saved_sd.items() if k in current_sd}
        self._net.load_state_dict(filtered, strict=False)

        # ---- 推理优化 ----
        # 1. BatchNorm 融合：将 BN 参数合并到前置卷积层
        from alphazero import fuse_all_batchnorms
        self._net = fuse_all_batchnorms(self._net)

        # 2. 批量 MCTS：CPU 下用 batch 推理 + 虚拟损失大幅减少 Python 开销
        #    GPU 用更大的 batch 充分利用并行能力
        if self._device.type == 'cuda':
            self._batch_size = 32
            self._virtual_loss = 3.0
        else:
            self._batch_size = 16
            self._virtual_loss = 3.0

    @property
    def name(self) -> str:
        return self._name

    def select_action(self, board: np.ndarray, player: int) -> int:
        mcts = AlphaZeroMCTS(self._net, n_sim=self._mcts_sims, cpuct=3.0,
                             dirichlet_alpha=0.0, dirichlet_eps=0.0,
                             batch_size=self._batch_size,
                             virtual_loss=self._virtual_loss)
        action_probs = mcts.get_action_probs(board, int(player), temperature=0.0)
        return int(np.argmax(action_probs))


# =============================================================================
# Player factory
# =============================================================================

_CACHE: dict[str, Player] = {}


def get_player(model_name: str) -> Player:
    """Return a (possibly cached) Player for the given model name."""
    if model_name in _CACHE:
        return _CACHE[model_name]

    if model_name == "random":
        p = RandomPlayer()
    elif model_name.startswith("minimax_d"):
        depth = int(model_name.split("d")[-1])
        time_limits = {2: None, 3: 0.3, 4: 0.5}
        p = _MinimaxAdapter(MinimaxPlayer(depth=depth, time_limit=time_limits.get(depth)))
    elif model_name.startswith("mcts_t"):
        time_limit = float(model_name.split("t")[-1])
        p = MCTSPlayer(time_limit=time_limit)
    elif model_name.startswith("az_"):
        p = _make_alphazero_player(model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    _CACHE[model_name] = p
    return p


def _find_latest_checkpoint() -> str | None:
    """返回最新的 AlphaZero checkpoint 路径，若没有则返回 None。"""
    checkpoints = sorted(glob.glob("checkpoints/alphazero_iter_*.pth"))
    return checkpoints[-1] if checkpoints else None


def _make_alphazero_player(model_name: str) -> AlphaZeroPlayer:
    """根据模型名称创建 AlphaZero 玩家。

    支持的格式：
      az_s{sims}          —— 最新 checkpoint + 指定 MCTS 模拟次数
      az_i{iter}_s{sims}  —— 指定 checkpoint 迭代数 + MCTS 模拟次数
    示例：
      az_s400             —— 最新模型, 400 次 MCTS 模拟
      az_i0200_s400       —— iter=200 的模型, 400 次 MCTS 模拟
    """
    import re

    # 解析模型名称
    match = re.match(r'^az_i(\d+)_s(\d+)$', model_name)
    if match:
        iteration = int(match.group(1))
        sims = int(match.group(2))
        checkpoint_path = f"checkpoints/alphazero_iter_{iteration:04d}.pth"
        if not os.path.exists(checkpoint_path):
            raise ValueError(
                f"Checkpoint 不存在: {checkpoint_path}\n"
                f"  可用的 checkpoint 迭代数: {_available_iterations()}"
            )
    else:
        match = re.match(r'^az_s(\d+)$', model_name)
        if match:
            sims = int(match.group(1))
            checkpoint_path = _find_latest_checkpoint()
            if checkpoint_path is None:
                raise ValueError("未找到任何 AlphaZero checkpoint，请先训练模型")
        else:
            raise ValueError(f"未知的 AlphaZero 模型名: {model_name}")

    return AlphaZeroPlayer(checkpoint_path, mcts_sims=sims)


def _available_iterations() -> list[int]:
    """返回所有可用 checkpoint 的迭代数列表。"""
    checkpoints = sorted(glob.glob("checkpoints/alphazero_iter_*.pth"))
    iterations = []
    for cp in checkpoints:
        basename = os.path.basename(cp)
        try:
            iter_str = basename[len("alphazero_iter_"):-len(".pth")]
            iterations.append(int(iter_str))
        except ValueError:
            pass
    return iterations


# =============================================================================
# Arena engine
# =============================================================================

@dataclass
class ArenaResult:
    model_a: str
    model_b: str
    episodes: int
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    wins_a_first: int = 0
    wins_a_second: int = 0
    wins_b_first: int = 0
    wins_b_second: int = 0
    elapsed: float = 0.0

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.episodes if self.episodes > 0 else 0.0

    def __str__(self) -> str:
        n = self.episodes
        h = n // 2
        h_a = h + (1 if n % 2 else 0)
        h_b = h
        safe_pct = lambda num, den: (num / den * 100) if den > 0 else 0.0
        return (
            f"\n{'=' * 52}\n"
            f"  Arena: {self.model_a}  vs  {self.model_b}\n"
            f"  Episodes: {n}  ({h_a} A first, {h_b} B first)\n"
            f"{'=' * 52}\n"
            f"  {self.model_a}\n"
            f"    Total wins:  {self.wins_a:>5}  ({self.win_rate_a * 100:5.1f}%)\n"
            f"    As first:    {self.wins_a_first:>5}  ({safe_pct(self.wins_a_first, h_a):5.1f}%)\n"
            f"    As second:   {self.wins_a_second:>5}  ({safe_pct(self.wins_a_second, h_b):5.1f}%)\n"
            f"  {self.model_b}\n"
            f"    Total wins:  {self.wins_b:>5}  ({safe_pct(self.wins_b, n):5.1f}%)\n"
            f"    As first:    {self.wins_b_first:>5}  ({safe_pct(self.wins_b_first, h_b):5.1f}%)\n"
            f"    As second:   {self.wins_b_second:>5}  ({safe_pct(self.wins_b_second, h_a):5.1f}%)\n"
            f"  Draws:         {self.draws:>5}  ({safe_pct(self.draws, n):5.1f}%)\n"
            f"  Time: {self.elapsed:.1f}s\n"
            f"{'=' * 52}"
        )


def _play_one_game(player_first: Player, player_second: Player) -> int:
    """Play one game. Returns 1 if first wins, -1 if second wins, 0 for draw."""
    player_first.reset()
    player_second.reset()

    env = GomokuEnv()
    obs, _ = env.reset()
    terminated = False

    while not terminated:
        action = player_first.select_action(obs, GomokuEnv.PLAYER_1)
        obs, reward, terminated, _, _ = env.step(action)
        if terminated:
            return 1 if reward > 0.5 else 0

        action = player_second.select_action(obs, GomokuEnv.PLAYER_2)
        obs, reward, terminated, _, _ = env.step(action)
        if terminated:
            return -1 if reward > 0.5 else 0

    return 0


def _play_one_game_task(args):
    """Top-level task for parallel execution."""
    model_a, model_b, a_first = args
    p_a = get_player(model_a)
    p_b = get_player(model_b)
    if a_first:
        return _play_one_game(p_a, p_b), True
    outcome = _play_one_game(p_b, p_a)
    return -outcome, False


def run_arena(model_a: str, model_b: str, episodes: int = 200,
              workers: int = 1) -> ArenaResult:
    """Pit two models against each other and return the result."""
    player_a = get_player(model_a)
    player_b = get_player(model_b)

    result = ArenaResult(
        model_a=player_a.name,
        model_b=player_b.name,
        episodes=episodes,
    )

    half = episodes // 2
    tasks = [(model_a, model_b, True) for _ in range(half + episodes % 2)]
    tasks += [(model_a, model_b, False) for _ in range(half)]

    n_workers = workers if workers > 0 else (os.cpu_count() or 4)
    n_tasks = len(tasks)

    t0 = time.perf_counter()

    if n_workers <= 1:
        for idx, task in enumerate(tasks):
            outcome, a_first = _play_one_game_task(task)
            _accumulate_result(result, outcome, a_first)
            if (idx + 1) % 50 == 0:
                print(f"  [{idx + 1}/{n_tasks}] ...")
    else:
        from multiprocessing import get_context
        n_workers = min(n_workers, n_tasks)
        ctx = get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            for idx, (outcome, a_first) in enumerate(pool.imap_unordered(
                    _play_one_game_task, tasks, chunksize=max(1, n_tasks // (n_workers * 4)))):
                _accumulate_result(result, outcome, a_first)
                done = result.wins_a + result.wins_b + result.draws
                if done % 50 == 0:
                    print(f"  [{done}/{n_tasks}] ...")

    result.elapsed = time.perf_counter() - t0
    return result


def _accumulate_result(result: ArenaResult, outcome: int, a_first: bool):
    if outcome == 1:
        result.wins_a += 1
        if a_first:
            result.wins_a_first += 1
        else:
            result.wins_a_second += 1
    elif outcome == -1:
        result.wins_b += 1
        if a_first:
            result.wins_b_second += 1
        else:
            result.wins_b_first += 1
    else:
        result.draws += 1


# =============================================================================
# Interactive menu
# =============================================================================

def _pick_model(prompt: str) -> str | None:
    models = discover_models()
    print()
    print(f"  {prompt}")
    print()
    print("  ── Random ──")
    print("    [1] random")
    print()
    print("  ── Minimax ──")
    minimax_models = [m for m in models if m.startswith("minimax_")]
    for i, name in enumerate(minimax_models):
        print(f"    [{i + 2:>2}] {name}")
    print()
    print("  ── MCTS ──")
    mcts_models = [m for m in models if m.startswith("mcts_")]
    offset = 2 + len(minimax_models)
    for i, name in enumerate(mcts_models):
        print(f"    [{i + offset:>2}] {name}")
    print()
    print("  ── AlphaZero (神经网络) ──")
    az_models = [m for m in models if m.startswith("az_")]
    az_offset = offset + len(mcts_models)
    for i, name in enumerate(az_models):
        print(f"    [{i + az_offset:>2}] {name}")
    print()
    print(f"    [b] 返回")

    model_list = ["random"] + minimax_models + mcts_models + az_models

    while True:
        ch = input("  选模型 > ").strip()
        if ch.lower() in ("b", "back", ""):
            return None
        try:
            idx = int(ch) - 1
            if 0 <= idx < len(model_list):
                return model_list[idx]
        except ValueError:
            pass
        if ch in model_list:
            return ch
        print(f"  无效选项，请输入 1~{len(model_list)} 或模型名")


def _interactive():
    print()
    print("╔══════════════════════════════════════╗")
    print("║       五 子 棋 模 型 试 炼 场        ║")
    print("╚══════════════════════════════════════╝")

    model_a = _pick_model("选择 模型 A（己方）")
    if model_a is None:
        print("  已取消。")
        return

    model_b = _pick_model("选择 模型 B（对手）")
    if model_b is None:
        print("  已取消。")
        return

    raw = input("\n  对弈局数（默认 200）> ").strip()
    episodes = int(raw) if raw.isdigit() and int(raw) > 0 else 200

    workers = 0
    print(f"\n  {model_a}  vs  {model_b}  × {episodes}  (workers={workers})")
    print("  开始对弈...")

    result = run_arena(model_a, model_b, episodes, workers=workers)
    print(result)


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Gomoku 模型试炼场")
    p.add_argument("--list", action="store_true", help="列出所有可用模型")
    p.add_argument("--a", default=None, help="模型 A（己方）")
    p.add_argument("--b", default=None, help="模型 B（对手）")
    p.add_argument("--episodes", type=int, default=200, help="对弈局数（默认 200）")
    p.add_argument("--workers", type=int, default=0,
                   help="并行 worker 数（默认自动检测 CPU 核心数）")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.list:
        print("可用模型：")
        for name in discover_models():
            print(f"  {name}")
        return

    if args.a and args.b:
        result = run_arena(args.a, args.b, args.episodes, workers=args.workers)
        print(result)
        return

    if args.a or args.b:
        print("Error: --a 和 --b 必须同时指定（或都不指定进入交互模式）")
        raise SystemExit(1)

    _interactive()


if __name__ == "__main__":
    main()
