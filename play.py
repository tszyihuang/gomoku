"""
五子棋浏览器对弈 —— Flask 单文件服务器。
支持 AI 对手: Minimax / MCTS(启发式) / AlphaZero(神经网络)
启动: python play.py
访问: http://<服务器IP>:8080
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, request

from gomoku import GomokuEnv
from 古典算法.minimax import minimax_opponent
from 古典算法.mcts import mcts_opponent

# ==============================================================================
#  AlphaZero 模型加载（启动时一次性加载）
# ==============================================================================

_alphazero_net = None      # PolicyValueNet 实例
_alphazero_device = None   # 推理设备
_alphazero_mcts_sims = 300 # 默认 MCTS 模拟次数

def _init_alphazero(checkpoint_path: str = "checkpoints/alphazero_iter_0200.pth",
                    num_channels: int = 128, num_res_blocks: int = 4,
                    mcts_sims: int = 300, use_gpu: bool = True):
    """
    加载训练好的 AlphaZero 模型，全局只调用一次。

    参数:
        checkpoint_path: 模型权重文件路径
        num_channels:    网络通道数（需与训练时一致）
        num_res_blocks:  残差块数量（需与训练时一致）
        mcts_sims:       MCTS 搜索模拟次数（越大越强，但也越慢）
        use_gpu:         是否使用 GPU 推理
    """
    global _alphazero_net, _alphazero_device, _alphazero_mcts_sims

    import numpy as np
    import torch

    from alphazero import PolicyValueNet, _unwrap_state_dict

    _alphazero_mcts_sims = mcts_sims

    # 设备选择
    if use_gpu and torch.cuda.is_available():
        _alphazero_device = torch.device('cuda')
        torch.set_float32_matmul_precision('high')
    else:
        _alphazero_device = torch.device('cpu')

    # 创建网络
    _alphazero_net = PolicyValueNet(
        num_channels=num_channels, num_res_blocks=num_res_blocks,
    ).to(_alphazero_device)
    _alphazero_net.eval()

    # 加载权重
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=_alphazero_device, weights_only=False)
        saved_sd = _unwrap_state_dict(ckpt['model_state_dict'])
        current_sd = _unwrap_state_dict(_alphazero_net.state_dict())
        filtered = {k: v for k, v in saved_sd.items() if k in current_sd}
        _alphazero_net.load_state_dict(filtered, strict=False)
        iteration = ckpt.get('iteration', '?')
        print(f'[AlphaZero] 已加载模型: {checkpoint_path} (iter={iteration})')
    else:
        print(f'[AlphaZero] 未找到 {checkpoint_path}，使用随机权重（弱 AI）')

    # torch.compile 加速（GPU 模式）
    #  注意：Flask 多线程环境不能使用 mode='reduce-overhead'（CUDA graph 依赖 TLS），
    #  改用 mode='default'，仅做 kernel fusion 不捕获 graph，线程安全。
    if _alphazero_device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            _alphazero_net = torch.compile(_alphazero_net, mode='default')
            dummy = torch.zeros(1, 3, 8, 8, device=_alphazero_device)
            _alphazero_net(dummy)
            print('[AlphaZero] torch.compile 加速已启用 (mode=default, 线程安全)')
        except Exception as e:
            print(f'[AlphaZero] torch.compile 失败: {e}')

    total_params = sum(p.numel() for p in _alphazero_net.parameters())
    print(f'[AlphaZero] 参数量: {total_params:,}, 设备: {_alphazero_device}, '
          f'MCTS={_alphazero_mcts_sims}')

    return _alphazero_net


def alphazero_opponent(board, player, mcts_sims: int | None = None):
    """
    使用 AlphaZero 神经网络 + MCTS 选择落子。

    参数:
        board:     (8, 8) numpy 数组
        player:    1 或 2，当前落子方
        mcts_sims: MCTS 模拟次数（None=使用全局默认值）

    返回:
        action: 落子位置索引 (0~63)
    """
    global _alphazero_net, _alphazero_device, _alphazero_mcts_sims

    import numpy as np

    from alphazero import MCTS, board_to_tensor, get_legal_actions

    if _alphazero_net is None:
        # 懒加载：第一次调用时自动初始化
        _init_alphazero()
        if _alphazero_net is None:
            # 回退到普通 MCTS
            return mcts_opponent(board, player, time_limit=1.0)

    sims = mcts_sims if mcts_sims is not None else _alphazero_mcts_sims

    # 根据设备选择批量模式
    batch_size = 32 if _alphazero_device.type == 'cuda' else 1

    mcts = MCTS(_alphazero_net, n_sim=sims, cpuct=3.0,
                dirichlet_alpha=0.0, dirichlet_eps=0.0,  # 对弈时不加噪声
                batch_size=batch_size)

    # 获取动作概率（temperature=0 → 贪心选择最强手）
    action_probs = mcts.get_action_probs(board, int(player), temperature=0.0)

    # 返回概率最高的动作
    return int(np.argmax(action_probs))


# ==============================================================================
#  Flask 应用
# ==============================================================================

app = Flask(__name__)

# ---------- 全局状态 ----------

_env: GomokuEnv | None = None
_ai_strategy: str = "alphazero"   # "minimax" / "mcts" / "alphazero"
_ai_param: float = 300.0          # minimax→depth, mcts→time_limit(s), alphazero→mcts_sims


# ---------- 尝试在启动时预加载 AlphaZero 模型 ----------

def _auto_load_model():
    """自动查找并加载最新的 AlphaZero checkpoint。"""
    import glob
    checkpoints = sorted(glob.glob("checkpoints/alphazero_iter_*.pth"))
    if checkpoints:
        latest = checkpoints[-1]
        print(f'[启动] 发现 AlphaZero 模型: {latest}')
        try:
            _init_alphazero(latest, mcts_sims=300)
            return True
        except Exception as e:
            print(f'[启动] 加载 AlphaZero 模型失败: {e}')
    else:
        print('[启动] 未找到 AlphaZero checkpoint，将使用随机权重')
        try:
            _init_alphazero(mcts_sims=300)
            return True
        except Exception:
            pass
    return False


# ---------- HTML ----------

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>五子棋</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { display:flex; flex-direction:column; align-items:center;
         background:#f5f0e8; font-family:Arial,sans-serif; padding:20px; }
  h1 { margin-bottom:8px; color:#333; }
  #status { font-size:18px; margin:6px 0 10px; min-height:28px; font-weight:bold; }

  /* 棋盘容器 */
  #board-wrap { position:relative; padding:20px;
                background:#dcb468; border-radius:6px; box-shadow:0 4px 12px rgba(0,0,0,.3); }

  .board-row { display:flex; align-items:center; }

  /* 每个交叉点 */
  .cell { width:32px; height:32px; position:relative; cursor:pointer; }
  /* 网格线 */
  .cell::before { content:''; position:absolute; top:50%; left:0; right:0;
                  height:1px; background:#5a3e1b; }
  .cell::after  { content:''; position:absolute; left:50%; top:0; bottom:0;
                  width:1px; background:#5a3e1b; }
  /* 四边截断 */
  .cell.top::after    { top:50%; }
  .cell.bottom::after { bottom:50%; }
  .cell.left::before  { left:50%; }
  .cell.right::before { right:50%; }

  /* 棋子（子元素盖在网格线上方） */
  .cell .stone { position:absolute; width:80%; height:80%; top:10%; left:10%;
                 border-radius:50%; z-index:1; }
  .cell .stone.black {
    background:radial-gradient(circle at 50% 50%,#222 48%,transparent 50%); }
  .cell .stone.white {
    background:radial-gradient(circle at 50% 50%,#eee 48%,transparent 50%);
    filter:drop-shadow(0 1px 2px rgba(0,0,0,.3)); }

  /* 星位 */
  .cell.star:not(.black):not(.white)::before,
  .cell.star:not(.black):not(.white)::after { z-index:1; }
  .cell.star:not(.black):not(.white) {
      background:radial-gradient(circle at 50% 50%,#5a3e1b 10%,transparent 12%); }

  /* 悬停提示 */
  .cell:hover:not(.black):not(.white) {
    background:radial-gradient(circle at 50% 50%,rgba(0,0,0,.15) 38%,transparent 40%); }

  /* 禁用状态 */
  .cell.disabled { cursor:default; pointer-events:none; }

  .controls { margin-top:14px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  select, button { font-size:15px; padding:6px 14px; border-radius:4px; border:1px solid #aaa; }
  button { cursor:pointer; background:#4a7; color:#fff; border:none; font-weight:bold; }
  button:hover { background:#3a6; }
  select { background:#fff; }
</style>
</head>
<body>
<h1>五子棋</h1>
<div id="status">你的回合，请落子</div>

<div id="board-wrap">
  <div id="board"></div>
</div>

<div class="controls">
  <label>AI：
    <select id="ai_strategy">
      <optgroup label="AlphaZero (神经网络)">
        <option value="az100">AlphaZero sims=100</option>
        <option value="az200">AlphaZero sims=200</option>
        <option value="az300" selected>AlphaZero sims=300</option>
        <option value="az500">AlphaZero sims=500</option>
        <option value="az800">AlphaZero sims=800</option>
      </optgroup>
      <optgroup label="MCTS (启发式)">
        <option value="mcts0.5">MCTS time=0.5s</option>
        <option value="mcts1.0">MCTS time=1.0s</option>
        <option value="mcts2.0">MCTS time=2.0s</option>
      </optgroup>
      <optgroup label="Minimax (搜索)">
        <option value="minimax2">Minimax depth=2</option>
        <option value="minimax3">Minimax depth=3</option>
        <option value="minimax4">Minimax depth=4</option>
      </optgroup>
    </select>
  </label>
  <button onclick="restart()">重新开始</button>
  <span style="font-size:13px;color:#888;" id="model_info"></span>
</div>

<script>
const ROWS=8, COLS=8;
const STARS=new Set();
let board=[]; // 8x8, 0=空 1=人 2=AI
let processing=false; // AI 思考中禁止点击
let generation=0;    // 用来忽略过时的响应
let gameOver=false;
let cells=[];

async function restart(){
  generation++;
  const ai=document.getElementById('ai_strategy').value;
  const r=await fetch('/restart',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ai})});
  const d=await r.json();
  board=d.board;
  processing=false;
  gameOver=false;
  document.getElementById('status').textContent='你的回合，请落子';
  document.getElementById('model_info').textContent=d.model_info||'';
  if(!cells.length) initBoard();
  render();
}

function initBoard(){
  const boardEl=document.getElementById('board');
  boardEl.addEventListener('click',e=>{
    const cell=e.target.closest('.cell');
    if(!cell) return;
    clickCell(+cell.dataset.row,+cell.dataset.col);
  });
  for(let r=0;r<ROWS;r++){
    cells[r]=[];
    const rowDiv=document.createElement('div');
    rowDiv.className='board-row';
    for(let c=0;c<COLS;c++){
      const cell=document.createElement('div');
      cell.className='cell';
      cell.dataset.row=r;
      cell.dataset.col=c;
      if(r===0) cell.classList.add('top');
      if(r===ROWS-1) cell.classList.add('bottom');
      if(c===0) cell.classList.add('left');
      if(c===COLS-1) cell.classList.add('right');
      if(STARS.has(r*COLS+c)) cell.classList.add('star');
      const stone=document.createElement('span');
      stone.className='stone';
      cell.appendChild(stone);
      rowDiv.appendChild(cell);
      cells[r][c]=cell;
    }
    boardEl.appendChild(rowDiv);
  }
}

function render(){
  for(let r=0;r<ROWS;r++){
    for(let c=0;c<COLS;c++){
      const cell=cells[r][c];
      const stone=cell.querySelector('.stone');
      cell.classList.remove('black','white','disabled');
      stone.classList.remove('black','white');
      if(board[r][c]===1){ cell.classList.add('black'); stone.classList.add('black'); }
      else if(board[r][c]===2){ cell.classList.add('white'); stone.classList.add('white'); }
      if(processing || gameOver) cell.classList.add('disabled');
    }
  }
}

async function clickCell(r,c){
  if(board[r][c]!==0 || processing || gameOver) return;
  const statusEl=document.getElementById('status');

  board[r][c]=1;
  processing=true;
  const myGen=generation;
  render();
  statusEl.textContent='AI 思考中...';

  try {
    const resp=await fetch('/move',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({row:r,col:c})});
    if(!resp.ok) throw new Error('HTTP '+resp.status);
    const d=await resp.json();
    if(generation!==myGen) return;  // 被 restart 中断，丢弃响应
    if(d.error) throw new Error(d.error);
    board=d.board;

    if(d.status==='human_win'){ statusEl.textContent='🎉 你赢了！'; gameOver=true; }
    else if(d.status==='ai_win'){ statusEl.textContent='😞 AI 赢了。'; gameOver=true; }
    else if(d.status==='draw'){ statusEl.textContent='🤝 平局！'; gameOver=true; }
    else statusEl.textContent='你的回合，请落子';
  } catch(e) {
    console.error(e);
    statusEl.textContent='出错了，请重新开始';
  } finally {
    if(generation===myGen){ processing=false; render(); }
  }
}

// 初始化
restart();
</script>
</body>
</html>"""


# ---------- 路由 ----------

@app.route("/")
def index():
    return HTML


@app.route("/restart", methods=["POST"])
def restart():
    global _env, _ai_strategy, _ai_param
    data = request.get_json()
    ai = data.get("ai", "az300")

    if ai.startswith("az"):
        _ai_strategy = "alphazero"
        _ai_param = float(ai[len("az"):])  # MCTS 模拟次数
    elif ai.startswith("minimax"):
        _ai_strategy = "minimax"
        _ai_param = int(ai[len("minimax"):])
    else:
        _ai_strategy = "mcts"
        _ai_param = float(ai[len("mcts"):])

    _env = GomokuEnv()
    _env.reset()

    # 返回模型信息
    model_info = ""
    if _ai_strategy == "alphazero":
        if _alphazero_net is not None:
            model_info = f"AlphaZero | MCTS={int(_ai_param)} | {_alphazero_device}"
        else:
            model_info = f"AlphaZero (随机权重) | MCTS={int(_ai_param)}"

    return jsonify(board=_env.board.tolist(), status="playing", model_info=model_info)


def _step_and_status(action, win_status):
    """执行一步，若终局返回响应，否则返回 (obs, None)。"""
    obs, reward, terminated, _, _ = _env.step(action)
    if not terminated:
        return obs, None
    status = win_status if reward > 0.5 else "draw"
    return obs, jsonify(board=obs.tolist(), status=status)


@app.route("/move", methods=["POST"])
def move():
    if _env is None:
        return jsonify(error="game not started"), 400

    data = request.get_json()
    row, col = data["row"], data["col"]
    action = row * GomokuEnv.COLS + col

    obs, resp = _step_and_status(action, "human_win")
    if resp:
        return resp

    # ---- AI 走子 ----
    if _ai_strategy == "minimax":
        ai_action = minimax_opponent(obs, GomokuEnv.PLAYER_2, depth=int(_ai_param))
    elif _ai_strategy == "mcts":
        ai_action = mcts_opponent(obs, GomokuEnv.PLAYER_2, time_limit=_ai_param)
    else:  # alphazero
        ai_action = alphazero_opponent(obs, GomokuEnv.PLAYER_2, mcts_sims=int(_ai_param))

    obs, resp = _step_and_status(ai_action, "ai_win")
    return resp or jsonify(board=obs.tolist(), status="playing")


# ---------- 启动 ----------

if __name__ == "__main__":
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    print("=" * 50)
    print("  五子棋 Web 服务器")
    print("=" * 50)
    _auto_load_model()
    print("=" * 50)
    print("  访问: http://0.0.0.0:8080")
    print("=" * 50)

    app.run(host="0.0.0.0", port=8080, debug=False)
