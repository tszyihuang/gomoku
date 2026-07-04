"""
五子棋浏览器对弈 —— Flask 单文件服务器。
启动: python web_play.py
访问: http://<服务器IP>:8080
"""

from __future__ import annotations


from flask import Flask, jsonify, request

from gomoku import GomokuEnv
from minimax import minimax_opponent
from mcts import mcts_opponent

app = Flask(__name__)

# ---------- 全局状态 ----------

_env: GomokuEnv | None = None
_ai_strategy: str = "mcts"      # "minimax" / "mcts"
_ai_param: float = 1.0          # minimax 用 depth(int), mcts 用 time_limit(float)

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

  .controls { margin-top:14px; display:flex; gap:12px; align-items:center; }
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
      <option value="minimax2">Minimax depth=2</option>
      <option value="minimax3">Minimax depth=3</option>
	      <option value="minimax4">Minimax depth=4</option>
      <option value="mcts0.5">MCTS time=0.5s</option>
      <option value="mcts1.0" selected>MCTS time=1.0s</option>
      <option value="mcts2.0">MCTS time=2.0s</option>
    </select>
  </label>
  <button onclick="restart()">重新开始</button>
</div>

<script>
const ROWS=15, COLS=15;
// 星位
const STARS=new Set([3,7,11].flatMap(r=>[3,7,11].map(c=>r*COLS+c)));
let board=[]; // 15x15, 0=空 1=人 2=AI
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
    ai = data.get("ai", "mcts1.0")
    if ai.startswith("minimax"):
        _ai_strategy = "minimax"
        _ai_param = int(ai[len("minimax"):])
    else:
        _ai_strategy = "mcts"
        _ai_param = float(ai[len("mcts"):])
    _env = GomokuEnv()
    _env.reset()
    return jsonify(board=_env.board.tolist(), status="playing")


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

    if _ai_strategy == "minimax":
        ai_action = minimax_opponent(obs, GomokuEnv.PLAYER_2, depth=int(_ai_param))
    else:
        ai_action = mcts_opponent(obs, GomokuEnv.PLAYER_2, time_limit=_ai_param)
    obs, resp = _step_and_status(ai_action, "ai_win")
    return resp or jsonify(board=obs.tolist(), status="playing")


if __name__ == "__main__":
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    print("五子棋 Web 服务器启动: http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
