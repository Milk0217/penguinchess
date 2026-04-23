"""
Flask API 服务端：企鹅棋前后端分离架构后端。
所有游戏逻辑在 PenguinChessCore（Python）执行，前端仅负责渲染和用户交互。

启动方式:
    cd /mnt/e/programming/penguinchess
    source .venv/bin/activate
    python server/app.py

前端开发（Vite）:
    cd frontend && npm run dev
"""

from __future__ import annotations

import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from .game import create_session, get_session
from .boards import list_boards, get_board, save_board, delete_board

app = Flask(__name__, static_folder="../frontend/dist", static_url_path="")
CORS(app)  # 开发环境允许跨域


# =============================================================================
# 静态文件（生产环境由 Nginx/CDN 提供）
# =============================================================================

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# =============================================================================
# 棋盘 API
# =============================================================================

@app.route("/api/boards", methods=["GET"])
def api_list_boards():
    """获取所有已保存棋盘列表"""
    return jsonify(list_boards())


@app.route("/api/boards", methods=["POST"])
def api_save_board():
    """
    保存新棋盘。

    Request body:
        { "name": <str>, "hexes": [{"q": int, "r": int, "s": int}, ...] }

    Response:
        { "id": <str>, "name": <str>, "hex_count": <int> }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid request"}), 400

    name = data.get("name", "未命名棋盘")
    hexes = data.get("hexes", [])

    if not hexes:
        return jsonify({"error": "hexes cannot be empty"}), 400
    if len(hexes) != 60:
        return jsonify({"error": f"hexes must be exactly 60, got {len(hexes)}"}), 400

    # 生成 ID
    import time
    board_id = f"custom-{int(time.time())}"

    result = save_board(board_id, name, hexes)
    return jsonify(result), 201


@app.route("/api/boards/<board_id>", methods=["GET"])
def api_get_board(board_id: str):
    """获取指定棋盘的详细信息"""
    board = get_board(board_id)
    if not board:
        return jsonify({"error": "board not found"}), 404
    return jsonify(board)


@app.route("/api/boards/<board_id>", methods=["DELETE"])
def api_delete_board(board_id: str):
    """删除指定棋盘"""
    success = delete_board(board_id)
    if not success:
        return jsonify({"error": "board not found or cannot be deleted"}), 404
    return "", 204


# =============================================================================
# 游戏 API
# =============================================================================

@app.route("/api/game", methods=["POST"])
def api_create_game():
    """
    创建新游戏。

    Request body (optional):
        { "seed": <int>, "board_id": <str> }

    Response:
        { "state": <GameState> }
    """
    data = request.get_json(silent=True) or {}
    seed = data.get("seed")
    board_id = data.get("board_id")
    if seed is not None:
        seed = int(seed)

    session = create_session(seed=seed, board_id=board_id)
    return jsonify({"state": session.state()})


@app.route("/api/game/<session_id>", methods=["GET"])
def api_get_state(session_id: str):
    """获取当前游戏状态。"""
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "session not found"}), 404
    return jsonify({"state": session.state()})


@app.route("/api/game/<session_id>/action", methods=["POST"])
def api_action(session_id: str):
    """
    提交玩家动作（放置或移动）。

    Request body:
        { "action": <int> }   # hexes 数组索引

    Response:
        { "state": <GameState>, "reward": <float>, "invalid": <bool> }
    """
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "session not found"}), 404

    data = request.get_json(silent=True) or {}
    action = data.get("action")
    if action is None:
        return jsonify({"error": "action is required"}), 400

    try:
        action = int(action)
    except (TypeError, ValueError):
        return jsonify({"error": "action must be an integer"}), 400

    result = session.step(action)
    return jsonify(result)


@app.route("/api/game/<session_id>/reset", methods=["POST"])
def api_reset(session_id: str):
    """重置当前游戏（使用相同 seed）。"""
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "session not found"}), 404

    return jsonify({"state": session.reset()})


# =============================================================================
# 健康检查
# =============================================================================

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok"})


# =============================================================================
# 启动
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"🐧 PenguinChess API starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
