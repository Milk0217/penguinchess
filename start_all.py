#!/usr/bin/env python
"""
同时启动后端 Flask 服务器和前端 Vite 开发服务器。
"""
from __future__ import annotations

import subprocess
import sys
import os
import threading
import time

def start_flask():
    """启动 Flask 后端服务"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.app import app
    print("[Flask] API starting on http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

def start_frontend():
    """启动 Vite 前端服务"""
    print("[Vite] Frontend starting on http://localhost:5173")
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    # Windows 需要 shell=True 才能解析 npm/bun 路径
    subprocess.run(["bun", "run", "dev"], cwd=frontend_dir, shell=True)

if __name__ == "__main__":
    # 使用 daemon 线程，这样 Ctrl+C 可以同时停止两个服务
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    # 等待 Flask 启动
    time.sleep(2)

    # 启动前端
    start_frontend()