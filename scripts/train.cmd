@echo off
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
uv run python -u -m penguinchess.ai.train_alphazero %*
