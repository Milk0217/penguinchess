@echo off
cd /d "%~dp0..\"
set PYTHONUNBUFFERED=1
uv run python alphazero\train.py --model 2m --iterations 50 --games 500 --simulations 800 --epochs 30 --batch-size 4096 --lr 3e-4 --eval-interval 5 --workers 16
