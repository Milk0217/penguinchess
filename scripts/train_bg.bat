@echo off
chcp 65001 >nul
REM train_bg.bat — 后台 AlphaZero 训练（CMD 版本）
REM 用法:
REM   scripts\train_bg                      默认 XL 50 迭代
REM   scripts\train_bg xl 20                XL 20 迭代
REM   scripts\train_bg large 30             Large 30 迭代

set NETWORK=%1
if "%NETWORK%"=="" set NETWORK=xl
set ITER=%2
if "%ITER%"=="" set ITER=50
set GAMES=%3
if "%GAMES%"=="" set GAMES=500
set SIMS=%4
if "%SIMS%"=="" set SIMS=800

set ROOT=%~dp0..
set LOG=%ROOT%\training.log

echo [%DATE% %TIME%] Starting AZ %NETWORK% %ITER% iters... (log: training.log)

start /B "" powershell -NoNewWindow -Command ^
  "$env:PYTHONIOENCODING='utf-8'; ^
   uv run python -m penguinchess.ai.train_alphazero ^
     --network %NETWORK% --iterations %ITER% ^
     --games %GAMES% --simulations %SIMS% ^
     --eval-interval 5 --auto-eval ^
   > '%LOG%' 2>&1"

echo PID: %ERRORLEVEL%
echo.
echo   type training.log -Wait   实时追踪日志
