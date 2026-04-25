@echo off
echo Starting PenguinChess server...
cd /d E:\programming\penguinchess
python run_server.py > server_output.txt 2>&1
echo Server started, PID=%ERRORLEVEL%
type server_output.txt