@echo off
cd /d "%~dp0"
set NO_PROXY=localhost,127.0.0.1
set no_proxy=localhost,127.0.0.1
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
