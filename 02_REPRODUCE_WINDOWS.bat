@echo off
setlocal
cd /d "%~dp0"
set PYTHONHASHSEED=2026
if not exist ".venv\Scripts\python.exe" (
  echo Run 01_SETUP_WINDOWS.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run.py reproduce %*
set RESULT=%ERRORLEVEL%
pause
exit /b %RESULT%
