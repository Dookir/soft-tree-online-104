@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist "project\.venv\Scripts\python.exe" (echo Run 01_SETUP.bat first. & pause & exit /b 2)
project\.venv\Scripts\python.exe -u launch_run.py formal
set "RESULT=%ERRORLEVEL%"
echo Exit code %RESULT%.
if /I not "%~1"=="--no-pause" if /I not "%REPRO_NO_PAUSE%"=="1" pause
exit /b %RESULT%
