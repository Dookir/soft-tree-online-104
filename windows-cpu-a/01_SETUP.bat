@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0project\setup_windows.ps1"
set "RESULT=%ERRORLEVEL%"
echo Exit code %RESULT%.
if /I not "%~1"=="--no-pause" if /I not "%REPRO_NO_PAUSE%"=="1" pause
exit /b %RESULT%
