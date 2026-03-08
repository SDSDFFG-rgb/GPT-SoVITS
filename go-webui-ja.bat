set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
if not exist ".venv\Scripts\python.exe" (
    echo .venv\Scripts\python.exe not found
    pause
    exit /b 1
)
set "webui_port_main=9875"
.venv\Scripts\python.exe webui.py ja_JP
pause
