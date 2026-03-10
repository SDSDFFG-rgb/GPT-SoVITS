@echo off
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
if not exist ".venv\Scripts\python.exe" (
    echo .venv\Scripts\python.exe not found
    pause
    exit /b 1
)
set "long_train_webui=9879"
.venv\Scripts\python.exe GPT_SoVITS\train_webui_long_ja.py ja_JP
pause
