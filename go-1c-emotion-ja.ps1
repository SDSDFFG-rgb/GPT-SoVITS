$ErrorActionPreference = "Stop"
chcp 65001
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error ".venv\\Scripts\\python.exe not found"
    pause
    exit 1
}
$env:infer_ttswebui = "9877"
& $python "$PSScriptRoot\GPT_SoVITS\inference_webui_1c_emotion_ja.py" ja_JP
pause
