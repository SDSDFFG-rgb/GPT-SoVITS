$ErrorActionPreference = 'Stop'
chcp 65001 > $null
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error '.venv\Scripts\python.exe not found'
    pause
    exit 1
}
$env:minute_train_webui = '9878'
& $python "$PSScriptRoot\GPT_SoVITS\train_webui_1m_ja.py" ja_JP
pause
