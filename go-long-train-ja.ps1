$ErrorActionPreference = 'Stop'
chcp 65001 > $null
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error '.venv\Scripts\python.exe not found'
    pause
    exit 1
}
$env:long_train_webui = '9879'
& $python "$PSScriptRoot\GPT_SoVITS\train_webui_long_ja.py" ja_JP
pause
