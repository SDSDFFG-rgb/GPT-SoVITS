$ErrorActionPreference = "Stop"
chcp 65001
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error ".venv\\Scripts\\python.exe not found"
    pause
    exit 1
}
$env:webui_port_main = "9875"
& $python "$PSScriptRoot\webui.py" ja_JP
pause
