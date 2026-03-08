param(
    [string]$PythonExe = "C:\Users\81809\AppData\Local\Programs\Python\Python310\python.exe",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"
chcp 65001 | Out-Null
Set-Location $PSScriptRoot

function Write-Step([string]$Message) {
    Write-Host "[setup] $Message" -ForegroundColor Cyan
}

function Ensure-FileDownload([string]$Url, [string]$Destination) {
    if (Test-Path $Destination) {
        Write-Step "Skip existing file: $Destination"
        return
    }
    $parent = Split-Path -Parent $Destination
    if ($parent -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
    Write-Step "Downloading: $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Destination
}

function Ensure-Directory([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

if (-not (Test-Path $PythonExe)) {
    throw "Python not found: $PythonExe"
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Step "Creating .venv"
    & $PythonExe -m venv .venv
}

Write-Step "Upgrading pip"
& $venvPython -m pip install --upgrade pip

Write-Step "Installing torch packages"
& $venvPython -m pip install torch torchaudio --index-url $TorchIndexUrl
& $venvPython -m pip install torchcodec

Write-Step "Installing Python dependencies"
& $venvPython -m pip install -r extra-req.txt --no-deps
& $venvPython -m pip install -r requirements.txt
& $venvPython -m pip install onnxruntime

$pretrainedZip = Join-Path $PSScriptRoot "pretrained_models.zip"
$g2pwZip = Join-Path $PSScriptRoot "G2PWModel.zip"
$nltkZip = Join-Path $PSScriptRoot "nltk_data.zip"
$openJTalkTar = Join-Path $PSScriptRoot "open_jtalk_dic_utf_8-1.11.tar.gz"

if (-not (Test-Path "GPT_SoVITS\pretrained_models\sv")) {
    Ensure-FileDownload "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip" $pretrainedZip
    Write-Step "Extracting pretrained base assets"
    Expand-Archive -Path $pretrainedZip -DestinationPath "GPT_SoVITS" -Force
    Remove-Item $pretrainedZip -Force
}

if (-not (Test-Path "GPT_SoVITS\text\G2PWModel")) {
    Ensure-FileDownload "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip" $g2pwZip
    Write-Step "Extracting G2PWModel"
    Expand-Archive -Path $g2pwZip -DestinationPath "GPT_SoVITS\text" -Force
    Remove-Item $g2pwZip -Force
}

$venvPrefix = (& $venvPython -c "import sys; print(sys.prefix)").Trim()
Ensure-FileDownload "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip" $nltkZip
Write-Step "Extracting NLTK data"
Expand-Archive -Path $nltkZip -DestinationPath $venvPrefix -Force
Remove-Item $nltkZip -Force

$pyopenjtalkDir = (& $venvPython -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))").Trim()
Ensure-FileDownload "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz" $openJTalkTar
Write-Step "Extracting OpenJTalk dictionary"
& tar -xzf $openJTalkTar -C $pyopenjtalkDir
Remove-Item $openJTalkTar -Force

$modelFiles = @(
    @{ Url = "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v4-pretrained/s2Gv4.pth"; Path = "GPT_SoVITS\pretrained_models\gsv-v4-pretrained\s2Gv4.pth" },
    @{ Url = "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v4-pretrained/vocoder.pth"; Path = "GPT_SoVITS\pretrained_models\gsv-v4-pretrained\vocoder.pth" },
    @{ Url = "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Dv2Pro.pth"; Path = "GPT_SoVITS\pretrained_models\v2Pro\s2Dv2Pro.pth" },
    @{ Url = "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Gv2Pro.pth"; Path = "GPT_SoVITS\pretrained_models\v2Pro\s2Gv2Pro.pth" },
    @{ Url = "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Dv2ProPlus.pth"; Path = "GPT_SoVITS\pretrained_models\v2Pro\s2Dv2ProPlus.pth" },
    @{ Url = "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Gv2ProPlus.pth"; Path = "GPT_SoVITS\pretrained_models\v2Pro\s2Gv2ProPlus.pth" },
    @{ Url = "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/sv/pretrained_eres2netv2w24s4ep4.ckpt"; Path = "GPT_SoVITS\pretrained_models\sv\pretrained_eres2netv2w24s4ep4.ckpt" }
)
foreach ($item in $modelFiles) {
    Ensure-Directory (Split-Path -Parent $item.Path)
    Ensure-FileDownload $item.Url $item.Path
}

Write-Step "Running syntax check"
& $venvPython -m py_compile "$PSScriptRoot\GPT_SoVITS\inference_webui_1c_ja.py"

Write-Step "Setup complete"
Write-Host "Run go-webui-ja.bat for the full Japanese WebUI"
Write-Host "Run go-1c-infer-ja.bat for the dedicated Japanese 1C inference UI"
