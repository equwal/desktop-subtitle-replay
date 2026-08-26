# Optional quality upgrade: a Finnish-fine-tuned Whisper 'small', converted to
# CTranslate2 int8 so faster-whisper can run it.
#
# Plain 'small' is a generalist and merely OK at Finnish. This fine-tune is the
# same size (= same speed on your CPU) but much better at Finnish specifically.
#
# torch + transformers are needed only for the one-off conversion, so they go
# into a throwaway venv under -BuildRoot, which you can delete afterwards.
# Needs roughly 8 GB free; use -BuildRoot to build on a roomier drive.
#
#   .\get-finnish-model.ps1
#   .\get-finnish-model.ps1 -BuildRoot D:\livecap-build
#   .\run.ps1 --model .\build\models\fi-small-ct2

param(
    [string]$BuildRoot = "$PSScriptRoot\build"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$src   = "RASMUS/Whisper_Finnish_finetuned_small_200k_samples"
$out   = Join-Path $BuildRoot "models\fi-small-ct2"
$venv  = Join-Path $BuildRoot ".venv-convert"

try {
    New-Item -ItemType Directory -Force -Path $BuildRoot -ErrorAction Stop | Out-Null
} catch {
    throw "Cannot create $BuildRoot ($($_.Exception.Message)).`n" +
          "Pick a writable location:  .\get-finnish-model.ps1 -BuildRoot D:\some\writable\dir"
}

# Keep the multi-GB HF download off C: as well.
$env:HF_HOME = Join-Path $BuildRoot "hf-cache"
$env:PIP_CACHE_DIR = Join-Path $BuildRoot "pip-cache"

if (Test-Path $out) {
    Write-Host "$out already exists. Delete it to rebuild." -ForegroundColor Yellow
    Write-Host "Use it with:  .\run.ps1 --model `"$out`""
    exit 0
}

$freeGB = $null
try {
    $root = [System.IO.Path]::GetPathRoot((Resolve-Path $BuildRoot).Path)
    $freeGB = [math]::Round((Get-CimInstance Win32_LogicalDisk `
        -Filter "DeviceID='$($root.TrimEnd('\'))'").FreeSpace / 1GB, 1)
} catch { }

if ($freeGB) {
    Write-Host "Build root: $BuildRoot  ($freeGB GB free)" -ForegroundColor Cyan
    if ($freeGB -lt 8) {
        throw "Need ~8 GB free for the conversion; only $freeGB GB available on $root.`n" +
              "Free up space, or use -BuildRoot on a roomier drive."
    }
} else {
    Write-Host "Build root: $BuildRoot (free space unknown; need ~8 GB)" -ForegroundColor Cyan
}

if (-not (Test-Path "$venv\Scripts\python.exe")) {
    Write-Host "Creating conversion venv (one-time, ~2.5 GB of torch)..." -ForegroundColor Cyan
    & py -3.11 -m venv $venv
    & "$venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
    & "$venv\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
    if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
    & "$venv\Scripts\python.exe" -m pip install transformers ctranslate2 "numpy<3"
    if ($LASTEXITCODE -ne 0) { throw "transformers/ctranslate2 install failed" }
}

Write-Host "Downloading + converting $src ..." -ForegroundColor Cyan
& "$venv\Scripts\ct2-transformers-converter.exe" `
    --model $src `
    --output_dir $out `
    --quantization int8 `
    --copy_files preprocessor_config.json tokenizer_config.json special_tokens_map.json `
                 vocab.json merges.txt normalizer.json added_tokens.json
if ($LASTEXITCODE -ne 0) { throw "conversion failed" }

# faster-whisper wants a real tokenizer.json, otherwise it quietly falls back to
# the stock openai/whisper-tiny tokenizer.
$py = @"
from transformers import WhisperTokenizerFast
WhisperTokenizerFast.from_pretrained(r'$src').save_pretrained(r'$out')
print('tokenizer.json written')
"@
$py | & "$venv\Scripts\python.exe" -

Write-Host ""
Write-Host "Done -> $out" -ForegroundColor Green
Write-Host "Use it with:  .\run.ps1 --model `"$out`""
Write-Host "Reclaim space afterwards with:  Remove-Item -Recurse -Force `"$venv`""
