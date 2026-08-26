# Runs livecap.py inside the venv. All arguments are passed straight through.
#   .\run.ps1                       -> Finnish captions, desktop audio
#   .\run.ps1 --selftest 12         -> 12s recording + speed check
#   .\run.ps1 --model medium --translate
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".\.venv\Scripts\python.exe")) { throw "Run .\setup.ps1 first." }
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
& ".\.venv\Scripts\python.exe" ".\livecap.py" @args
