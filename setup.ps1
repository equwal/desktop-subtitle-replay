# One-time setup: creates .venv and installs dependencies.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = $null
foreach ($cand in @("py -3.11", "py -3.12", "py -3", "python")) {
    $parts = $cand.Split(" ")
    $exe = $parts[0]
    $rest = $parts[1..($parts.Length - 1)]
    try {
        $v = & $exe @rest -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) {
            $minor = [int]($v.Split(".")[1])
            if ($minor -ge 9 -and $minor -le 12) { $py = $cand; Write-Host "Using Python $v ($cand)"; break }
        }
    } catch { }
}
if (-not $py) { throw "Need Python 3.9-3.12. Install 3.11 from python.org and re-run." }

$parts = $py.Split(" ")
& $parts[0] @($parts[1..($parts.Length - 1)]) -m venv .venv

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host ""
Write-Host "Done. Next:" -ForegroundColor Green
Write-Host "  .\run.ps1 --list-devices"
Write-Host "  .\run.ps1 --selftest 12"
Write-Host "  .\run.ps1"
