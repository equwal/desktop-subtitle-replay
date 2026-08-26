# Starts both halves: live captions now, and automatic subtitling of every
# replay clip OBS saves.
#
#   .\start.ps1                     # Finnish
#   .\start.ps1 -Lang ru            # Russian
#   .\start.ps1 -Lang ja -Model base
#   .\start.ps1 -LiveOnly
#
# Two windows open. Close either to stop that half.

param(
    [string]$Lang = "fi",
    [string]$Model = "small",
    [string]$ReplayModel = "large-v3-turbo",
    [string]$WatchFolder = "",
    [int]$ServePort = 8778,
    [switch]$LiveOnly,
    [switch]$ReplayOnly
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run .\setup.ps1 first." }

if (-not $ReplayOnly) {
    Write-Host "Starting live captions ($Lang, $Model)..." -ForegroundColor Cyan
    Start-Process -FilePath $py `
        -ArgumentList ".\livecap.py", "--lang", $Lang, "--model", $Model `
        -WorkingDirectory $PSScriptRoot
    Start-Sleep -Seconds 2
    Write-Host "  OBS overlay   http://127.0.0.1:8777/overlay.html?ws=8765"
    Write-Host "  Reader        http://127.0.0.1:8777/reader.html?ws=8765" -ForegroundColor Green
    Write-Host "  Control       http://127.0.0.1:8777/control.html?ws=8765"
}

if (-not $LiveOnly) {
    $watchArgs = @(".\subtitle.py", "--lang", $Lang, "--model", $ReplayModel,
                   "--serve", "$ServePort", "--watch")
    if ($WatchFolder) { $watchArgs += $WatchFolder }

    Write-Host ""
    Write-Host "Starting replay watcher ($ReplayModel)..." -ForegroundColor Cyan
    Start-Process -FilePath $py -ArgumentList $watchArgs -WorkingDirectory $PSScriptRoot
    Start-Sleep -Seconds 2
    Write-Host "  Clips         http://127.0.0.1:$ServePort/" -ForegroundColor Green
    Write-Host "  Save a replay in OBS and its mining page appears there."
}

Write-Host ""
Write-Host "Open the Reader and the Clips pages in the browser where Yomitan lives." -ForegroundColor Yellow
