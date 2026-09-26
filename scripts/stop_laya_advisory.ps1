param([int]$Port = 8765, [int]$TimeoutSeconds = 20)
$ErrorActionPreference = 'Stop'
$DataHome = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA 'LayaResearch' } else { Join-Path $env:TEMP 'LayaResearch' }
$TokenPath = Join-Path $DataHome 'laya_advisory_shutdown.token'
$PidPath = Join-Path $DataHome 'laya_advisory.pid'
$Url = "http://127.0.0.1:$Port"
if (!(Test-Path -LiteralPath $TokenPath)) { throw 'Shutdown token is unavailable; refusing unauthenticated termination.' }
$Token = [IO.File]::ReadAllText($TokenPath, [Text.Encoding]::UTF8).Trim()
$Body = '{}'
try {
    Invoke-RestMethod -Method Post -Uri "$Url/shutdown" -Headers @{ 'X-Laya-Shutdown-Token' = $Token } -ContentType 'application/json' -Body $Body -TimeoutSec 5 | Out-Null
} catch { throw "Authenticated graceful shutdown failed: $($_.Exception.Message)" }
if (Test-Path -LiteralPath $PidPath) {
    $ServicePid = [int][IO.File]::ReadAllText($PidPath, [Text.Encoding]::UTF8).Trim()
    $Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $Deadline) {
        if (!(Get-Process -Id $ServicePid -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 250
    }
    if (Get-Process -Id $ServicePid -ErrorAction SilentlyContinue) { throw "Service PID $ServicePid did not stop gracefully." }
    Remove-Item -LiteralPath $PidPath -Force -ErrorAction SilentlyContinue
}
Remove-Item -LiteralPath $TokenPath -Force -ErrorAction SilentlyContinue
Write-Output 'Laya advisory service stopped gracefully.'
