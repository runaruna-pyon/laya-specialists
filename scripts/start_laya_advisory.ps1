param(
    [int]$Port = 8765,
    [int]$StartupTimeoutSeconds = 90
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Service = Join-Path $Root 'src\laya_advisory_service.py'
$Results = Join-Path $Root 'results'
$DataHome = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA 'LayaResearch' } else { Join-Path $env:TEMP 'LayaResearch' }
New-Item -ItemType Directory -Force -Path $Results,$DataHome | Out-Null
if (!(Test-Path -LiteralPath $Python) -or !(Test-Path -LiteralPath $Service)) { throw 'Laya advisory runtime files are missing.' }
$Url = "http://127.0.0.1:$Port"
try {
    $Health = Invoke-RestMethod -Uri "$Url/health" -TimeoutSec 2
    if ($Health.status -eq 'ok' -and $Health.model_sha256 -eq '761ad958e6879c73cb6bf5ba9529b68aa8eb6a6eaf23eaa18f602a4e4ee38ff6') {
        Write-Output "Laya advisory service already healthy at $Url (PID $($Health.pid))."
        exit 0
    }
    throw "An unexpected service occupies $Url; refusing to reuse it."
} catch [System.Net.WebException] { }
$RandomBytes = New-Object byte[] 32
$RandomGenerator = [Security.Cryptography.RandomNumberGenerator]::Create()
try { $RandomGenerator.GetBytes($RandomBytes) } finally { $RandomGenerator.Dispose() }
$Token = ([BitConverter]::ToString($RandomBytes)).Replace('-', '')
$TokenPath = Join-Path $DataHome 'laya_advisory_shutdown.token'
[IO.File]::WriteAllText($TokenPath, $Token, [Text.UTF8Encoding]::new($false))
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Stdout = Join-Path $Results "laya_advisory_service_$Stamp.stdout.log"
$Stderr = Join-Path $Results "laya_advisory_service_$Stamp.stderr.log"
$OldToken = $env:LAYA_ADVISORY_SHUTDOWN_TOKEN
try {
    $env:LAYA_ADVISORY_SHUTDOWN_TOKEN = $Token
    $Process = Start-Process -FilePath $Python -ArgumentList @('-u', "`"$Service`"", '--host', '127.0.0.1', '--port', "$Port") -WorkingDirectory $Root -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
} finally {
    $env:LAYA_ADVISORY_SHUTDOWN_TOKEN = $OldToken
}
$PidPath = Join-Path $DataHome 'laya_advisory.pid'
$Deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
while ((Get-Date) -lt $Deadline) {
    if ($Process.HasExited) { throw "Laya advisory service exited during startup; see $Stderr" }
    try {
        $Health = Invoke-RestMethod -Uri "$Url/health" -TimeoutSec 2
        if ($Health.status -eq 'ok' -and $Health.model_loaded -and $Health.model_sha256 -eq '761ad958e6879c73cb6bf5ba9529b68aa8eb6a6eaf23eaa18f602a4e4ee38ff6') {
            [IO.File]::WriteAllText($PidPath, [string]$Health.pid, [Text.UTF8Encoding]::new($false))
            Write-Output "Laya advisory service ready at $Url (PID $($Health.pid)); logs: $Stdout"
            exit 0
        }
    } catch { Start-Sleep -Milliseconds 500 }
    Start-Sleep -Milliseconds 500
}
try { Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue } catch { }
throw "Laya advisory service did not become healthy within $StartupTimeoutSeconds seconds; see $Stderr"
