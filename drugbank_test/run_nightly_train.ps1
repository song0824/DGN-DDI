# DGN-DDI nightly trainer launcher (ASCII-only for Windows PowerShell 5.x compatibility)
# Flow: each epoch train -> auto test/record -> continue until 3 epochs no improve
# Window: start 20:30, pause at 08:00; mid-epoch checkpoint resume next night.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_nightly_train.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_nightly_train.ps1 -Fresh

param(
    [string]$Profile = "full",
    [string]$StopAt = "08:00",
    [switch]$Fresh,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
# Avoid PowerShell treating Python stderr warnings as terminating errors later.
$PSNativeCommandUseErrorActionPreference = $false

if (-not $PythonExe) {
    $PythonExe = (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
}
if (-not $PythonExe) {
    throw "python not found. Add it to PATH or pass -PythonExe."
}
if (-not (Test-Path $PythonExe)) {
    throw ("PythonExe not found: {0}" -f $PythonExe)
}

$LogDir = Join-Path $ScriptDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir ("nightly_{0}.log" -f $Stamp)
$ProgressFile = Join-Path $LogDir "nightly_progress.json"
$LatestTest = Join-Path $LogDir "latest_test_results.json"
$TestHistory = Join-Path $LogDir "nightly_test_history.jsonl"

function Write-Log([string]$Message, [switch]$Append) {
    if ($Append) {
        $Message | Tee-Object -FilePath $LogFile -Append | Out-Host
    } else {
        $Message | Tee-Object -FilePath $LogFile | Out-Host
    }
}

# Skip if already converged/completed
if (-not $Fresh -and (Test-Path $ProgressFile)) {
    try {
        $prog = Get-Content $ProgressFile -Raw -Encoding UTF8 | ConvertFrom-Json
        $st = [string]$prog.status
        if ($st -eq "converged" -or $st -eq "completed" -or $st.StartsWith("converged") -or $st.StartsWith("completed")) {
            Write-Log ("Training already finished (status={0}). Skip this nightly run." -f $st)
            exit 0
        }
    } catch {
        # ignore broken progress file and continue
    }
}

$ArgsList = @(
    "transductive_train.py",
    "--profile", $Profile,
    "--nightly",
    "--stop-at", $StopAt
)
if ($Fresh) {
    $ArgsList += "--fresh"
}

$Header = @(
    "========== DGN-DDI Nightly Train ==========",
    ("Start: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")),
    ("Python: {0}" -f $PythonExe),
    ("WorkDir: {0}" -f $ScriptDir),
    ("Profile: {0}" -f $Profile),
    ("StopAt: {0}" -f $StopAt),
    ("Fresh: {0}" -f [bool]$Fresh),
    "Loop: each epoch -> auto test/record -> stop after 3 no-improve epochs",
    ("Command: {0} {1}" -f $PythonExe, ($ArgsList -join " ")),
    "==========================================="
) -join "`r`n"
Write-Log $Header

$existing = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match "python" -and
        $_.CommandLine -and
        $_.CommandLine -match "transductive_train\.py" -and
        $_.CommandLine -match "--nightly"
    }
if ($existing) {
    $pids = @($existing | ForEach-Object { $_.ProcessId }) -join ","
    Write-Log ("Nightly train already running (PID={0}). Skip." -f $pids) -Append
    exit 0
}

$ErrorActionPreference = "Continue"

try {
    # Native stderr (Python logging WARNING) must NOT abort the pipeline under Stop mode.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExe @ArgsList 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) {
            $_.Exception.Message
        } else {
            $_
        }
    } | Tee-Object -FilePath $LogFile -Append | Out-Host
    $ExitCode = $LASTEXITCODE
    if ($null -eq $ExitCode) { $ExitCode = 0 }
    $ErrorActionPreference = $prevEap
} catch {
    Write-Log ("ERROR: {0}" -f $_.Exception.Message) -Append
    $ExitCode = 1
}

$Footer = @(
    "",
    "========== Nightly session ended ==========",
    ("ExitCode: {0}" -f $ExitCode),
    ("End: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")),
    ("Progress: {0}" -f $ProgressFile),
    ("LatestTest: {0}" -f $LatestTest),
    ("TestHistory: {0}" -f $TestHistory),
    ("Log: {0}" -f $LogFile),
    ""
) -join "`r`n"
Write-Log $Footer -Append

if (Test-Path $ProgressFile) {
    Write-Log "`r`n----- nightly_progress.json -----`r`n" -Append
    Get-Content $ProgressFile -Encoding UTF8 | Tee-Object -FilePath $LogFile -Append | Out-Host
}
if (Test-Path $LatestTest) {
    Write-Log "`r`n----- latest_test_results.json -----`r`n" -Append
    Get-Content $LatestTest -Encoding UTF8 | Tee-Object -FilePath $LogFile -Append | Out-Host
}

# Keep window visible briefly when launched by Task Scheduler
if ($Host.Name -eq "ConsoleHost") {
    Start-Sleep -Seconds 3
}

exit $ExitCode
