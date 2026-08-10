# Register Windows scheduled task: daily 20:30 start nightly training (pause at 08:00)
#
# Usage (run as Administrator if possible):
#   powershell -ExecutionPolicy Bypass -File .\register_nightly_task.ps1
#   powershell -ExecutionPolicy Bypass -File .\register_nightly_task.ps1 -StartTime 20:30 -StopAt 08:00
#
# Query:
#   schtasks /Query /TN "DGN-DDI-Nightly-Train" /V /FO LIST
#
# Run once now:
#   schtasks /Run /TN "DGN-DDI-Nightly-Train"

param(
    [string]$TaskName = "DGN-DDI-Nightly-Train",
    [string]$StartTime = "20:30",
    [string]$StopAt = "08:00",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $ScriptDir "run_nightly_train.ps1"

if (-not (Test-Path $Runner)) {
    throw "Runner script not found: $Runner"
}

if (-not $PythonExe) {
    $PythonExe = (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
}

$ActionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`" -StopAt `"$StopAt`""
if ($PythonExe) {
    $ActionArgs += " -PythonExe `"$PythonExe`""
}

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $ActionArgs `
    -WorkingDirectory $ScriptDir

$Trigger = New-ScheduledTaskTrigger -Daily -At $StartTime

# Window is ~11.5h (20:30->08:00); allow a little headroom past stop_at for flush/save.
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 13) `
    -MultipleInstances IgnoreNew `
    -WakeToRun

$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "DGN-DDI nightly train 20:30-08:00 with mid-epoch resume and per-epoch test" |
    Out-Null

Write-Host ("Registered scheduled task: {0}" -f $TaskName)
Write-Host ("  Daily start: {0}" -f $StartTime)
Write-Host ("  Stop at: {0} (passed to runner --stop-at)" -f $StopAt)
Write-Host ("  Runner: {0}" -f $Runner)
Write-Host ""
Write-Host "Useful commands:"
Write-Host ('  schtasks /Query /TN "{0}" /V /FO LIST' -f $TaskName)
Write-Host ('  schtasks /Run /TN "{0}"' -f $TaskName)
Write-Host ('  schtasks /End /TN "{0}"' -f $TaskName)
Write-Host "  powershell -ExecutionPolicy Bypass -File .\unregister_nightly_task.ps1"
