# 卸载夜间训练计划任务
param(
    [string]$TaskName = "DGN-DDI-Nightly-Train"
)

$ErrorActionPreference = "Stop"
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Write-Host "已卸载计划任务: $TaskName"
