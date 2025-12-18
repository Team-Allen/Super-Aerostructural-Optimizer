# PowerShell helper for Agent HQ
# Usage: .\agent_hq\run_agent.ps1 -All  or -TaskId 1
param(
    [switch]$All,
    [int]$TaskId
)

if ($All) {
    python agent_hq/run_agent.py --run-all
} elseif ($TaskId) {
    python agent_hq/run_agent.py --task-id $TaskId
} else {
    Write-Host "Specify -All or -TaskId <n>"
}
