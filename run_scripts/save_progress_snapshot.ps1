param(
    [string]$BackupRoot = "",
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($BackupRoot)) {
    $BackupRoot = Join-Path (Split-Path $repoRoot -Parent) "LOCAL_PROGRESS_BACKUPS"
}
$null = New-Item -ItemType Directory -Path $BackupRoot -Force

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$snapshotName = "super_aero_progress_$timestamp"
$snapshotDir = Join-Path $BackupRoot $snapshotName
$repoSnapshotDir = Join-Path $snapshotDir "repo_snapshot"
$null = New-Item -ItemType Directory -Path $repoSnapshotDir -Force

$includeFiles = @(
    ".gitignore",
    "docs/REAL_PHYSICS_PIPELINE.md",
    "run_scripts/animate_full_workflow_proof.py",
    "run_scripts/animate_full_workflow_proof.ps1",
    "run_scripts/desktop_pipeline_viewer.py",
    "run_scripts/prove_chain_meaningful.py",
    "run_scripts/prove_chain_meaningful_wsl.ps1",
    "run_scripts/prove_full_workflow_matplotlib.py",
    "run_scripts/run_real_physics_mdo.py",
    "run_scripts/run_real_physics_mdo_wsl.ps1",
    "run_scripts/show_proof_matplotlib.py",
    "run_scripts/start_desktop_pipeline_viewer.ps1",
    "run_scripts/test_real_physics_e2e.py",
    "run_scripts/test_real_physics_e2e_wsl.ps1",
    "run_scripts/verify_real_physics_local_paths.ps1",
    "run_scripts/save_progress_snapshot.ps1"
)

$includeDirs = @(
    "Super_Aerostructural_Optimizer/pipeline",
    "configs",
    "results/real_physics",
    "prove_chain_meaningful_out"
)

function Copy-RelativePath {
    param(
        [string]$RelativePath
    )

    $src = Join-Path $repoRoot $RelativePath
    if (-not (Test-Path $src)) {
        Write-Warning "Missing path, skipped: $RelativePath"
        return $false
    }

    $dst = Join-Path $repoSnapshotDir $RelativePath
    $dstParent = Split-Path $dst -Parent
    $null = New-Item -ItemType Directory -Path $dstParent -Force
    Copy-Item -Path $src -Destination $dst -Recurse -Force
    return $true
}

$copied = @()
foreach ($f in $includeFiles) {
    if (Copy-RelativePath -RelativePath $f) {
        $copied += $f
    }
}
foreach ($d in $includeDirs) {
    if (Copy-RelativePath -RelativePath $d) {
        $copied += $d
    }
}

$statusPath = Join-Path $snapshotDir "git_status.txt"
Push-Location $repoRoot
try {
    $gitStatus = git status --short 2>&1
    $gitBranch = git branch --show-current 2>&1
    $gitHead = git rev-parse HEAD 2>&1
}
catch {
    $gitStatus = "git status unavailable: $($_.Exception.Message)"
    $gitBranch = "unknown"
    $gitHead = "unknown"
}
finally {
    Pop-Location
}

@(
    "Timestamp: $timestamp",
    "Repo: $repoRoot",
    "Branch: $gitBranch",
    "HEAD: $gitHead",
    "",
    "git status --short:",
    $gitStatus
) | Set-Content -Path $statusPath -Encoding UTF8

$files = Get-ChildItem -Path $repoSnapshotDir -Recurse -File
$manifestFiles = foreach ($file in $files) {
    $rel = $file.FullName.Substring($repoSnapshotDir.Length + 1).Replace("\", "/")
    $hash = (Get-FileHash -Algorithm SHA256 -Path $file.FullName).Hash
    [PSCustomObject]@{
        path = $rel
        size_bytes = [int64]$file.Length
        sha256 = $hash
    }
}

$totalBytes = ($files | Measure-Object -Property Length -Sum).Sum
if ($null -eq $totalBytes) {
    $totalBytes = 0
}

$manifest = [PSCustomObject]@{
    snapshot_name = $snapshotName
    created_local = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss zzz")
    repo_root = $repoRoot
    copied_paths = $copied
    file_count = [int]$files.Count
    total_size_bytes = [int64]$totalBytes
    git_status_file = "git_status.txt"
    files = $manifestFiles
}

$manifestPath = Join-Path $snapshotDir "snapshot_manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8

$zipPath = $null
if (-not $NoZip) {
    $zipPath = Join-Path $BackupRoot "$snapshotName.zip"
    if (Test-Path $zipPath) {
        Remove-Item -Path $zipPath -Force
    }
    Compress-Archive -Path $snapshotDir -DestinationPath $zipPath -Force
}

Write-Output "Snapshot saved: $snapshotDir"
if ($zipPath) {
    Write-Output "Snapshot zip: $zipPath"
}
Write-Output "Manifest: $manifestPath"
Write-Output "Git status log: $statusPath"

