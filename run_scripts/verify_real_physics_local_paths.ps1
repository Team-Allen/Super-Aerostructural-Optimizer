param(
    [string]$Config = "configs/real_physics_pipeline.json"
)

$repoWin = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$drive = $repoWin.Substring(0, 1).ToLower()
$tail = $repoWin.Substring(2).Replace("\", "/")
$repoWsl = "/mnt/$drive$tail"

$paths = @(
    "Super_Aerostructural_Optimizer/pipeline/__init__.py",
    "Super_Aerostructural_Optimizer/pipeline/config.py",
    "Super_Aerostructural_Optimizer/pipeline/geometry.py",
    "Super_Aerostructural_Optimizer/pipeline/aero.py",
    "Super_Aerostructural_Optimizer/pipeline/transfer.py",
    "Super_Aerostructural_Optimizer/pipeline/structure.py",
    "Super_Aerostructural_Optimizer/pipeline/workflow.py",
    "Super_Aerostructural_Optimizer/pipeline/openmdao_opt.py",
    "run_scripts/run_real_physics_mdo.py",
    "run_scripts/run_real_physics_mdo_wsl.ps1",
    "configs/real_physics_pipeline.json.example",
    "docs/REAL_PHYSICS_PIPELINE.md"
)

Write-Host "Repo (Windows): $repoWin"
Write-Host "Repo (WSL):     $repoWsl"
Write-Host ""
Write-Host "Required files:"
foreach ($rel in $paths) {
    $full = Join-Path $repoWin $rel
    if (Test-Path $full) {
        Write-Host "  OK   $rel"
    } else {
        Write-Host "  MISS $rel"
    }
}

Write-Host ""
$cfg = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $repoWin $Config }
Write-Host "Config path: $cfg"
if (Test-Path $cfg) {
    Write-Host "  OK   config exists"
} else {
    Write-Host "  INFO config missing; it will be auto-created on first run"
}

