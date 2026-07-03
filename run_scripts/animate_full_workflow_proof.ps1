param(
    [string]$Config = "configs/real_physics_pipeline.proof.json",
    [string]$CudaVisibleDevices = "0",
    [int]$Fps = 6,
    [switch]$NoMp4,
    [switch]$SkipRun,
    [switch]$Open
)

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$cmd = @(
    "python",
    "run_scripts/animate_full_workflow_proof.py",
    "--config", $Config,
    "--cuda-visible-devices", $CudaVisibleDevices,
    "--fps", "$Fps"
)

if ($NoMp4) { $cmd += "--no-mp4" }
if ($SkipRun) { $cmd += "--skip-run" }
if ($Open) { $cmd += "--open" }

Push-Location $repo
try {
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

