param(
    [string]$Config = "configs/real_physics_pipeline.json",
    [string]$OutDir = "results/real_physics/meaningful_proof",
    [int]$OptIters = 4,
    [int]$CoupledIters = 8,
    [string]$CudaVisibleDevices = "0",
    [switch]$OpenPlot
)

$repoWin = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$drive = $repoWin.Substring(0, 1).ToLower()
$tail = $repoWin.Substring(2).Replace("\", "/")
$repoWsl = "/mnt/$drive$tail"

if ([System.IO.Path]::IsPathRooted($Config)) {
    if (Test-Path $Config) {
        $configWin = (Resolve-Path $Config).Path
    } else {
        $configWin = $Config
    }
} else {
    $candidate = Join-Path $repoWin $Config
    if (Test-Path $candidate) {
        $configWin = (Resolve-Path $candidate).Path
    } else {
        $configWin = $candidate
    }
}

if ([System.IO.Path]::IsPathRooted($OutDir)) {
    $outWin = $OutDir
} else {
    $outWin = Join-Path $repoWin $OutDir
}

$configDrive = $configWin.Substring(0, 1).ToLower()
$configTail = $configWin.Substring(2).Replace("\", "/")
$configWsl = "/mnt/$configDrive$configTail"

$outDrive = $outWin.Substring(0, 1).ToLower()
$outTail = $outWin.Substring(2).Replace("\", "/")
$outWsl = "/mnt/$outDrive$outTail"

$cudaPrefix = ""
if (-not [string]::IsNullOrWhiteSpace($CudaVisibleDevices)) {
    $cudaPrefix = "export CUDA_VISIBLE_DEVICES='$CudaVisibleDevices'; "
}

$cmd = "source /opt/miniforge3/etc/profile.d/conda.sh; " +
       "conda activate mdo-best; " +
       $cudaPrefix +
       "cd '$repoWsl'; " +
       "python run_scripts/prove_chain_meaningful.py --config '$configWsl' --out-dir '$outWsl' --opt-iters $OptIters --coupled-iters $CoupledIters"

wsl -d Ubuntu-22.04 -- bash -lc $cmd
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($OpenPlot) {
    $plotWin = Join-Path $outWin "meaningful_workflow_proof.png"
    if (Test-Path $plotWin) {
        Start-Process $plotWin
    }
}
