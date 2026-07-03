param(
    [ValidateSet("analyze", "optimize")]
    [string]$Mode = "analyze",
    [string]$Config = "configs/real_physics_pipeline.json",
    [string]$ProgressFile = "",
    [string]$CudaVisibleDevices = ""
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

$configDrive = $configWin.Substring(0, 1).ToLower()
$configTail = $configWin.Substring(2).Replace("\", "/")
$configWsl = "/mnt/$configDrive$configTail"

$progressArg = ""
if (-not [string]::IsNullOrWhiteSpace($ProgressFile)) {
    if ([System.IO.Path]::IsPathRooted($ProgressFile)) {
        if (Test-Path $ProgressFile) {
            $progressWin = (Resolve-Path $ProgressFile).Path
        } else {
            $progressWin = $ProgressFile
        }
    } else {
        $progressWin = (Join-Path $repoWin $ProgressFile)
    }
    $progressDrive = $progressWin.Substring(0, 1).ToLower()
    $progressTail = $progressWin.Substring(2).Replace("\", "/")
    $progressWsl = "/mnt/$progressDrive$progressTail"
    $progressArg = "--progress-file '$progressWsl'"
}

$cudaPrefix = ""
if (-not [string]::IsNullOrWhiteSpace($CudaVisibleDevices)) {
    $cudaPrefix = "export CUDA_VISIBLE_DEVICES='$CudaVisibleDevices'; "
}

$cmd = "source /opt/miniforge3/etc/profile.d/conda.sh; " +
       "conda activate mdo-best; " +
       $cudaPrefix +
       "cd '$repoWsl'; " +
       "python run_scripts/run_real_physics_mdo.py --mode $Mode --config '$configWsl' $progressArg"

wsl -d Ubuntu-22.04 -- bash -lc $cmd
