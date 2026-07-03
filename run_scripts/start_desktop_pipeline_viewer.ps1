$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$viewer = Join-Path $repoRoot "run_scripts\desktop_pipeline_viewer.py"

python $viewer

