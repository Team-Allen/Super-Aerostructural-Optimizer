# Real-Physics Aerostructural Pipeline

This repo now includes a fully scripted coupled pipeline at:

- `Super_Aerostructural_Optimizer/pipeline/`
- runner: `run_scripts/run_real_physics_mdo.py`
- config: `configs/real_physics_pipeline.json`

Tracked template config:

- `configs/real_physics_pipeline.json.example`

## Local Paths (This PC)

- Windows repo path: `F:\MDO LAB\Super-Aerostructural-Optimizer`
- WSL repo path: `/mnt/f/MDO LAB/Super-Aerostructural-Optimizer`
- Active local config path: `F:\MDO LAB\Super-Aerostructural-Optimizer\configs\real_physics_pipeline.json`

Verify local files/paths:

```powershell
.\run_scripts\verify_real_physics_local_paths.ps1
```

## Solver Stack

- Aerodynamics: spanwise finite-wing physics model (lift + induced drag + viscous drag)
- Transfer: FUNtoFEM (`RBF` default, `MELD` available via config)
- Structure: TACS shell FEM via `pyTACS`
- MDO: OpenMDAO + pyOptSparse (`IPOPT` fallback to `SLSQP`, `PSQP`)

## Run Requirements

Run this inside your Linux/WSL `mdo-best` environment (contains `tacs`, `funtofem`, `pyoptsparse`, `openmdao`).

Example from PowerShell:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc '
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate mdo-best
cd "/mnt/f/MDO LAB/Super-Aerostructural-Optimizer"
python run_scripts/run_real_physics_mdo.py --mode analyze
'
```

Optimization run:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc '
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate mdo-best
cd "/mnt/f/MDO LAB/Super-Aerostructural-Optimizer"
python run_scripts/run_real_physics_mdo.py --mode optimize
'
```

PowerShell shortcut:

```powershell
.\run_scripts\run_real_physics_mdo_wsl.ps1 -Mode analyze
.\run_scripts\run_real_physics_mdo_wsl.ps1 -Mode optimize
```

## Proper End-to-End Test (Coupling + Handoff)

Run formal E2E validation:

```powershell
.\run_scripts\test_real_physics_e2e_wsl.ps1
```

CUDA-required validation:

```powershell
.\run_scripts\test_real_physics_e2e_wsl.ps1 -RequireCuda
```

This test verifies:
- Coupling loop executes end-to-end
- Aero->struct and struct->aero handoff arrays remain finite
- Optimization loop executes and outputs finite metrics
- CUDA backend availability report (Torch/JAX/CuPy)

## Desktop Live Viewer

Launch desktop GUI viewer:

```powershell
.\run_scripts\start_desktop_pipeline_viewer.ps1
```

The viewer provides:
- Live pipeline stdout logs
- Stage status (`Aero`, `Transfer`, `TACS`, `Coupling`, `Optimizer`)
- Real-time coupling metrics per iteration (`CL`, `CD`, `Mass`, `KS`, `Tip deflection`)
- Convergence plots (`load_rel_change`, `disp_rel_change`)
- End-to-end run control (`analyze`, `optimize`, and E2E test launch)
- CUDA controls (`CUDA_VISIBLE_DEVICES`, optional CUDA-required E2E gate)

## Outputs

- `results/real_physics/generated/wing_half_struct.bdf`: generated structural BDF
- `results/real_physics/real_physics_mdo_analysis.json`: coupled analysis result
- `results/real_physics/real_physics_mdo_optimum.json`: optimization result
- `results/real_physics/real_physics_mdo_opt_cases.sql`: OpenMDAO recorder DB (if enabled)
- `results/real_physics/tacs_solution/`: structural solution files (if enabled)
