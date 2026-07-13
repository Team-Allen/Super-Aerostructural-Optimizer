# Agent Guidance — PIML-MDO-Pipeline

This file contains conventions and guidance for AI agents working on this repository.

## Active Codebase

The active aerostructural discipline module is in **`piml_mdo/`** and driven by **`scripts/run_pipeline.py`** with **`config/piml_aerostruct_run.yaml`**.  
This is **not** the full aircraft-level MDO — it computes the aerostructural discipline score (currently focused on composite ply optimization) that a higher-level aircraft MDO would combine with RCS, propulsion, and other discipline scores.

Legacy / reference code:
- `wing_mdo/`
- `MDO/` (appears to be a separate repo/submodule)
- `vsp_screening/`

Do not modify legacy code unless explicitly asked.

## Key Conventions

1. **Python 3.12** on Windows (MSYS2/MINGW64 bash).
2. Use **NumPy 1.26.4** compatible APIs (e.g., `np.trapz`, not `np.trapezoid`).
3. Prefer **SciPy** optimizers; COBYLA is the current production optimizer.
4. Keep aerodynamic solvers interchangeable:
   - `neuralfoil`
   - `surrogate_cfd`
   - `openaerostruct`
   - `pinn`
5. Composite laminate ply counts are continuous in the optimizer and rounded to integers inside `_apply_design_to_wing()`.
6. Flight conditions in `MDOProblem` must be passed from `PipelineConfig`, not hard-coded.
7. All results go under `results/`; archive old test runs rather than deleting them.
8. Run `python scripts/validate_installation.py` after installing new dependencies.

## Common Pitfalls

- `scripts/run_pipeline.py` previously had `--method` defaulting to `"L-BFGS-B"`, overriding YAML. Keep default `None`.
- COBYLA in SciPy 1.16+ uses `maxiter` as the function-evaluation budget; set it generously.
- `result.nit` does not exist for COBYLA; use `getattr(result, 'nit', None)`.
- The active aircraft config uses a 12 m full-span wing, Mach 0.85 cruise at 10 000 m, and CL_target = 0.30. This produces ~14.8 t of lift, consistent with the aircraft MDO feasible envelope (6.5–17.7 t GTOW).

## Testing

- Use `--quick` for fast smoke tests.
- Verify a successful run produces:
  - `optimization_summary.json`
  - `optimization_history.json`
  - `*_wing_undeformed.vtk`
  - `*_wing_deformed.vtk`
  - `paraview_screenshots/*.png`

## Documentation

Update `docs/PIML_AEROSTRUCT_MDO_REPORT.md` if you change the physics, methodology, or major config options.
