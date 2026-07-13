# Aerostructural Discipline Score — aircraft Wing Composite Ply Optimization

> **Scope:** This document describes the **aerostructural discipline module** only, not the full aircraft-level MDO.  
> **Run:** `results/wing_piml_mdo_50iter/Aircraft_Wing_Aerostructural_Optimization/`  
> **Date:** 2026-07-09  
> **Supersedes:** the generic v1 report in `archive_piml_v1/PIML_AEROSTRUCT_MDO_REPORT.md`

---

## 1. Where this module sits in the bigger picture

A full aircraft MDO optimizes many things at once:

```
┌─────────────────────────────────────────────────────────┐
│        Aircraft-level MDO (outside our scope)           │
│   Optimizes: span, sweep, engine, RCS, weight, etc.     │
│   Combines discipline scores with weights               │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   Aerodynamics      Aerostructural           RCS
   score            score (this module)      score
   (CL, CD, ...)    (composite ply,          (...)
                    deflection, mass)
```

This repository implements **only the aerostructural score box** — and even within that, we are currently focusing on the **composite laminate / ply optimization** of the wing.

The aircraft-level optimizer (e.g. COBYLA, SLSQP, or an OpenMDAO driver) would call this module, get back an aerostructural score, and combine it with other discipline scores.

> **Key point:** The COBYLA loop inside `scripts/run_pipeline.py` is **not** the aircraft-level MDO. It is the optimizer for the aerostructural sub-problem: given a fixed wing planform and flight condition, find the best airfoil, twist, thickness, and composite ply stack.

---

## 2. What this module does

Given a wing planform and flight condition, it:

1. Builds the wing geometry (CST airfoil + spanwise twist).
2. Computes aerodynamic loads using a fast physics-informed model (`NeuralFoil`).
3. Solves the structural response using a 1-D composite beam.
4. Iterates between aero and structure until the deflected shape converges (Gauss–Seidel coupling).
5. Optimizes the design variables to minimize an aerostructural objective:
   - airfoil shape,
   - wing twist,
   - skin thickness scaling,
   - composite ply counts (0°, 45°, −45°, 90°),
   - angle of attack.
6. Returns the optimized design and the aerostructural score (CD + mass penalty + constraint penalties).

### Physics-Informed Machine Learning (PIML) in this module

- **NeuralFoil** is the physics-informed / machine-learning part. It replaces slow 2-D airfoil CFD with a neural network trained on airfoil data.
- The structural solver is a classical physics-based 1-D Euler–Bernoulli composite beam.
- The coupling loop is classical fixed-point iteration.

So the “PIML” acceleration is in the aerodynamic evaluation, which is inside the inner aerostructural loop.

### 2.1 VAM-aligned structural modeling and MYSTRAN surrogate

The 1-D beam solver can use either the legacy CLT box-beam approximation or a **VAM-aligned cross-sectional stiffness module** (`piml_mdo.structures.vam_section`). The VAM module integrates the laminate ABD stiffness around a thin-walled rectangular wing box to produce a 6×6 cross-sectional stiffness matrix, giving more rigorous EA, EI, GJ, and bend-twist coupling terms than the old hand formula.

For higher-fidelity truth data, a **MYSTRAN runner** (`piml_mdo.structures.mystran_runner`) builds parametric composite wing-box BDF decks with pyNastran, runs the local MYSTRAN executable, and parses F06 displacements and composite stresses. The offline scripts

```bash
python scripts/generate_structural_doe.py --n_samples 150 --output results/structural_doe/doe.csv
python scripts/train_structural_surrogate.py --doe results/structural_doe/doe.csv --output results/structural_doe/surrogate.pt
```

vary ply counts, angles, thicknesses, and loads, then train a PyTorch MLP surrogate. Three structural solver modes are available via the YAML config (`structural_solver`):

- `"clt"` — legacy CLT box-beam approximation (retained for comparison).
- `"vam"` — VAM cross-sectional stiffness (default).
- `"surrogate"` — VAM beam + MYSTRAN-trained neural surrogate correction. Setting `structural_solver: surrogate` loads `results/structural_doe/surrogate.pt` by default; a custom path can be set with `structural_surrogate_path`.

The surrogate is currently trained on a small parametric wing-box DOE, so it corrects the 1-D beam responses for that simplified geometry. Scaling it to the full aircraft wing requires adding geometry features (span, box width, box height) to the DOE — this is the next step.

---

## 3. aircraft wing sizing used for this run

The dimensions come from:
- `Reference Docs/Aircraft DESIGN COMPARISON.docx`
- `MDO/config/wing_design_space.csv`
- `MDO/data/wing_feasible.csv`

| Parameter | Value | Source / reason |
|-----------|-------|-----------------|
| Full span | 12.0 m | Near median of 8–16 m envelope |
| Root chord | 4.5 m | Near design-space baseline |
| Tip chord | 1.5 m | Taper ≈ 0.33 |
| LE sweep | 35° | Typical aircraft strike/ISR wing |
| Cruise altitude | 10 000 m | Troposphere, ISA model valid |
| Cruise velocity | 255 m/s | Mach ≈ 0.85 |
| Target CL | 0.30 | Produces ~14.8 t lift, inside GTOW envelope |
| Material | IM7/8552 CFRP | Standard high-performance composite |
| Baseline laminate | 26-ply `thick_wing_skin` | Needed for feasible failure index |

Lift check: `L = q · S · CL ≈ 13.4 kPa · 36 m² · 0.30 ≈ 1.45e5 N` → mass ≈ **14.8 t**, which fits the feasible GTOW range of **6.5–17.7 t**.

---

## 4. Design variables and constraints

### 4.1 Design variables (25 total)

| Group | Count | What it controls |
|-------|-------|------------------|
| CST upper | 3 | Airfoil upper surface |
| CST lower | 3 | Airfoil lower surface |
| Twist | 3 | Washout along span |
| Skin thickness scale | 3 | Thickness multiplier at stations |
| Ply counts | 3 stations × 4 angles | 0°, 45°, −45°, 90° plies |
| Angle of attack | 1 | Trim |

### 4.2 Constraints

| Constraint | Limit | Final value |
|------------|-------|-------------|
| CL target | 0.30 ± 0.01 | 0.290 |
| Failure index | ≤ 0.8 | 0.788 |
| Tip deflection | ≤ 1.5 m | 0.018 m |
| Airfoil thickness | 8%–20% | satisfied |

---

## 5. The aerostructural score (objective function)

For this sub-problem, the score is:

```
Score = CD + 0.1 · (mass / 1000)
        + 10 000 · CL-trim penalty
        +   100 · failure-index penalty
        +   100 · tip-deflection penalty
        +   100 · thickness-bound penalties
```

- **Primary:** drag coefficient `CD`.
- **Secondary:** structural mass.
- **Hard constraints:** encoded as quadratic penalties so the optimizer cannot ignore them.

The CL-trim penalty is heavily weighted so the optimizer cannot simply collapse lift to avoid structural loads.

In a full aircraft MDO, this score would be passed upward and weighted against other discipline scores (RCS, propulsion, etc.).

---

## 6. Results of the aerostructural sub-optimization

> **Status update:** VAM-based end-to-end baseline + sizing/trim optimization completed.
> Coefficients below are **integrated over the semi-span and include the induced-drag
> correction**.

Baseline and optimized analyses were run with the production aircraft config
(`config/piml_aerostruct_run.yaml`): NeuralFoil xlarge aero, VAM beam, 20 beam
elements, `thick_wing_skin` laminate, 1g coupling.

```bash
python scripts/optimize_alpha_scale.py \
  --config config/piml_aerostruct_run.yaml \
  --run-name wing_vam_alpha_scale_optimized
```

### VAM baseline

| Quantity | Baseline (NACA2412) | Notes |
|----------|---------------------|-------|
| CL | 0.446 | Off-trim vs. target 0.30 |
| CD | 0.0224 | Total drag (profile + induced) |
| L/D | 19.9 | Baseline off-trim |
| Wing mass | 128.8 kg | 26-ply `thick_wing_skin` |
| Failure index | **1.497** | Above 0.8 limit; shear-driven in ±45° plies |
| Tip deflection | 0.0221 m | Negligible |

### Optimized result

A direct α + uniform-thickness-scale COBYLA optimization was used.  The 25-DV
full MDO (CST + twist + laminate) is feasible but very expensive because
integer ply counts make the landscape noisy; the α + scale run isolates the
primary trade-off (trim vs. strength) and returns a clean, converged feasible
design in ~18 minutes.

| Quantity | Optimized | Notes |
|----------|-----------|-------|
| CL | 0.300 | Trimmed to target (0.2998) |
| CD | 0.0130 | Total drag reduced vs. baseline |
| L/D | 23.0 | Improved by trimming and modest sizing |
| Wing mass | 171.9 kg | Uniform skin thickness scale = 1.334 |
| Failure index | 0.797 | Just below 0.8 limit; still shear-driven |
| Tip deflection | 0.0103 m | Negligible |
| Angle of attack | 1.60° | Reduced α to hit CL = 0.30 |
| Evaluations | 56 | COBYLA budget = 80 |
| Coupling convergence | 5 iterations | Δw < 1e-3 |

The optimizer reduced the total drag coefficient by ~42% and improved L/D by
~16% while satisfying the CL trim and failure-index constraints.  The mass
increase is the cost of bringing the failure index from 1.50 down to 0.80.

### Final design vector

| Group | Final values |
|-------|--------------|
| CST upper | 0.15, 0.17, 0.19 (initial NACA2412) |
| CST lower | −0.15, −0.14, −0.13 (initial NACA2412) |
| Twist (root→tip) | 0°, −1°, −2° (linear washout) |
| Thickness scale | 1.334 uniform |
| Ply counts (half-stack) | Baseline `[2/2/8/1]` for `thick_wing_skin` |
| Angle of attack | 1.60° |

---

## 7. Output files and visual evidence

The latest VAM end-to-end result is in:

```text
results/wing_vam_alpha_scale_optimized/
```

Key files:

- `summary.json` — final design vector and performance metrics
- `optimization_history.csv` (legacy full MDO runs) / history embedded in `summary.json` (direct α+scale run)

### ParaView screenshots

| # | File | Shows |
|---|------|-------|
| 1 | `01_undeformed_wing.png` | Wing shape with pressure coefficient |
| 2 | `02_deformed_wing.png` | Wing under load (~18 mm tip deflection) |
| 3 | `03_pressure_distribution.png` | Pressure coefficient field |
| 4 | `04_failure_index.png` | Composite failure index field |
| 5 | `05_laminate_thickness.png` | Total laminate thickness |
| 6 | `06_optimization_convergence.png` | Score vs. evaluations |

### Other outputs

- `optimization_summary.json` — final design and performance
- `optimization_history.json` — full history
- `optimization_convergence.csv` — convergence table
- `optimized_airfoil.dat` — final airfoil coordinates
- `*_wing_undeformed.vtk` / `*_wing_deformed.vtk` — 3-D geometry
- `mdo_results.png` — 6-panel summary plot
- `config.json` / `pipeline_stages.json` — run metadata

---

## 8. What was rejected and why

| Alternative | Why rejected |
|-------------|--------------|
| 15 m semi-span / 100 m/s generic wing | Produced ~10-tonne lift; not aircraft |
| 2-tonne / 45 m/s v1 config | Too small; outside aircraft GTOW envelope |
| 18-ply `optimized_wing_skin` baseline | Failure index > 1.5 under 1g loads |
| Mach > 0.9 / supersonic cruise | NeuralFoil strongest in high-subsonic range |
| L-BFGS-B / gradient-based optimizer | Finite-difference gradients expensive and fragile; COBYLA is robust for this sub-problem |
| ADflow / TACS / mphys | Not buildable on Windows |

---

## 9. Limitations (honest)

1. **Aerodynamic fidelity:** 2-D NeuralFoil sections + induced drag. No 3-D RANS or full transonic correction.
2. **Structural fidelity:** 1-D VAM beam with optional MYSTRAN-trained surrogate correction. No shell buckling, no detailed joints, and the surrogate was trained on a simplified parametric wing box rather than the full aircraft wing.
3. **Coupling:** Fixed-point Gauss–Seidel with relaxation; no analytic sensitivities.
4. **Scope:** This is only the aerostructural discipline score. RCS, propulsion, mission performance, etc. are not included.
5. **Objective nuance:** The score minimizes `CD + mass penalty` at fixed CL. A different aircraft-level objective would change the weighting.

---

## 10. How to use this in the full aircraft MDO

In the future, an outer aircraft-level framework (e.g. OpenMDAO) would:

1. Set aircraft-level design variables (span, sweep, Mach, etc.).
2. Call this module to get the **aerostructural score**.
3. Call other modules for RCS, propulsion, weights, etc.
4. Combine scores with weights into a total objective.
5. Update aircraft-level variables and repeat.

This module is therefore a **replaceable discipline component** — exactly the kind of white-box component OpenMDAO is designed to host.

---

## 11. References

- `MDO/config/wing_design_space.csv` — aircraft design-space bounds
- `MDO/data/wing_feasible.csv` — feasible aircraft designs
- `Reference Docs/Aircraft DESIGN COMPARISON.docx` — competitor/design-impact summary
- `Reference Docs/Using OpenMDAO for aircraft Workflow.docx` — MDO framework concepts
- `Reference Docs/MYSTRAN/Masti/Composite_Wing.bdf` — MYSTRAN reference laminate convention
- `config/piml_aerostruct_run.yaml` — this run’s configuration
