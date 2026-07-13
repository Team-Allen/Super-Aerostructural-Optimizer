# PIML Aerostructural MDO Pipeline — Gap Analysis & Roadmap

> **Date:** 2026-07-09  
> **Scope:** Honest assessment of the current `piml_mdo` pipeline vs. proven aerostructural MDO stacks, and a concrete roadmap to bridge the gap.  
> **Author:** Agent analysis based on full codebase scan + open-literature research.

---

## 1. What You Have Built (Summary)

You have built a **credible conceptual-design aerostructural discipline module** with the following components:

| Layer | What you built | Status |
|-------|---------------|--------|
| **Aero** | NeuralFoil (2-D NN surrogate trained on XFoil) + lifting-line induced drag | ✅ Works, fast |
| **Structure** | 1-D VAM composite beam + CLT ABD matrices | ✅ Physically motivated |
| **Coupling** | Gauss–Seidel fixed point with relaxation | ✅ Standard approach |
| **Optimization** | scipy COBYLA with penalty method | ⚠️ Gradient-free, fragile |
| **Surrogate** | PyTorch MLP trained on MYSTRAN DOE data | ⚠️ Exists but not validated |
| **Post-processing** | VTK export + ParaView screenshots | ✅ Good for reporting |

**Bottom line:** This is a valid *first step* — a fast, Windows-friendly, scriptable prototype. But it is **not** a proven MDO pipeline. It is a custom script that performs sizing + trim, not a true aerostructural shape optimization with analytic gradients.

---

## 2. What the World Actually Uses (The Proven Stacks)

### 2.1 The Gold Standard: MACH Framework (MDO Lab, University of Michigan)

This is **the** reference stack for high-fidelity aerostructural optimization. It has been used in dozens of published designs (Boeing CRM, NASA tilt-wing, eVTOL, etc.).

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MACH FRAMEWORK (MDO Lab)                         │
├─────────────────────────────────────────────────────────────────────┤
│  Geometry:     pyGeo  → Free-Form Deformation (FFD) volume          │
│  Mesh:         pyHyp / IDWarp → Structured mesh + mesh deformation  │
│  Aero:         ADflow → RANS CFD with DISCRETE ADJOINT              │
│  Structure:    TACS   → Composite shell FEA with ADJOINT            │
│  Coupling:     MPhys  → OpenMDAO multiphysics standard              │
│  Derivatives:  OpenMDAO MAUD → Coupled adjoint (analytic)           │
│  Optimizer:    pyOptSparse → SNOPT / IPOPT                          │
│  Surrogates:   SMT    → Kriging, GE-KPLS, MFK                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight:** The entire stack is built around **analytic gradients via the coupled adjoint method**. This is not optional — it is the only way to optimize with O(10³) design variables. Finite differences cannot scale.

**References:**
- Kenway et al., "Multidisciplinary Design Optimization of aircraft Wings Using High-Fidelity Aerostructural Models" (AIAA, 2014)
- Jasa et al., "Open-source coupled aerostructural optimization using Python" (Struct. Multidisc. Optim., 2018)
- Yildirim et al., "A Modular Multiphysics Simulation Framework Using OpenMDAO" (AIAA, 2020)

### 2.2 The Educational / Rapid-Prototyping Standard: OpenAeroStruct

Also from MDO Lab. This is a **low-fidelity** tool but it is architecturally correct — it uses OpenMDAO, has full analytic derivatives, and can optimize a wing in minutes on a laptop.

```
┌─────────────────────────────────────────────────────────────────────┐
│                 OpenAeroStruct (OpenMDAO-based)                     │
├─────────────────────────────────────────────────────────────────────┤
│  Aero:         Vortex Lattice Method (VLM)                          │
│  Structure:    6-DOF spatial beam OR wingbox model                  │
│  Coupling:     OpenMDAO nonlinear solvers (NLBGS / Newton)          │
│  Derivatives:  FULL ANALYTIC via OpenMDAO adjoint / direct          │
│  Optimizer:    SLSQP / SNOPT via pyOptSparse                        │
│  Objective:    Fuel burn (Breguet range equation)                   │
│  Constraints:  CL trim, KS failure, fuel volume, thickness          │
│  Multipoint:   Cruise + maneuver (2.5g) built-in                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Why this matters for you:** OpenAeroStruct already has:
- A **wingbox structural model** (Chauhan & Martins, 2018) that is far more realistic than your 1-D beam
- **Multipoint optimization** (cruise + maneuver sizing)
- **Fuel burn objective** via Breguet range
- **Composite material support** (albeit simplified)
- **Full analytic derivatives** — no finite differences

It is also the teaching tool used in the University of Michigan MDO course. Students build full aerostructural optimizations with it as homework.

**Reference:** [OpenAeroStruct Docs](https://mdolab-openaerostruct.readthedocs-hosted.com/)

### 2.3 The Industry Standard: NASTRAN SOL 144 / SOL 200

- **Doublet-Lattice aerodynamics** + **NASTRAN structural**
- Used by virtually every aerospace company for flutter/stress optimization
- Not scriptable for large-scale MDO, but the physics is trusted

### 2.4 The Open-Source CFD+FEA Alternative: DAFoam + TACS + OpenFOAM

- DAFoam = discrete adjoint for OpenFOAM (RANS CFD)
- TACS = composite shell FEA with adjoint
- Coupled via MPhys + OpenMDAO
- Used by Zhejiang University and others for high-fidelity design

---

## 3. Honest Gap Analysis: Your Pipeline vs. Proven Stacks

| Capability | Your Pipeline | OpenAeroStruct | MACH (High-Fidelity) | Why It Matters |
|-----------|-------------|----------------|---------------------|----------------|
| **MDO Framework** | Custom scipy wrapper | **OpenMDAO** | **OpenMDAO** | OpenMDAO handles derivative chain rule, solver convergence, parallel execution |
| **Gradient Method** | Finite differences (FD) | **Analytic adjoint** | **Coupled adjoint** | FD scales as O(n). Adjoint is O(1) w.r.t. design vars. With 25 DVs you're okay; with 250 you're dead |
| **Optimizer** | COBYLA (gradient-free) | **SLSQP/SNOPT** | **SNOPT/IPOPT** | Gradient-free works for <10 DVs. For 25+ DVs you need gradients |
| **Aero Solver** | NeuralFoil (2-D sections) | **VLM (3-D lifting surface)** | **ADflow (RANS CFD)** | VLM captures 3-D effects, downwash, wake rollup. NeuralFoil is just 2-D strip theory |
| **Structural Solver** | 1-D VAM beam | **6-DOF beam / wingbox** | **TACS (shell FEA)** | Wingbox models realistic skin-spar-rib structure. 1-D beam misses local buckling |
| **Coupling** | Gauss–Seidel fixed point | **OpenMDAO NLBGS/Newton** | **NLBGS + Aitken** | OpenMDAO has built-in convergence acceleration, robust fallback |
| **Multipoint** | Single 1g cruise | **Cruise + 2.5g maneuver** | **Multi-load envelope** | Structures are sized by maneuver/gust loads, not cruise |
| **Objective** | CD + mass penalty | **Fuel burn (Breguet)** | **Fuel burn / DOC** | Fuel burn is the real aircraft metric; CD alone is incomplete |
| **Constraints** | Strength + trim | **KS failure + fuel vol + trim** | **Buckling + flutter + fatigue** | Missing buckling and flutter means your design may fail in reality |
| **Geometry Param.** | CST weights | **FFD (pyGeo) or B-splines** | **FFD (pyGeo)** | FFD smoothly deforms mesh; CST only changes airfoil section |
| **Surrogate** | Custom PyTorch MLP | — | **SMT (Kriging, GE-KPLS)** | SMT is battle-tested for aerospace; supports gradients, multi-fidelity |
| **Composite Modeling** | CLT ply counts | **Simplified wingbox** | **TACS shell composites** | Your VAM approach is actually reasonable for conceptual design |
| **Platform** | Windows-native | **Cross-platform** | **Linux/HPC** | MACH requires Linux. OpenAeroStruct runs on Windows |

---

## 4. The Critical Insight: Why OpenMDAO Is Non-Negotiable

Your pipeline's **single biggest architectural gap** is that it does not use **OpenMDAO**.

Here's why OpenMDAO is the backbone of every serious aerostructural MDO pipeline:

1. **Automatic derivative computation:** You define partial derivatives for each component. OpenMDAO uses the chain rule (MAUD theory) to assemble total derivatives — including the coupled adjoint. This is the difference between O(n) and O(1) gradient cost.

2. **Modular solver architecture:** You can swap NLBGS for Newton, change convergence criteria, add disciplines, all without rewriting your physics.

3. **Parallel execution:** Multipoint optimization (cruise + maneuver) runs in parallel.

4. **Standard interfaces:** MPhys standardizes how CFD/FEA codes plug in. ADflow, TACS, DAFoam, OpenAeroStruct — all speak the same OpenMDAO protocol.

5. **Ecosystem:** pyOptSparse, Dymos (mission optimization), pyCycle (engine) — all OpenMDAO-native.

**Every** paper you read on aerostructural MDO — whether from MDO Lab, NASA Glenn, MIT, Zhejiang, or Iowa State — uses OpenMDAO.

---

## 5. Concrete Roadmap: From Prototype to Production

### Phase 1: Fix the Foundation (2–3 weeks)
**Goal:** Make your existing pipeline honest, reproducible, and properly benchmarked.

1. **Validate VAM beam against MYSTRAN**
   - Run 10 parametric cases through both solvers
   - Plot VAM vs. MYSTRAN for tip deflection, max stress, FI
   - Quantify error bounds; document where VAM is valid

2. **Switch to a continuous CL penalty**
   - Replace the ±0.01 deadband with `(CL − CL_target)²`
   - This removes the post-hoc manual trim problem

3. **Add a buckling check (even simplified)**
   - Euler buckling of upper skin: `P_cr = π²EI / L²`
   - Compare to actual compressive load from bending
   - This is the minimum bar for a credible structural design

4. **Document coefficient definitions**
   - Always report both mean-sectional (2-D) and integrated (3-D) coefficients
   - Never switch definitions mid-project

### Phase 2: Adopt OpenMDAO + OpenAeroStruct (3–4 weeks)
**Goal:** Replace custom coupling/optimization with the industry-standard framework.

1. **Install and run OpenAeroStruct tutorials**
   ```bash
   pip install openaerostruct openmdao
   ```
   Work through:
   - Aerodynamic optimization
   - Structural optimization
   - Aerostructural optimization (tubular spar)
   - Aerostructural optimization (wingbox)

2. **Reproduce your aircraft wing in OpenAeroStruct**
   - Same planform: 12 m span, 4.5/1.5 m chords, 35° sweep
   - Same flight condition: Mach 0.85, 10 000 m, CL = 0.30
   - Compare OpenAeroStruct result to your current result

3. **Run the wingbox multipoint optimization**
   - Cruise point for fuel burn
   - 2.5g maneuver point for structural sizing
   - This will immediately show you what your current pipeline is missing

4. **Study the derivative system**
   - Use `check_partials` and `total_derivs` in OpenMDAO
   - Understand why analytic gradients beat finite differences

### Phase 3: Hybrid PIML Pipeline — Merge Your Work with OpenMDAO (4–6 weeks)
**Goal:** Keep your fast NeuralFilo surrogate but embed it in a proper MDO framework.

1. **Wrap NeuralFoil as an OpenMDAO Component**
   - Implement `compute()` and `compute_partials()`
   - Use finite differences or neural-network Jacobian for partials
   - This gives you fast aero with OpenMDAO's gradient machinery

2. **Replace VAM beam with OpenAeroStruct wingbox**
   - Or keep VAM but wrap it as an OpenMDAO component
   - The wingbox model has built-in composite support and is already validated

3. **Use OpenMDAO's coupled solver**
   - NLBGS with Aitken acceleration
   - Or Newton if coupling is strong

4. **Add SMT surrogates for acceleration**
   ```bash
   pip install smt
   ```
   - Train Kriging or GE-KPLS on OpenAeroStruct data
   - Use multi-fidelity Kriging (MFK) if you have both low- and high-fidelity data
   - SMT supports gradient-enhanced surrogates — use adjoint derivatives if available

5. **Switch to SLSQP or SNOPT**
   - `pyOptSparse` interface through OpenMDAO
   - SNOPT is the aerospace industry standard for large-scale NLP
   - SLSQP is freely available via scipy if SNOPT license is unavailable

### Phase 4: High-Fidelity Upgrade Path (Future)
**Goal:** Scale to RANS CFD + shell FEA when resources allow.

| When | Upgrade |
|------|---------|
| You have Linux/HPC access | Add ADflow (RANS) + TACS (shell FEA) |
| You have OpenFOAM | Add DAFoam (discrete adjoint for OpenFOAM) |
| You need more design vars | Use coupled adjoint (cost ~ independent of n_DVs) |
| You need multi-objective | Use SMT's EGO or NSGA-II via pymoo |

---

## 6. Recommended Tech Stack (Tiered)

### Tier 1: Immediate (Windows-friendly, fast)
| Component | Tool | Why |
|-----------|------|-----|
| MDO Framework | **OpenMDAO** | Non-negotiable. Handles derivatives, coupling, parallel execution |
| Low-fidelity Aero | **NeuralFoil** (your own) or **OpenAeroStruct VLM** | Keep NeuralFoil for 2-D; add VLM for 3-D |
| Low-fidelity Structure | **OpenAeroStruct wingbox** | Proven, has composite support, analytic derivatives |
| Surrogate | **SMT** (Kriging, GE-KPLS) | MDO Lab / ONERA standard; gradient-enhanced |
| Optimizer | **SLSQP** (scipy) or **SNOPT** (pyOptSparse) | Gradient-based, handles constraints |
| Geometry | **CST** (keep yours) or **FFD** (pyGeo) | FFD is more flexible for planform changes |

### Tier 2: Medium-term (Linux or WSL)
| Component | Tool | Why |
|-----------|------|-----|
| RANS CFD | **ADflow** or **DAFoam** | Discrete adjoint for efficient gradients |
| Shell FEA | **TACS** | Composite shell optimization with adjoint |
| Mesh Deformation | **IDWarp** | Propagates surface changes to volume mesh |
| Geometry | **pyGeo** (FFD) | Industry-standard shape parameterization |

### Tier 3: Advanced
| Component | Tool | Why |
|-----------|------|-----|
| Multi-fidelity | **SMT MFK** | Fuses low- and high-fidelity data |
| Bayesian Opt. | **SMT EGO** | Sample-efficient global optimization |
| Mission Analysis | **Dymos** (OpenMDAO) | Trajectory optimization integrated with design |

---

## 7. Key Papers & Resources to Study

1. **OpenMDAO paper** — Gray et al., "OpenMDAO: an open-source framework for multidisciplinary design, analysis, and optimization" (Struct. Multidisc. Optim., 2019)
2. **OpenAeroStruct paper** — Jasa et al., "Open-source coupled aerostructural optimization using Python" (Struct. Multidisc. Optim., 2018)
3. **MACH framework** — Kenway et al., "Multidisciplinary Design Optimization of aircraft Wings Using High-Fidelity Aerostructural Models" (AIAA, 2014)
4. **Coupled adjoint** — Martins et al., "An automated method for sensitivity analysis using complex variables" (AIAA, 2001) and subsequent MAUD papers
5. **SMT** — Bouhlel et al., "A Python surrogate modeling framework with derivatives" (Adv. Eng. Software, 2019)
6. **MPhys** — Yildirim et al., "A Modular Multiphysics Simulation Framework Using OpenMDAO" (AIAA, 2020)
7. **SMT 2.0** — Saves et al., "SMT 2.0: A Surrogate Modeling Toolbox" (JOSS, 2023)

### Online Resources
- [OpenAeroStruct Documentation](https://mdolab-openaerostruct.readthedocs-hosted.com/)
- [OpenMDAO Documentation](https://openmdao.org/)
- [MDO Lab Software](https://mdolab.engin.umich.edu/software)
- [SMT Documentation](https://smt.readthedocs.io/)
- [MACH-Aero Tutorials](https://github.com/mdolab/MACH-Aero)

---

## 8. Bottom Line

Your current pipeline is a **valid conceptual-design prototype**. The physics is defensible, the code is clean, and the results are physically consistent. But it has three fundamental limitations:

1. **No OpenMDAO** — which means no automatic derivative computation, no modular solver architecture, and no access to the entire aerospace MDO ecosystem.
2. **Gradient-free optimization** — COBYLA with finite differences cannot scale beyond ~10 design variables. The full 25-DV problem was infeasible for this reason.
3. **No multipoint / no fuel burn objective** — A wing sized only for cruise is not a real aircraft wing. You need maneuver loads and Breguet-range fuel burn.

### The single best next step:

> **Install OpenAeroStruct, reproduce your aircraft wing in it, and compare the results.**

This will teach you:
- How OpenMDAO works (the framework you'll use for everything)
- What a proper wingbox model looks like
- How analytic derivatives enable 25-DV optimization in minutes, not hours
- What multipoint optimization reveals about your design

From there, you can decide whether to:
- **Path A:** Migrate your NeuralFoil + VAM beam into OpenMDAO components (hybrid PIML)
- **Path B:** Use OpenAeroStruct as your low-fidelity engine and add SMT surrogates for acceleration
- **Path C:** Eventually upgrade to ADflow + TACS for high-fidelity design refinement

All three paths require OpenMDAO as the foundation. That is the investment that pays off every step of the way.

---

*"OpenAeroStruct captures some of the same trends as high-fidelity analyses. It can be used to explore the design space before resorting to more computationally expensive methods."* — Jasa et al., 2018
