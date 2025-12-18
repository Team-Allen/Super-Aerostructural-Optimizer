# Composite Aerostructural Optimization: Classical Laminate Theory

## 🎯 Why Composites Are Better Than Metal

```
ALUMINUM (OLD):
├─ Density: 2,700 kg/m³ (heavy)
├─ E (stiffness): 70 GPa
├─ Strength: 300 MPa
├─ Cost: Low
├─ Fatigue: Sensitive
└─ Sustainability: Not recyclable easily

CARBON FIBER COMPOSITE (MODERN):
├─ Density: 1,600 kg/m³ (40% lighter!)
├─ E (stiffness): 140-180 GPa (2-3× stiffer)
├─ Strength: 1,200+ MPa (4× stronger)
├─ Cost: High (but weight savings offset it)
├─ Fatigue: Excellent
├─ Sustainability: Recyclable, growing circular economy
└─ Tailorability: Can orient fibers for specific loads
```

**Why optimize composites?**
```
Aluminum wing: Fixed properties (can't change after manufacturing)
              ├─ Stiffness same in all directions
              ├─ Either strong or light, not both
              └─ Limited optimization

Composite wing: Design freedom through fiber orientation
               ├─ Orient fibers 0°, 45°, 90° as needed
               ├─ Can be strong AND light AND stiff
               └─ Much more optimization potential
```

---

## 📚 Classical Laminate Theory (CLT) Fundamentals

### What is a Laminate?

```
LAYUP = Stack of fiber-reinforced plies

Example 8-ply laminate (carbon/epoxy):
┌─────────────────────────────┐
│  Ply 1: [0°]    0° fibers   │  Ply 1 (outer)
├─────────────────────────────┤
│  Ply 2: [±45°]  45° fibers  │  Ply 2
├─────────────────────────────┤
│  Ply 3: [90°]   90° fibers  │  Ply 3
├─────────────────────────────┤
│  Ply 4: [0°]    0° fibers   │  Ply 4 (midplane)
├─────────────────────────────┤
│  Ply 5: [0°]    0° fibers   │  Ply 5
├─────────────────────────────┤
│  Ply 6: [90°]   90° fibers  │  Ply 6
├─────────────────────────────┤
│  Ply 7: [±45°]  45° fibers  │  Ply 7
├─────────────────────────────┤
│  Ply 8: [0°]    0° fibers   │  Ply 8 (inner)
└─────────────────────────────┘

Each ply: 0.125 mm thick
Total thickness: 1 mm
```

### Fiber Orientation Matters

```
CARBON FIBER IN EACH DIRECTION:

0° (Longitudinal):
  ║║║║║║║║║║║║
  ├─ Carries bending loads
  ├─ Resists sag from gravity
  └─ Critical for wing strength

90° (Transverse):
  ════════════════════
  ├─ Carries shear forces
  ├─ Resists torsion
  └─ Provides lateral stiffness

±45° (Diagonal):
  ╱╱╱╱╱╱╱╱╱╱╱╱╱
  ├─ Carries torsional loads
  ├─ Provides damage tolerance
  └─ Absorbs impact energy
```

---

## 🔬 Classical Laminate Theory Equations

### Step 1: Individual Ply Properties (Material Properties)

```
Single ply of carbon fiber/epoxy (Unidirectional):

Longitudinal modulus (along fibers):
  E₁ = Ef × Vf + Em × (1 - Vf)
  
  Where:
    E₁  = Effective modulus in fiber direction
    Ef  = Fiber modulus (~230 GPa for carbon)
    Em  = Matrix modulus (~3.5 GPa for epoxy)
    Vf  = Fiber volume fraction (~60%)
  
  E₁ = 230 × 0.6 + 3.5 × 0.4 = 140 GPa

Transverse modulus (perpendicular to fibers):
  E₂ = Em / (1 - √Vf × (1 - Em/Ef))
  E₂ ≈ 10 GPa

Shear modulus:
  G₁₂ ≈ 5 GPa

Poisson's ratio:
  ν₁₂ ≈ 0.28

BASIC PLY PROPERTIES (4×4 stiffness matrix):
┌                           ┐
│ E₁/denom   ν₁₂E₂/denom    │
│ ν₁₂E₂/denom  E₂/denom     │  = Compliance matrix [S]
│            G₁₂            │
└                           ┘
```

### Step 2: Transform to Laminate Coordinate System

```
When a ply is at angle θ from reference frame:

[Q̄] = Transformation matrix × [Q] × Inverse transformation

For 0° ply:   [Q̄] = [Q]        (no rotation)
For 45° ply:  [Q̄] = rotated version
For 90° ply:  [Q̄] = mostly transverse properties

Example transformation for 45° ply:
Q̄₁₁ = Q₁₁cos⁴θ + Q₂₂sin⁴θ + 2Q₁₂sin²θcos²θ + 4G₁₂sin²θcos²θ
Q̄₁₂ = Q₁₂(sin⁴θ + cos⁴θ) + (Q₁₁ + Q₂₂ - 4G₁₂)sin²θcos²θ
Q̄₁₆ = (Q₁₁ - Q₁₂ - 2G₁₂)sinθcos³θ + (Q₁₂ - Q₂₂ + 2G₁₂)sin³θcosθ
[... similar for other terms]
```

### Step 3: Build Laminate Stiffness Matrix (ABD)

```
For a laminate with N plies:

A_ij = ∑(Q̄_ij)_k × (h_k)  k=1 to N

Where:
  A_ij = Extensional stiffness matrix (3×3)
  Q̄_ij = Transformed stiffness of ply k
  h_k  = Thickness of ply k

B_ij = ∑(Q̄_ij)_k × (z_k² - z_{k-1}²) / 2

Where:
  B_ij = Coupling stiffness matrix (3×3)
  z_k  = z-coordinate of ply k

D_ij = ∑(Q̄_ij)_k × (z_k³ - z_{k-1}³) / 3

Where:
  D_ij = Bending stiffness matrix (3×3)
```

**Full ABD Matrix (6×6):**
```
┌                    ┐
│ A₁₁ A₁₂ A₁₆│B₁₁ B₁₂ B₁₆ │
│ A₁₂ A₂₂ A₂₆│B₁₂ B₂₂ B₂₆ │
│ A₁₆ A₂₆ A₆₆│B₁₆ B₂₆ B₆₆ │
├────────────────────────┤
│ B₁₁ B₁₂ B₁₆│D₁₁ D₁₂ D₁₆ │
│ B₁₂ B₂₂ B₂₆│D₁₂ D₂₂ D₂₆ │
│ B₁₆ B₂₆ B₆₆│D₁₆ D₂₆ D₆₆ │
└                    ┘

A = Extensional stiffness (tension/compression)
B = Bending-extensional coupling (causes twisting under bending)
D = Bending stiffness (resists bending)
```

### Step 4: Calculate Strains from Forces/Moments

```
Laminate loads:
┌     ┐   ┌     ┐
│ Nx  │   │ Nxx │  Membrane forces (N/m)
│ Ny  │ = │ Nyy │
│ Nxy │   │ Nxy │
├─────┤   ├─────┤
│ Mx  │   │ Mxx │  Bending moments (N·m/m)
│ My  │   │ Myy │
│ Mxy │   │ Mxy │
└     ┘   └     ┘

Laminate strains (using ABD inverse):
[ε₀] = [A B]⁻¹ [N]
[κ]   [B D]    [M]

Where:
  ε₀ = mid-plane strains (extension)
  κ  = curvatures (bending)

For a wing section:
  N_x = Bending moment distributed
  M_y = Torsional moment
  Etc.
```

### Step 5: Calculate Ply Stresses

```
At each ply k, local stress:
σ_k = Q̄_k × (ε₀ + z_k × κ)

Where:
  σ_k = stress components in ply k
  z_k = distance from midplane to ply k
  
Stress increases with:
  ├─ Higher applied loads (moment)
  ├─ Greater distance from midplane (z)
  ├─ Lower stiffness in that direction
  └─ Higher fiber angle (if not aligned with load)

Example: For a bending moment M_y:
  At top fiber (z = +h/2):  Compression stress
  At bottom fiber (z = -h/2): Tension stress
  At midplane (z = 0):        Zero bending stress
```

### Step 6: Failure Criterion (Tsai-Wu)

```
Tsai-Wu Interactive Failure Index:

F = σ₁²/(S₁ᵗ×S₁ᶜ) + σ₂²/(S₂ᵗ×S₂ᶜ) - σ₁σ₂/(S₁ᵗ×S₁ᶜ)
    + τ₁₂²/S₁₂² + (1/S₁ᵗ - 1/S₁ᶜ)σ₁ + (1/S₂ᵗ - 1/S₂ᶜ)σ₂

Where:
  σ₁, σ₂ = Longitudinal and transverse stresses
  τ₁₂ = Shear stress
  
  S₁ᵗ = Tensile strength in fiber direction (~1500 MPa)
  S₁ᶜ = Compressive strength in fiber direction (~1000 MPa)
  S₂ᵗ = Transverse tensile strength (~60 MPa)
  S₂ᶜ = Transverse compressive strength (~150 MPa)
  S₁₂ = Shear strength (~80 MPa)

FAILURE if F ≥ 1.0
SAFE if F < 1.0

Margin of Safety = (1/F) - 1
Example: F = 0.5 → MOS = 100% → Safe with 2× load capacity
```

---

## 🏗️ Composite Wing Structure Model

### Wing Cross-Section (Box Beam)

```
Top skin (thin composite shell)
┌─────────────────────────────────────┐
│  [0°/±45°/90°] laminate             │  t_skin ≈ 2mm
│                                      │
│  Upper spar web                      │
│  ┌────────────────────────────────┐  │
│  │ [±45°/0°] laminate (torsion)   │  │ t_spar ≈ 1.5mm
│  │                                │  │
│  │                                │  │
│  │ [±45°/0°] laminate (torsion)   │  │
│  └────────────────────────────────┘  │
│                                      │
│  Lower spar web                      │
│  ┌────────────────────────────────┐  │
│  │ [±45°/0°] laminate (torsion)   │  │ t_spar ≈ 1.5mm
│  │                                │  │
│  │                                │  │
│  │ [±45°/0°] laminate (torsion)   │  │
│  └────────────────────────────────┘  │
│                                      │
└─────────────────────────────────────┘
Bottom skin (thin composite shell)
│  [0°/±45°/90°] laminate             │  t_skin ≈ 2mm

DESIGN DECISIONS:
├─ Skin thickness (controls bending stiffness)
├─ Spar thickness (controls torsional stiffness)
├─ Fiber orientation at each location
├─ Number of plies (redundancy for damage tolerance)
└─ Material system (carbon fiber type, epoxy resin)
```

### Composite Layup Design Variables

```
For each wing section, we optimize:

1. SKIN LAYUP (Box outer surface):
   [θ₁ₛ/θ₂ₛ/θ₃ₛ/θ₄ₛ]_T
   
   Example: [0/±45/90]_T = 8 plies total
   - θ₁ₛ = 0°   (bending: 2 plies)
   - θ₂ₛ = ±45° (torsion: 4 plies)
   - θ₃ₛ = 90°  (shear: 2 plies)
   
   Optimization: Which angles? How many plies each?

2. SPAR LAYUP (Internal webs):
   [θ₁ₐ/θ₂ₐ/θ₃ₐ]_T
   
   Example: [±45/0]_T = 6 plies total
   - Mostly ±45° (pure torsion)
   - Some 0° (longitudinal stiffness)

3. PLY THICKNESS (Each ply = 0.125 mm):
   Total thickness = n_plies × 0.125 mm
   
   Optimization: How many plies in each direction?

4. FIBER ORIENTATION ANGLES:
   θ ∈ [0°, 90°] for each ply
   
   Optimization: Best angle for each ply?
   
   Constraints:
   ├─ Manufacturing: Only 0°, ±45°, 90° (discrete angles)
   └─ Balance: ±45° plies must be symmetric
```

---

## 🔄 Composite Aerostructural Optimization Loop

```
┌─────────────────────────────────────────────────────────────┐
│            COMPOSITE WING OPTIMIZATION CYCLE                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │ 1. DESIGN VARIABLES (per section)    │
    │                                      │
    │ For each of 30 wing sections:        │
    │ ├─ Airfoil shape (from Stage 1)     │
    │ ├─ Skin fiber angles: [0°/45°/90°] │
    │ ├─ Spar fiber angles: [45°/0°]      │
    │ ├─ Skin ply count: n_skin = 1-12    │
    │ ├─ Spar ply count: n_spar = 1-8     │
    │ └─ Material properties (Vf, thickness)
    │                                      │
    │ Global wing variables:               │
    │ ├─ Span = [15, 20] m                │
    │ ├─ Taper = [0.3, 1.0]               │
    │ ├─ Sweep = [0°, 30°]                │
    │ ├─ Dihedral = [0°, 20°]             │
    │ └─ Twist = [-10°, +10°]             │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │ 2. BUILD COMPOSITE WING              │
    │                                      │
    │ For each section:                    │
    │ ├─ Define airfoil (from Stage 1)    │
    │ ├─ Define chord (from taper)         │
    │ ├─ Define chord twist (from twist)   │
    │ ├─ Create box beam cross-section     │
    │ ├─ Assign layup (skin + spar)        │
    │ ├─ Calculate ABD matrix (CLT)        │
    │ └─ Get section stiffness properties  │
    │                                      │
    │ OUTPUT: 30 composite sections        │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │ 3. AERODYNAMIC ANALYSIS (GNN)            │
    │                                          │
    │ For each section:                        │
    │ ├─ Use airfoil shape (fixed from Stage 1)
    │ ├─ GNN predicts: CL, CD at local AoA   │
    │ ├─ Convert to distributed lift: w(y)   │
    │ └─ Time: ~0.05 seconds total            │
    │                                          │
    │ OUTPUT: Distributed aerodynamic load    │
    └──────────────────┬───────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │ 4. COMPOSITE STRUCTURAL ANALYSIS (CLT)   │
    │                                          │
    │ For each section:                        │
    │                                          │
    │ Step A: Calculate ABD matrix from layup  │
    │   ├─ Transform ply properties (Q̄)       │
    │   ├─ Integrate across thickness          │
    │   └─ Get A, B, D matrices                │
    │                                          │
    │ Step B: Apply aerodynamic loads          │
    │   ├─ N_x = bending moment / chord        │
    │   ├─ M_y = torsional moment              │
    │   └─ Resolve in laminate frame           │
    │                                          │
    │ Step C: Solve CLT equations              │
    │   ├─ Calculate strains: ε = A⁻¹ × N     │
    │   ├─ Calculate curvatures: κ = D⁻¹ × M  │
    │   └─ Account for coupling (B matrix)     │
    │                                          │
    │ Step D: Calculate ply stresses           │
    │   ├─ For each ply at each section        │
    │   ├─ σ_ply = Q̄ × (ε + z × κ)            │
    │   └─ Account for position in laminate    │
    │                                          │
    │ Step E: Check failure criterion (Tsai-Wu)
    │   ├─ Calculate F for each ply            │
    │   ├─ Find max(F) across all plies        │
    │   └─ FAIL if max(F) > 1.0                │
    │                                          │
    │ Step F: Calculate composite properties   │
    │   ├─ Effective stiffness (EI from ABD)   │
    │   ├─ Effective mass                      │
    │   │   = (ρ_composite × Volume)           │
    │   │   = ρ × n_plies × ply_thickness      │
    │   └─ Natural frequencies                 │
    │                                          │
    │ OUTPUT: Stresses, masses, frequencies    │
    └──────────────────┬───────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │ 5. CONSTRAINT CHECKING                   │
    │                                          │
    │ Ply-level constraints:                   │
    │ ├─ Tsai-Wu F ≤ 1.0 (no failure)         │
    │ ├─ Ply strain ≤ max_strain               │
    │ └─ Ply stress ≤ allowable                │
    │                                          │
    │ Section-level constraints:               │
    │ ├─ Max section stress ≤ 1500 MPa        │
    │ └─ No section failure                    │
    │                                          │
    │ Wing-level constraints:                  │
    │ ├─ Buckling margin ≥ 0.2                │
    │ ├─ Flutter margin ≥ 0.2                  │
    │ ├─ Deflection ≤ span/20                  │
    │ └─ Natural frequency > 2 × flight freq   │
    │                                          │
    │ Laminate constraints:                    │
    │ ├─ Balance: ±45° plies equal             │
    │ ├─ Symmetry: [θ₁/θ₂/θ₂/θ₁]_T             │
    │ ├─ 10% rule: No single orientation > 10% │
    │ └─ Ply-drop: Max 4 plies per drop        │
    │                                          │
    │ OUTPUT: Feasibility flags + penalties    │
    └──────────────────┬───────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │ 6. MULTI-OBJECTIVE FITNESS               │
    │                                          │
    │ Objectives:                              │
    │ ├─ Maximize L/D (aerodynamics)           │
    │ ├─ Minimize total mass (material)        │
    │ ├─ Maximize natural frequency (dynamics) │
    │ └─ Maximize failure margin (safety)      │
    │                                          │
    │ Fitness = w₁×(-L/D)                      │
    │         + w₂×(mass/2500)                 │
    │         + w₃×(1/freq)                    │
    │         + w₄×(1/margin)                  │
    │         + penalties                      │
    │                                          │
    │ Weights (user-defined):                  │
    │ ├─ w₁ = 1.0 (aerodynamics dominant)     │
    │ ├─ w₂ = 0.5 (material cost)              │
    │ ├─ w₃ = 0.2 (vibration)                  │
    │ └─ w₄ = 0.3 (safety)                     │
    │                                          │
    │ OUTPUT: Single scalar fitness score      │
    └──────────────────┬───────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │ 7. OPTIMIZER UPDATE (CMA-ES)             │
    │                                          │
    │ Given 16 candidate designs with scores:  │
    │ ├─ Learn which variables matter         │
    │ ├─ Increase good values (more 0° plies?)│
    │ ├─ Decrease bad values                   │
    │ ├─ Explore promising regions             │
    │ └─ Generate next population              │
    │                                          │
    │ Converge when improvement stalls         │
    │ OUTPUT: Next 16 candidate designs        │
    └──────────────────┬───────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
    CONTINUE LOOP?            DONE - OUTPUT RESULTS
        │                             │
        NO                            YES
        │                             │
        ▼                             ▼
    [14 iterations = 126 total]    [Pareto front]
                                   [Best designs]
```

---

## 📊 Composite Material Properties (Carbon Fiber IM7/8552 Epoxy)

```
UNIDIRECTIONAL PLY (single fiber direction):

ELASTIC PROPERTIES:
├─ E₁ (longitudinal):    165 GPa
├─ E₂ (transverse):      11.2 GPa
├─ G₁₂ (shear):          8.4 GPa
├─ ν₁₂ (Poisson):        0.31
└─ Thickness per ply:    0.125 mm

STRENGTH PROPERTIES:
├─ F₁ᵗ (tensile, 0°):    2,326 MPa
├─ F₁ᶜ (compressive, 0°): 1,389 MPa
├─ F₂ᵗ (tensile, 90°):   57 MPa
├─ F₂ᶜ (compressive, 90°): 228 MPa
├─ F₆ (shear):           76 MPa
└─ F₁₂ (in-plane shear): 76 MPa

DENSITY:
├─ Carbon fiber:         1.60 g/cm³
├─ Epoxy resin:          1.30 g/cm³
└─ Composite (60% Vf):   1.56 g/cm³

ENVIRONMENTAL:
├─ Operating temp:       -55 to +177°C
├─ Moisture sensitivity: ~0.8% gain/1% RH
└─ Fatigue: ~65% of static strength @ 10⁶ cycles
```

---

## 🎯 Optimization Variables vs Parameters

### Design Variables (We Optimize These)

```
PER SECTION (30 sections × variables):

1. SKIN FIBER ANGLES:
   ├─ θ₁ˢ ∈ [0°, 90°]  (primary load direction)
   ├─ θ₂ˢ ∈ [±45°]     (torsion/shear)
   └─ θ₃ˢ ∈ [0°, 90°]  (secondary)

2. SKIN PLY COUNTS:
   ├─ n₀ˢ ∈ [1, 6]     (0° plies)
   ├─ n₄₅ˢ ∈ [2, 8]    (±45° plies, must be even)
   └─ n₉₀ˢ ∈ [1, 4]    (90° plies)

3. SPAR FIBER ANGLES:
   ├─ θ₁ᵃ ∈ [±45°]     (pure torsion)
   └─ θ₂ᵃ ∈ [0°, 90°]  (longitudinal)

4. SPAR PLY COUNTS:
   ├─ n₄₅ᵃ ∈ [2, 6]    (±45° plies)
   └─ n₀ᵃ ∈ [1, 4]     (0° plies)

GLOBAL WING VARIABLES (Same 6 as before):

5. WING SPAN:
   ├─ Span ∈ [15, 20] m
   
6. TAPER RATIO:
   ├─ Taper ∈ [0.3, 1.0]

7. SWEEP ANGLE:
   ├─ Sweep ∈ [0°, 30°]

8. DIHEDRAL:
   ├─ Dihedral ∈ [0°, 20°]

9. ROOT TWIST:
   ├─ Twist_root ∈ [-10°, +10°]

10. TIP TWIST:
    ├─ Twist_tip ∈ [-10°, +10°]

TOTAL: 30 × (3 angles + 6 ply counts) + 6 global = ~100-120 variables
```

### Fixed Parameters (NOT Optimized)

```
MATERIAL PROPERTIES (given):
├─ E₁, E₂, G₁₂, ν₁₂ (from material spec)
├─ Strengths: F₁ᵗ, F₁ᶜ, F₂ᵗ, F₂ᶜ, F₆
└─ Density: ρ = 1,560 kg/m³

CONSTRAINTS (given):
├─ Max stress: 1,500 MPa (safety factor ~1.5)
├─ Max strain: 0.7% (failure criterion)
├─ Flutter margin: ≥ 0.2
├─ Buckling margin: ≥ 0.2
└─ Min natural frequency: 2.5 Hz

LAMINATE RULES (manufacturing):
├─ Balance rule: Equal ±45° plies
├─ Symmetry rule: [θ₁/θ₂/θ₂/θ₁]
├─ 10% rule: No orientation > 10% of total plies
├─ Ply-drop rule: Max 4 plies dropped
└─ Discrete angles: Only 0°, ±45°, 90° allowed

AIRFOIL (given from Stage 1):
├─ Airfoil coordinates (fixed)
├─ CL, CD values (from trained GNN)
└─ Performance metrics (given)
```

---

## ⚙️ How Optimization Works With Composites

### Example: Optimal Root Section Layup

```
PROBLEM:
├─ Root experiences highest bending moment
├─ Needs high 0° modulus (E₁ = 165 GPa)
├─ Needs torsional strength (±45°)
└─ Budget: minimize plies (weight)

ITERATION 1 (Random start):
Layup: [0/45/90/45]_T (4 plies)
├─ 0° plies: 1
├─ ±45° plies: 2
└─ 90° plies: 1
Result: Failure! Tsai-Wu F = 1.3 > 1.0 (too weak)

ITERATION 2 (Add plies):
Layup: [0/±45/90/±45/0]_T (6 plies)
├─ 0° plies: 2
├─ ±45° plies: 4
└─ 90° plies: 0
Result: Safe! Tsai-Wu F = 0.85 < 1.0 ✓
Mass: 0.6 kg/m

ITERATION 3 (Change angles):
Layup: [0/0/±45/±45]_T (4 plies)
├─ 0° plies: 2 (more bending capacity)
├─ ±45° plies: 2 (still torsion)
└─ 90° plies: 0
Result: Safe! Tsai-Wu F = 0.72 < 1.0 ✓
Mass: 0.5 kg/m (lighter!)

ITERATION 4 (Further optimize):
Layup: [0/±45/0/±45]_T (4 plies) — SYMMETRIC
├─ 0° plies: 2
├─ ±45° plies: 2
└─ ABD coupling B = 0 (balanced)
Result: Safe AND no unwanted coupling! ✓
Mass: 0.5 kg/m
Coupling matrix: B = [0] (uncoupled bending-torsion)

FINAL: [0/±45/0/±45]_T is optimal for this section
```

---

## 🔍 Key Differences: Composites vs Metal

| Aspect | Metal (Aluminum) | Composite (Carbon) |
|--------|------------------|-------------------|
| **Stiffness** | Fixed (E = 70 GPa) | Tunable via fiber angle (E₁ = 165 GPa, E₂ = 11 GPa) |
| **Strength** | Isotropic (~300 MPa) | Anisotropic (F₁ = 2300 MPa, F₂ = 57 MPa) |
| **Weight** | 2,700 kg/m³ | 1,560 kg/m³ (42% lighter) |
| **Optimization** | Limited (size only) | Rich (fiber angle, ply count, orientation) |
| **Coupling** | None | Can design in/out via B matrix |
| **Failure** | Von Mises (simple) | Tsai-Wu criterion (complex) |
| **Manufacturing** | Casting, machining | Layup, autoclave, compression molding |
| **Cost** | Low material, high labor | High material, moderate labor |
| **Sustainability** | Recyclable | Increasingly recyclable |
| **Tailorability** | No | Yes! Optimize for each section |

---

## 🚀 Implementation Steps

### Step 1: Load Material Database
```
Materials/carbon_im7_8552.json
├─ Elastic moduli: E₁, E₂, G₁₂, ν₁₂
├─ Strength values: F₁ᵗ, F₁ᶜ, F₂ᵗ, F₂ᶜ, F₆
├─ Density: 1,560 kg/m³
└─ Ply thickness: 0.125 mm
```

### Step 2: Define Laminate Layup
```python
class CompositeLayup:
    def __init__(self, skin_plies, spar_plies):
        self.skin_angles = [0, 45, 90, -45]  # [0°/±45°/90°/±45°]
        self.skin_counts = [2, 4, 0, 0]      # [2, 4, 0, 2] for ±45
        self.spar_angles = [45, 0, -45]      # [±45°/0°/±45°]
        self.spar_counts = [2, 2, 2]         # [2, 2, 2]
        
    def build_layup(self):
        # Create actual ply sequence respecting symmetry
        # [0/±45/90/±45]_T symmetric balanced layup
        pass
```

### Step 3: Calculate ABD Matrix
```python
def calculate_abd_matrix(layup, material_props):
    """
    Calculate extensional (A), coupling (B), bending (D) matrices
    """
    # For each ply in layup:
    for ply in layup.plies:
        # 1. Get ply properties (E₁, E₂, G₁₂, ν₁₂)
        # 2. Transform to laminate frame (angle θ)
        # 3. Calculate Q̄ matrix (transformed stiffness)
        # 4. Integrate: A += Q̄ × thickness
        #            B += Q̄ × (z² term)
        #            D += Q̄ × (z³ term)
    
    return A, B, D
```

### Step 4: Analyze Under Load
```python
def analyze_composite_wing():
    """
    Full CLT analysis for composite wing section
    """
    for section in wing.sections:
        # Get aerodynamic loads (from GNN)
        N_x, M_y, M_z, T = aero_load[section]
        
        # Calculate ABD matrix for this section's layup
        A, B, D = section.layup.get_abd_matrix()
        
        # Solve for strains and curvatures
        eps_0 = inv(A) @ N_x
        kappa = inv(D) @ M_y  # Bending curvature
        
        # For each ply in section
        max_failure_index = 0
        for ply in section.layup.plies:
            # Calculate stress: σ = Q̄ × (ε₀ + z × κ)
            sigma = ply.Q_bar @ (eps_0 + ply.z * kappa)
            
            # Tsai-Wu failure index
            F = tsai_wu_criterion(sigma, material_strengths)
            max_failure_index = max(max_failure_index, F)
        
        # Store results
        section.max_stress = max_stress
        section.failure_index = max_failure_index
        section.mass = section.layup.total_thickness * density
```

### Step 5: Optimize
```python
class CompositeWingOptimizer(CMA_ES):
    def objective(self, design_vars):
        """
        Multi-objective fitness for composite wing
        """
        # Unpack design variables
        span = design_vars[0]
        taper = design_vars[1]
        skin_angles = design_vars[2:5]
        skin_plies = design_vars[5:8]
        spar_angles = design_vars[8:11]
        spar_plies = design_vars[11:14]
        # ... more variables
        
        # Build wing with this design
        wing = CompositeWing(span, taper, ...)
        
        # Analyze aerodynamics (GNN)
        aero_result = gnn.analyze(wing)  # Gets L/D
        
        # Analyze structures (CLT)
        struct_result = analyze_composite_wing()  # Gets stress, mass
        
        # Check constraints
        feasible = struct_result.failure_index < 1.0
        
        # Compute fitness
        if not feasible:
            fitness = huge_penalty
        else:
            fitness = (-(L/D) 
                     + 0.01 * (mass / 2500)
                     + 0.05 * (1 / natural_frequency)
                     + 0.1 * (1 - failure_margin))
        
        return fitness
```

---

## 📈 Expected Improvements Over Metal

```
ALUMINUM WING:
├─ Mass: 800 kg
├─ Max stress: 180 MPa
├─ Natural frequency: 2.8 Hz
├─ Failure margin: 0.67 (safe but heavy)
└─ L/D: 12 (fixed, can't optimize much)

OPTIMIZED COMPOSITE WING (after our optimization):
├─ Mass: 450 kg (-44% lighter!) ⚡
├─ Max stress: 850 MPa (uses material better)
├─ Natural frequency: 4.2 Hz (10x stiffer) 📈
├─ Failure margin: 0.6 (optimized for safety)
└─ L/D: 12-14 (can improve with design)

BENEFITS:
├─ 350 kg weight saved → 2-3% fuel burn reduction
├─ Higher stiffness → reduced flutter risk
├─ Tailored design → optimized for specific loads
├─ Better fatigue resistance → 30,000+ flight hours
└─ Sustainability → recyclable at end-of-life
```

---

## ✅ Summary: Composite vs Metal

You're **absolutely right** that composites require complete redesign:

- ❌ **Metal:** Same properties in all directions → limited optimization
- ✅ **Composite:** Fiber orientation + ply count → massive optimization potential

**Classical Laminate Theory** is essential because:
1. **ABD matrix** captures coupling (bending can cause twisting!)
2. **Fiber orientation** is a design variable (which angle minimizes mass?)
3. **Tsai-Wu criterion** is nonlinear (not simple stress limits)
4. **Ply-level analysis** needed (stresses vary through thickness)

This framework transforms wing design from "guessing sizes" to "scientifically optimizing every ply in every section." 🚀

Ready to implement this in the optimization code?
