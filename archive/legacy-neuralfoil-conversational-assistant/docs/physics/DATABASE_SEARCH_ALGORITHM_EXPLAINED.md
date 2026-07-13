"""
DATABASE SEARCH SCORING ALGORITHM - COMPLETE BREAKDOWN
=======================================================

The database search uses a PHYSICS-BASED 100-POINT SCORING SYSTEM
with 5 criteria weighted by aerodynamic importance.
"""

## SEARCH PROCESS - 3 STEPS
## =========================

### STEP 1: REYNOLDS NUMBER PRE-FILTERING
```
For each airfoil in database (686 total):
    Get validated Reynolds range: [Re_min, Re_max]
    
    If Re_min ≤ Re_user ≤ Re_max:
        ✅ Add to search pool (wind tunnel validated)
    
    Elif 0.5×Re_min ≤ Re_user ≤ 2×Re_max:
        ⚠️  Add to search pool (acceptable extrapolation)
    
    Else:
        ❌ Reject (too far outside validated range)

Result: Filtered pool of Re-compatible airfoils
```

### STEP 2: MULTI-CRITERIA SCORING (100 points total)
```
For each airfoil in search pool:
    score = 0
    score += reynolds_match_score(airfoil)      # 35 points
    score += application_match_score(airfoil)   # 25 points  
    score += thickness_match_score(airfoil)     # 25 points
    score += performance_score(airfoil)         # 25 points
    score += lift_capability_score(airfoil)     # 20 points
    
    airfoil.total_score = min(score, 100.0)
```

### STEP 3: RANKING & SELECTION
```
Sort all airfoils by total_score (descending)
Select airfoil with highest score
Return top match + 3 alternatives
```


## DETAILED SCORING CRITERIA
## ==========================

### 1️⃣ REYNOLDS NUMBER MATCH (35 points - MOST CRITICAL)
**Physics Basis:** Boundary layer behavior changes dramatically with Re

```python
re_min, re_max = airfoil.reynolds_range  # Wind tunnel validated range
re_user = user_requirements.reynolds_number

if re_min ≤ re_user ≤ re_max:
    # PERFECT: Within wind tunnel validated range
    score = 35 points
    
    Physical meaning:
    • Boundary layer transition verified
    • Separation behavior validated
    • Drag measurements accurate
    • CL_max confirmed by testing

elif 0.5×re_min ≤ re_user ≤ 2×re_max:
    # EXTRAPOLATION: Apply quadratic penalty
    
    if re_user < re_min:
        ratio = re_user / re_min  # Lower Re extrapolation
    else:
        ratio = re_max / re_user  # Higher Re extrapolation
    
    score = 35 × ratio²
    
    Physical meaning:
    • Extrapolation introduces uncertainty
    • Quadratic penalty reflects:
      - Boundary layer thickness ~ Re^(-1/2)
      - Transition point shifts ~ Re
      - Separation bubble changes
    
    Example penalties:
    • Re_user = 0.75×Re_min → ratio=0.75 → score = 19.7 pts
    • Re_user = 0.5×Re_min  → ratio=0.5  → score = 8.75 pts
    • Re_user = 2×Re_max    → ratio=0.5  → score = 8.75 pts

else:
    # DANGEROUS: Far outside validated range
    score = 5 points (reject unless no alternatives)
    
    Physical meaning:
    • Flow regime may be completely different
    • Laminar → turbulent transition unpredictable
    • Separation point highly uncertain
    • Performance data unreliable
```

**Why Reynolds is weighted 35%:**
- Most critical aerodynamic parameter after geometry
- Determines boundary layer regime (laminar/turbulent/transitional)
- Directly affects:
  * Skin friction: CF ~ Re^(-1/7) for turbulent
  * Separation: More likely at low Re
  * Stall behavior: CL_max varies significantly with Re
  * Drag: CD can change 50-200% outside validated range


### 2️⃣ APPLICATION TYPE MATCH (25 points)
**Physics Basis:** Different aircraft need different pressure distributions

```python
user_type = requirements.aircraft_type  # "general", "glider", "uav", etc.
airfoil_type = airfoil.application_type

if airfoil_type == user_type:
    # PERFECT: Designed for this application
    score = 25 points
    
    Physical justifications:
    
    GLIDER:
    • High camber (4-6%) → maximize CL/CD
    • t/c = 12-15% → L/D > 40 typical
    • Gentle stall (Cm_stall < 0.1)
    • Low Re performance (Re ~ 10^5 - 10^6)
    
    FIGHTER:
    • Low thickness (6-8%) → minimize wave drag
    • Low camber (0-2%) → symmetric-ish
    • High Mach capability (transonic M > 0.8)
    • Sharp leading edge → vortex lift
    
    TRANSPORT:
    • Medium thickness (12-14%) → structure + fuel
    • Moderate camber (2-4%) → cruise efficiency  
    • High CL_max (>1.8 with flaps) → short field
    • Predictable stall → safety
    
    UAV:
    • Low Re optimized (Re ~ 10^4 - 10^6)
    • Separation resistant
    • High endurance (L/D > 15)
    • Light weight → high t/c acceptable
    
    GENERAL AVIATION:
    • t/c = 12-15% → good compromise
    • CL_max ~ 1.6 → low speed handling
    • Docile stall → pilot friendly
    • Wide Re range → versatile

elif airfoil_type == "general":
    # ACCEPTABLE: General-purpose design
    score = 15 points
    
    Physical meaning:
    • Not optimized for specific mission
    • Good all-around characteristics
    • Safe fallback choice

else:
    # MISMATCH: Wrong pressure distribution
    score = 0 points
    
    Example problems:
    • Glider airfoil on fighter → too much drag
    • Fighter airfoil on transport → stall issues
    • UAV airfoil at high Re → not optimized
```


### 3️⃣ THICKNESS RATIO MATCH (25 points)
**Physics Basis:** Structural vs aerodynamic trade-off

```python
ideal_thickness = (requirements.min_thickness + requirements.max_thickness) / 2
airfoil_thickness = airfoil.thickness_ratio
thickness_error = abs(airfoil_thickness - ideal_thickness)

if thickness_error < 0.02:  # Within 2%
    score = 25 points
    
elif thickness_error < 0.05:  # Within 5%
    score = 15 points
    
else:
    score = 0 points

Physical significance:

CD_pressure ≈ k × (t/c - t_optimal)²

Where t_optimal depends on Mach number:
• Subsonic (M < 0.5):   t_opt = 12-15% (structural efficiency)
• High subsonic (0.7):  t_opt = 10-12% (drag rise delay)
• Transonic (M > 0.8):  t_opt = 6-8%   (wave drag minimum)

Effects of thickness:

TOO THIN (t/c < 8%):
• ❌ Low structural strength
• ❌ Less fuel volume
• ❌ Higher manufacturing cost
• ✅ Lower wave drag at high Mach
• ✅ Lower form drag

TOO THICK (t/c > 18%):
• ✅ High structural strength  
• ✅ More fuel volume
• ❌ Higher form drag (separated flow)
• ❌ Wave drag rise at lower Mach
• ❌ Lower L/D ratio

OPTIMAL RANGE (8-15%):
• Balanced drag (friction + pressure)
• Adequate structure
• Good fuel volume
• Manufacturable
```


### 4️⃣ PERFORMANCE CAPABILITY (25 points)
**Physics Basis:** Lift-to-drag ratio from polar estimates

```python
# Estimate maximum L/D from geometry
estimated_ld = airfoil.cl_range[1] / airfoil.cd_min
required_ld = requirements.min_lift_to_drag_ratio

if estimated_ld ≥ required_ld:
    # EXCELLENT: Meets or exceeds requirement
    score = 25 points
    
    Physical validation:
    L/D_max ≈ √(π × AR × e / CD₀) × 0.5
    
    Where:
    CD₀ = zero-lift drag (friction + form)
    AR = aspect ratio
    e = span efficiency (0.7-0.95)
    
    Typical values:
    • Glider: L/D = 40-70 (very low CD₀)
    • Transport: L/D = 18-22
    • Fighter (subsonic): L/D = 10-15
    • General aviation: L/D = 12-16

elif estimated_ld ≥ 0.8 × required_ld:
    # MARGINAL: Close but not ideal
    score = 15 points
    
    Physical meaning:
    • May meet requirement with optimization
    • Wing design (AR, taper) can compensate
    • Some performance margin sacrificed

else:
    # INSUFFICIENT: Cannot meet requirement
    score = 0 points
    
    Physical problem:
    • CD too high → excessive drag
    • CL_max too low → need more wing area
    • Poor cruise efficiency
```


### 5️⃣ LIFT COEFFICIENT CAPABILITY (20 points)
**Physics Basis:** Operating point within airfoil polar

```python
target_cl = requirements.target_lift_coefficient
cl_min, cl_max = airfoil.cl_range

if cl_min ≤ target_cl ≤ cl_max:
    # PERFECT: Operating point in validated range
    score = 20 points
    
    Physical meaning:
    • Airfoil tested at this CL
    • No extrapolation needed
    • Known boundary layer behavior
    • Predictable stall margin
    
    CL physics:
    CL = 2π(α - α_L0)  # Thin airfoil theory
    
    Where:
    α = angle of attack
    α_L0 = zero-lift angle (depends on camber)
    
    Typical ranges:
    • Symmetric (NACA 0012): CL = -1.2 to +1.2
    • Cambered (NACA 2412): CL = -0.4 to +1.6
    • High-lift (flapped): CL = -0.5 to +2.5

elif target_cl < cl_max:
    # ACCEPTABLE: Below maximum but outside range
    score = 10 points
    
    Physical meaning:
    • Below tested CL_min (unusual)
    • May indicate negative α required
    • Or different Re behavior expected

else:
    # DANGEROUS: Exceeds CL_max
    score = 0 points
    
    Physical problem:
    • Risk of stall
    • Separated flow likely
    • High drag
    • Loss of control effectiveness
```


## SCORING SUMMARY TABLE
## ======================

| Criterion | Max Points | Physical Basis | Why Important |
|-----------|-----------|----------------|---------------|
| **Reynolds Match** | 35 | Boundary layer regime | Determines drag, separation, transition |
| **Application Type** | 25 | Pressure distribution | Mission-specific optimization |
| **Thickness Ratio** | 25 | Drag vs structure | Form drag + wave drag trade-off |
| **L/D Performance** | 25 | Efficiency ratio | Overall aerodynamic quality |
| **CL Capability** | 20 | Lift range | Operating point validation |
| **TOTAL** | **130*** | (capped at 100) | - |

*Note: Maximum 130 points possible, but capped at 100 to normalize scores


## EXAMPLE SCORING
## ================

**Scenario:** Design a glider for Re = 1×10⁶, CL_cruise = 1.0

**Candidate A: NACA 2412** (general aviation airfoil)
```
Reynolds Match:
  Re_range = [5×10⁵, 5×10⁶]
  1×10⁶ is within range → 35 points

Application:
  Type = "general" (not "glider") → 15 points

Thickness:
  t/c = 0.12, ideal = 0.115 → error = 0.005 → 25 points

Performance:
  L/D_est = 1.4/0.012 = 117, required = 40 → 25 points

CL Capability:
  Range = [-0.4, 1.6], target = 1.0 → in range → 20 points

TOTAL: 35 + 15 + 25 + 25 + 20 = 120 → capped at 100 points
```

**Candidate B: Eppler E387** (glider airfoil)
```
Reynolds Match:
  Re_range = [2×10⁵, 2×10⁶]  
  1×10⁶ is within range → 35 points

Application:
  Type = "glider" (perfect match!) → 25 points

Thickness:
  t/c = 0.097, ideal = 0.115 → error = 0.018 → 25 points

Performance:
  L/D_est = 1.5/0.009 = 167, required = 40 → 25 points

CL Capability:
  Range = [0.2, 1.8], target = 1.0 → in range → 20 points

TOTAL: 35 + 25 + 25 + 25 + 20 = 130 → capped at 100 points
```

**Winner: Eppler E387** (better application match)


## SEARCH ALGORITHM COMPLEXITY
## =============================

Time Complexity:
```
Step 1 (Re filter):  O(n) where n = 686 airfoils
Step 2 (Scoring):    O(m) where m = filtered count
Step 3 (Sorting):    O(m log m)

Total: O(n + m log m) ≈ O(n log n) worst case
       Typically ~0.001 seconds for 686 airfoils
```

Space Complexity: O(n) for storing airfoil database


## WHY THIS SCORING SYSTEM?
## =========================

1. **Physics-Based Weights:**
   - Reynolds (35%): Most critical for accuracy
   - Application (25%): Mission-specific optimization
   - Thickness (25%): Drag/structure trade-off
   - Performance (25%): Overall quality
   - CL range (20%): Operating point validation

2. **Validated Ranges Preferred:**
   - Extrapolation penalized quadratically
   - Reflects increasing uncertainty with Re deviation

3. **Multi-Objective:**
   - Not just "best L/D"
   - Considers manufacturability, mission, constraints
   - Balanced solution

4. **Transparent:**
   - Each criterion independently calculated
   - User can see why airfoil was selected
   - Alternative options provided
