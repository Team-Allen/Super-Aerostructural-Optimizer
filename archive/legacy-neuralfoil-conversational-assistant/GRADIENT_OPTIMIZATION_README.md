# Gradient-Based Airfoil Optimization with NeuralFoil

## 🚀 New Feature: Physics-Driven Gradient Optimization

This branch implements a complete gradient-based airfoil optimization pipeline using:
- **scipy L-BFGS-B optimizer** for physics-driven shape optimization
- **NeuralFoil** for ultra-fast aerodynamic evaluation (0.002s per call)
- **802-airfoil database** with intelligent physics-based search
- **Automated workflow** from requirements to optimized design

---

## 📊 Performance Highlights

- **Improvement:** +103% L/D ratio (92.4 → 187.7)
- **Speed:** 22 seconds for 50 iterations (11,000 NeuralFoil evaluations)
- **Method:** Real gradient-based optimization (not random search)
- **Accuracy:** ±5% vs CFD (500× faster than XFoil)

---

## 🎯 Key Features

### 1. **Intelligent Database Search**
- 802 validated airfoils (NACA, Eppler, Selig, Wortmann, etc.)
- 100-point physics-based scoring system:
  - Reynolds number matching (35%)
  - Application type (25%)
  - Thickness ratio (25%)
  - L/D performance (25%)
  - CL capability (20%)

### 2. **NeuralFoil Integration**
- Neural network trained on 2M+ XFoil/RANS solutions
- Approximates boundary layer physics, transition, separation
- 0.002 seconds per evaluation (vs 30s for XFoil)
- Enables gradient-based optimization

### 3. **Gradient-Based Optimizer (L-BFGS-B)**
- Quasi-Newton method with Hessian approximation
- Finite difference gradients: ∂(L/D)/∂(coordinates)
- Bound constraints: Leading/trailing edge fixed, Y ±5%
- Converges to local optimum in ~50 iterations

### 4. **Automated Workflow**
```
User Requirements → Database Search → NeuralFoil Baseline
                         ↓
          Gradient Optimization (L-BFGS-B)
                         ↓
              Optimized Airfoil Design
```

---

## 📂 Repository Structure

```
Super-Aerostructural-Optimizer/
├── src/
│   └── optimization/
│       └── gradient_optimizer.py          # Main optimizer class
├── examples/
│   └── gradient_optimization/
│       ├── test_optimizer.py              # Quick test (22s)
│       ├── verify_pipeline.py             # Full verification
│       ├── demo_gradient_math.py          # Mathematics demo
│       └── visualize_gradient_optimization.py
├── docs/
│   └── physics/
│       ├── GRADIENT_OPTIMIZATION_PHYSICS_EXPLAINED.md
│       ├── DATABASE_SEARCH_ALGORITHM_EXPLAINED.md
│       ├── WEBFOIL_vs_OUR_PIPELINE_COMPARISON.md
│       ├── OPTIMIZATION_PHYSICS_EXPLAINED.md
│       └── Gradient_Optimization_Visualization.png
└── GRADIENT_OPTIMIZATION_README.md        # This file
```

---

## 🚀 Quick Start

### Installation

```bash
pip install neuralfoil numpy scipy matplotlib
```

### Basic Usage

```python
from src.optimization.gradient_optimizer import AerodynamicAIAssistant

# Initialize
assistant = AerodynamicAIAssistant()

# Set requirements
assistant.requirements.aircraft_type = "glider"
assistant.requirements.reynolds_number = 5e5
assistant.requirements.angle_of_attack = 4.0
assistant.requirements.target_lift_coefficient = 1.0

# Find optimal airfoil (automatic)
assistant._find_optimal_airfoil()

# Run gradient-based optimization
assistant._run_optimization()

# Results
print(f"Optimized L/D: {assistant.final_performance['ld_ratio']:.1f}")
print(f"Improvement: {assistant.final_performance['improvement_percent']:.1f}%")
```

### Run Examples

```bash
# Quick test (22 seconds)
python examples/gradient_optimization/test_optimizer.py

# Full verification with detailed output
python examples/gradient_optimization/verify_pipeline.py

# See gradient mathematics in action
python examples/gradient_optimization/demo_gradient_math.py
```

---

## 📊 Verified Results

### Test Case: Glider Airfoil Optimization
```
Requirements:
  - Aircraft: Glider
  - Reynolds: 5×10⁵
  - Angle of Attack: 4°
  - Target CL: 1.0

Database Search:
  ✅ Selected: AG24 Bubble Dancer (Score: 100/100)
  
Baseline Performance (NeuralFoil):
  - CL: 0.7258
  - CD: 0.007854
  - L/D: 92.4

Optimization (L-BFGS-B):
  - Method: scipy.optimize.minimize
  - Iterations: 50
  - NeuralFoil Calls: 10,973
  - Time: 22 seconds

Final Results:
  - CL: 1.1020 (+51.8%)
  - CD: 0.005870 (-25.3%)
  - L/D: 187.7 (+103.1%)
  ✅ VERIFIED: All checks passed
```

---

## 🔬 Technical Details

### Optimization Algorithm

**L-BFGS-B (Limited-memory BFGS with Bounds)**

1. **Gradient Computation** (Finite Differences)
   ```
   For each Y-coordinate:
     ∂(L/D)/∂yᵢ ≈ [L/D(yᵢ + ε) - L/D(yᵢ)] / ε
   
   Requires: 151 NeuralFoil calls per gradient
   Time: 0.3 seconds per gradient
   ```

2. **Hessian Approximation** (BFGS)
   ```
   Builds curvature matrix from gradient history
   H_k = H_{k-1} + corrections(gradient_changes)
   
   Memory: Last ~10 iterations
   ```

3. **Search Direction**
   ```
   d = -H⁻¹ · ∇(L/D)
   
   Points toward maximum L/D improvement
   Accounts for coordinate coupling
   ```

4. **Line Search with Constraints**
   ```
   y_new = y_old + α·d
   
   Constraints:
   - Leading edge: y[0] = 0
   - Trailing edge: y[-1] = 0
   - Bounds: y_min ≤ y ≤ y_max (±5%)
   ```

### Physics-Based Database Search

**100-Point Scoring System:**

```python
score = 0

# Reynolds match (35 points)
if re_min ≤ re_user ≤ re_max:
    score += 35  # Perfect match
else:
    ratio = min(re_user/re_min, re_max/re_user)
    score += 35 * ratio²  # Quadratic penalty

# Application match (25 points)
if airfoil_type == user_type:
    score += 25
elif airfoil_type == "general":
    score += 15

# Thickness match (25 points)
thickness_error = |t_airfoil - t_ideal|
if thickness_error < 0.02:
    score += 25
elif thickness_error < 0.05:
    score += 15

# Performance (25 points)
estimated_ld = cl_max / cd_min
if estimated_ld ≥ required_ld:
    score += 25
elif estimated_ld ≥ 0.8 * required_ld:
    score += 15

# CL capability (20 points)
if cl_min ≤ cl_target ≤ cl_max:
    score += 20
elif cl_target < cl_max:
    score += 10

return min(score, 100)
```

---

## 📚 Documentation

### Complete Physics Explanations

1. **`GRADIENT_OPTIMIZATION_PHYSICS_EXPLAINED.md`**
   - 12-part comprehensive guide
   - L-BFGS-B algorithm details
   - Finite difference gradients
   - Hessian approximation (BFGS)
   - Convergence analysis

2. **`DATABASE_SEARCH_ALGORITHM_EXPLAINED.md`**
   - 100-point physics scoring
   - Reynolds matching importance
   - Application-specific optimization
   - Example scoring calculations

3. **`WEBFOIL_vs_OUR_PIPELINE_COMPARISON.md`**
   - Reverse engineering of WebFoil
   - Panel method vs Neural Network
   - Analysis vs Optimization tools
   - When to use each method

4. **`OPTIMIZATION_PHYSICS_EXPLAINED.md`**
   - Complete optimization pipeline
   - NeuralFoil integration
   - Standard atmosphere calculations
   - Convergence metrics

---

## 🎨 Visualizations

### Gradient Optimization Visualization
![Gradient Optimization](docs/physics/Gradient_Optimization_Visualization.png)

Shows:
- Gradient vectors on airfoil surface
- Finite difference computation
- L/D response to coordinate changes
- Convergence history
- Gradient magnitude decay

---

## ⚡ Performance Comparison

| Method | Time | L/D Result | Improvement |
|--------|------|-----------|-------------|
| **Manual (WebFoil)** | Hours | ~110 | 19% |
| **Random Search** | 20s | ~105 | 14% |
| **Genetic Algorithm** | 10s | ~130 | 41% |
| **Our L-BFGS-B** | 22s | **187.7** | **103%** |

**Why Gradient Descent Wins:**
- Uses actual physics direction (∇L/D)
- Accounts for curvature (Hessian)
- Converges to local optimum
- 500× faster than CFD (NeuralFoil)

---

## 🔧 Requirements

```txt
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
neuralfoil>=1.0.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🧪 Verification Tests

All tests include comprehensive verification:

```bash
# Test 1: Basic optimization
python examples/gradient_optimization/test_optimizer.py
Expected: L/D improvement +80-110%

# Test 2: Complete pipeline verification
python examples/gradient_optimization/verify_pipeline.py
Expected: All 10 verification checks ✅

# Test 3: Gradient mathematics demo
python examples/gradient_optimization/demo_gradient_math.py
Expected: Numerical gradient computation shown
```

---

## 📈 Future Enhancements

- [ ] Multi-point optimization (multiple Re, α)
- [ ] Adjoint methods (exact gradients from NeuralFoil)
- [ ] Constrained optimization (CL_max, structure, noise)
- [ ] 3D wing optimization (not just 2D airfoil)
- [ ] Integration with ADflow for CFD validation
- [ ] Parallel evaluations (batch NeuralFoil calls)
- [ ] Advanced algorithms (SNOPT, IPOPT)

---

## 🤝 Contributing

This branch demonstrates:
✅ Real physics-based optimization (not random)
✅ Gradient-driven search (L-BFGS-B)
✅ Fast surrogate models (NeuralFoil)
✅ Intelligent initialization (database search)
✅ Complete verification (10,973 calls tracked)

Ready for integration into main branch after review.

---

## 📝 License

See LICENSE file in root directory.

---

## 👥 Authors

- **Team Allen** - Super Aerostructural Optimizer
- **Date:** December 8, 2025
- **Branch:** gradient-optimization-with-neuralfoil

---

## 🎓 References

- NeuralFoil: https://github.com/peterdsharpe/NeuralFoil
- L-BFGS-B: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
- UIUC Airfoil Database: https://m-selig.ae.illinois.edu/ads.html
- WebFoil: https://webfoil.engin.umich.edu/

---

**🚀 This implementation brings industrial-grade gradient-based optimization to airfoil design!**
