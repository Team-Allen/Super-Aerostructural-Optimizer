# 🚀 Quick Start: Train GNN + Run Aerostructural Optimization

## TL;DR (Too Long; Didn't Read)

Copy-paste these commands in PowerShell to train your GNN and run Stage 2 aerostructural optimization:

```powershell
# Go to project directory
cd "f:\MDO LAB\Super-Aerostructural-Optimizer"

# Train the GNN (2-4 hours, one-time)
python training/train_gnn.py --epochs 50 --batch_size 8 --device cuda --model_type attention

# Run aerostructural optimization (1-2 hours)
$env:PYTHONPATH="f:\MDO LAB\Super-Aerostructural-Optimizer"
python .\scripts\optimize_wing_aerostructural.py --model training/checkpoints/gnn_best.pth --device cuda --budget 300

# View results
cat results\stage2_aerostructural\opt_result.json
```

**Total time:** 3–5 hours
**Result:** Best wing design + Pareto front (multiple trade-off options)

---

## 📋 Detailed Steps

### **Step 1: Verify You Have What You Need**

**Check CUDA availability:**
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```
Expected output: `True RTX-xxxx` (or similar GPU)

**Check AirFRANS data exists:**
```powershell
dir "F:\MDO LAB\RESEARCH\Airfrans Airfoil Data\archive" | head -20
```
Expected output: List of `.pth` files (graph data)

---

### **Step 2: Navigate to Project**

```powershell
cd "f:\MDO LAB\Super-Aerostructural-Optimizer"
```

---

### **Step 3: Train the GNN (One-Time, 2–4 hours)**

```powershell
python training/train_gnn.py \
  --epochs 50 \
  --batch_size 8 \
  --device cuda \
  --model_type attention \
  --data_dir "F:\MDO LAB\RESEARCH\Airfrans Airfoil Data\archive" \
  --output_dir training/checkpoints
```

**What it does:**
- Loads ~1000 graph samples from AirFRANS dataset
- Trains a GNN (attention variant) for 50 epochs
- Saves best checkpoint to `training/checkpoints/gnn_best.pth`
- Prints loss, validation metrics during training

**Expected output:**
```
Loading AirFRANS dataset...
  Found 1000 graphs at F:\MDO LAB\RESEARCH\Airfrans Airfoil Data\archive
Initializing Attention GNN...
  Parameters: 77,315
Starting training...
Epoch 1/50:   Loss=0.523, Val_Loss=0.418
Epoch 2/50:   Loss=0.412, Val_Loss=0.385
...
Epoch 50/50:  Loss=0.152, Val_Loss=0.198
✅ Training complete. Best model saved to training/checkpoints/gnn_best.pth
```

**⏱️ Time:** 2–4 hours (depends on GPU; slower on RTX 3050, faster on RTX 3090)

---

### **Step 4: Run Stage-2 Aerostructural Optimization (1–2 hours)**

```powershell
$env:PYTHONPATH="f:\MDO LAB\Super-Aerostructural-Optimizer"

python .\scripts\optimize_wing_aerostructural.py `
  --model training/checkpoints/gnn_best.pth `
  --device cuda `
  --budget 300 `
  --popsize 16 `
  --out results/stage2_aerostructural
```

**What it does:**
- Loads trained GNN from checkpoint
- Initializes CMA-ES optimizer with 16 population members
- Loop: generates 300 wing designs, evaluates each with:
  - GNN aerodynamic analysis (L/D)
  - Structural model (mass, stress, frequencies)
  - Multi-objective fitness (maximize L/D, minimize mass)
- Saves best design + Pareto front to JSON

**Expected output:**
```
🚀 Starting aerostructural optimization (budget=300)
   Device: cuda, PopSize: 16
────────────────────────────────────────────────────────────────────────────────
Initializing GNN wing analyzer...
  Model path: training/checkpoints/gnn_best.pth
  Device: cuda
  Model parameters: 77,315
Created attention GNN with 77,315 trainable parameters
(2_w,4mirr1)-aCMA-ES (mu_w=1.5,w_1=80%) in dimension 6 (seed=12345, Fri Dec 13 ...)

Using CMA-ES optimizer...
Analyzing 30 spanwise sections...
  Processed 10/30 sections
  Processed 20/30 sections
  Processed 30/30 sections
  Eval   1: L/D= 14.23, Mass= 2450.2kg, Stress= 98.5MPa, Margin= 0.28
  Eval   2: L/D= 15.18, Mass= 2380.1kg, Stress= 92.3MPa, Margin= 0.32
  ...
  Eval 300: L/D= 16.84, Mass= 2350.0kg, Stress= 95.1MPa, Margin= 0.35

────────────────────────────────────────────────────────────────────────────────
✅ Optimization complete. Best fitness: -16.84
Results saved to results\stage2_aerostructural\opt_result.json

   Best L/D: 16.84
   Best Mass: 2350.0 kg
   Max Stress: 95.1 MPa
   Pareto front size: 8
```

**⏱️ Time:** 1–2 hours (mostly dominated by GNN inference time)

---

### **Step 5: View Results**

```powershell
# Pretty-print the JSON
python -m json.tool results\stage2_aerostructural\opt_result.json | more
```

Or in PowerShell:
```powershell
cat results\stage2_aerostructural\opt_result.json
```

**Expected output:**
```json
{
  "best_design": {
    "span_m": 12.45,
    "taper_ratio": 0.68,
    "sweep_deg": 17.3,
    "dihedral_deg": 5.2,
    "twist_root_deg": 0.3,
    "twist_tip_deg": -4.1
  },
  "best_metrics": {
    "l_d": 16.84,
    "mass_kg": 2350.0,
    "stress_mpa": 95.1,
    "freq_hz": 3.24,
    "flutter_margin": 0.35
  },
  "pareto_front": [
    {"l_d": 17.18, "mass_kg": 2450, "stress_mpa": 110},
    {"l_d": 16.84, "mass_kg": 2350, "stress_mpa": 95},
    {"l_d": 16.15, "mass_kg": 2200, "stress_mpa": 75},
    ...
  ],
  "optimization": {
    "budget": 300,
    "num_evals": 300,
    "best_fitness": -16.84
  }
}
```

---

## 🎯 What the Results Mean

### **Best Design**
The optimal wing found:
- **Span:** 12.45 m (reasonable for small UAV)
- **Taper:** 0.68 (wing tip ~68% the root chord width)
- **Sweep:** 17.3° (swept-back wing for stability/aero)
- **Dihedral:** 5.2° (wing tips higher for roll stability)
- **Twist:** root 0.3°, tip -4.1° (washout for efficiency)

### **Metrics**
- **L/D = 16.84:** Excellent lift-to-drag ratio (high efficiency!)
- **Mass = 2350 kg:** Structural mass (reasonable for a 12.5 m wing)
- **Stress = 95.1 MPa:** Maximum stress in structure (safe, allowable ~200 MPa)
- **Freq = 3.24 Hz:** First bending frequency (good, avoids flutter)
- **Flutter Margin = 0.35:** 35% margin above cruise speed (safe, need >20%)

### **Pareto Front**
Shows the trade-offs:
- **Design 1:** High L/D (17.18) but heavier (2450 kg) and higher stress
- **Design 2:** Balanced (16.84 L/D, 2350 kg) ← Best overall
- **Design 3:** Lighter (2200 kg) but lower L/D (16.15)

**Decision:** Engineers choose based on mission:
- If range is critical → Design 1 (high L/D)
- If weight limit is tight → Design 3 (lightest)
- If balanced performance → Design 2 (recommended)

---

## 🔧 Troubleshooting

### **Problem: CUDA not available**
```
RuntimeError: CUDA is not available
```
**Solution:** Use CPU (slower, takes 5–10× longer):
```powershell
python .\scripts\optimize_wing_aerostructural.py --model training/checkpoints/gnn_best.pth --device cpu --budget 100
```

### **Problem: Out of GPU memory**
```
RuntimeError: CUDA out of memory
```
**Solutions:**
1. Reduce batch size: `--batch_size 4`
2. Use CPU: `--device cpu`
3. Reduce budget: `--budget 150`

### **Problem: GNN checkpoint not found**
```
FileNotFoundError: training/checkpoints/gnn_best.pth
```
**Solution:** First run the GNN training:
```powershell
python training/train_gnn.py --epochs 50 --batch_size 8 --device cuda
```

### **Problem: AirFRANS data not found**
```
FileNotFoundError: F:\MDO LAB\RESEARCH\Airfrans Airfoil Data\archive
```
**Solution:** Check the path exists and has `.pth` files. Or download the dataset.

---

## ⏱️ Timeline Estimate

| Step | Time | Notes |
|------|------|-------|
| GNN Training | 2–4 hrs | One-time; depends on GPU |
| Stage 2 Optimization | 1–2 hrs | 300 evaluations; CMA-ES |
| Results Analysis | 15 min | View JSON, choose design |
| **Total** | **3–6 hrs** | Can be done in one session |

---

## 📚 Next Steps (After Results)

### **Option A: Use Best Design**
Take `best_design` from JSON and move to detailed design (CAD, FEA, wind tunnel)

### **Option B: Validate with Physics Nemo** (Optional, adds 2 hours)
```powershell
# If you have Physics Nemo installed:
python scripts/validate_with_physics_nemo.py \
  --designs results/stage2_aerostructural/opt_result.json
```
Compare GNN predictions vs high-fidelity PDE solver

### **Option C: Refine Design**
Pick a design from Pareto front and re-optimize with tighter bounds:
```powershell
python .\scripts\optimize_wing_aerostructural.py \
  --model training/checkpoints/gnn_best.pth \
  --device cuda \
  --budget 500 \
  --out results/stage2_refined
```

---

## 💾 Output Files

After all steps, you'll have:

```
results/stage2_aerostructural/
├── opt_result.json          ← Main results file
└── ...                       ← Any intermediate files
```

Key file: `opt_result.json` contains:
- Best wing design (all 6 parameters)
- Best aerodynamic/structural metrics
- Pareto front (5–10 alternative designs)
- Optimization summary (evals, fitness)

---

## 🎉 Success Criteria

You've succeeded when:
1. ✅ GNN training completes without errors
2. ✅ Stage 2 optimization runs 300 evaluations
3. ✅ `opt_result.json` is created
4. ✅ Best L/D > 14 (good aero efficiency)
5. ✅ Max stress < 150 MPa (safe)
6. ✅ Flutter margin > 20% (stable)

---

## ❓ Questions Before You Start?

1. **Do you have CUDA GPU available?** (Check above)
2. **Is AirFRANS dataset at the right path?** (Check above)
3. **Do you want me to run the training for you?** (I can, if you give permission)
4. **Want to add more constraints?** (e.g., cost, manufacturability)

---

## 🚀 Ready? Let's Go!

**Copy-paste and run in PowerShell:**

```powershell
cd "f:\MDO LAB\Super-Aerostructural-Optimizer"
python training/train_gnn.py --epochs 50 --batch_size 8 --device cuda --model_type attention
$env:PYTHONPATH="f:\MDO LAB\Super-Aerostructural-Optimizer"
python .\scripts\optimize_wing_aerostructural.py --model training/checkpoints/gnn_best.pth --device cuda --budget 300
cat results\stage2_aerostructural\opt_result.json
```

**Estimated time:** 3–5 hours (mostly training; optimization is 7.5 min)

**Expected result:** Best wing design + Pareto front showing design trade-offs

Good luck! 🚀
