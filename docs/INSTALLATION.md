# 📦 Installation Guide

## 🎯 **Overview**

This guide provides comprehensive installation instructions for the AI-Aerodynamic Design Assistant. The system requires Python 3.8+ and several scientific computing packages.

---

## 📋 **System Requirements**

### **Minimum Requirements:**
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB free space
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)

### **Recommended Setup:**
- **Python**: 3.9+ (best compatibility with NeuralFoil)
- **RAM**: 8GB or more (for smooth real-time visualization)
- **Storage**: 1GB+ (for datasets and examples)
- **Display**: 1920x1080+ (for optimal visualization experience)

---

## ⚡ **Quick Installation**

### **Option 1: Pip Installation (Recommended)**
```bash
# Install all dependencies at once
pip install neuralfoil numpy matplotlib pandas scipy dataclasses

# Clone the repository
git clone https://github.com/your-username/AI-Aerodynamic-Design-Assistant.git
cd AI-Aerodynamic-Design-Assistant

# Test installation
python examples/demo_final_aerodynamics.py
```

### **Option 2: Using Requirements File**
```bash
# Clone repository first
git clone https://github.com/your-username/AI-Aerodynamic-Design-Assistant.git
cd AI-Aerodynamic-Design-Assistant

# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Verify installation
python tests/test_installation.py
```

---

## 🔧 **Detailed Installation Steps**

### **Step 1: Python Environment Setup**

#### **Windows:**
```powershell
# Check Python version
python --version

# Should output: Python 3.8.x or higher
# If not installed, download from https://python.org

# Create virtual environment (recommended)
python -m venv ai_aerodynamics_env

# Activate virtual environment
ai_aerodynamics_env\Scripts\activate

# Verify activation (should show environment name in prompt)
```

#### **macOS:**
```bash
# Check Python version
python3 --version

# Install Python via Homebrew if needed
brew install python

# Create virtual environment
python3 -m venv ai_aerodynamics_env

# Activate virtual environment
source ai_aerodynamics_env/bin/activate
```

#### **Linux (Ubuntu/Debian):**
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv ai_aerodynamics_env

# Activate virtual environment
source ai_aerodynamics_env/bin/activate
```

### **Step 2: Core Dependencies Installation**

#### **NeuralFoil (Primary AI Engine):**
```bash
# Install NeuralFoil - the core AI aerodynamics engine
pip install neuralfoil

# Verify NeuralFoil installation
python -c "import neuralfoil; print('NeuralFoil version:', neuralfoil.__version__)"
```

**Troubleshooting NeuralFoil:**
```bash
# If pip install fails, try:
pip install --upgrade pip setuptools wheel
pip install neuralfoil

# If still failing, install dependencies first:
pip install torch torchvision  # PyTorch dependencies
pip install neuralfoil

# Alternative: conda installation
conda install -c conda-forge neuralfoil
```

#### **Scientific Computing Stack:**
```bash
# NumPy for numerical computations
pip install numpy>=1.19.0

# Matplotlib for visualization
pip install matplotlib>=3.3.0

# SciPy for scientific algorithms
pip install scipy>=1.6.0

# Pandas for data handling
pip install pandas>=1.2.0
```

#### **Additional Utilities:**
```bash
# Dataclasses for Python 3.6 compatibility (if needed)
pip install dataclasses

# Type hints support
pip install typing-extensions

# Progress bars for optimization
pip install tqdm
```

### **Step 3: Repository Setup**
```bash
# Clone the repository
git clone https://github.com/your-username/AI-Aerodynamic-Design-Assistant.git

# Navigate to directory
cd AI-Aerodynamic-Design-Assistant

# Verify directory structure
ls -la

# Should see:
# src/        - Source code
# examples/   - Usage examples
# docs/       - Documentation
# tests/      - Test suite
```

### **Step 4: Installation Verification**
```bash
# Run installation test
python tests/test_installation.py

# Expected output:
# ✅ Python version check passed
# ✅ NeuralFoil engine loaded
# ✅ All dependencies available
# ✅ Installation successful!

# Run demo to verify full functionality
python examples/demo_final_aerodynamics.py
```

---

## 🐳 **Docker Installation**

For containerized deployment:

### **Using Pre-built Image:**
```bash
# Pull the image
docker pull your-username/ai-aerodynamic-assistant:latest

# Run container
docker run -it --rm \
    -v $(pwd)/output:/app/output \
    your-username/ai-aerodynamic-assistant:latest
```

### **Building from Source:**
```bash
# Clone repository
git clone https://github.com/your-username/AI-Aerodynamic-Design-Assistant.git
cd AI-Aerodynamic-Design-Assistant

# Build Docker image
docker build -t ai-aerodynamic-assistant .

# Run container with volume mounting for outputs
docker run -it --rm \
    -v $(pwd)/output:/app/output \
    ai-aerodynamic-assistant
```

**Dockerfile Contents:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run tests
RUN python tests/test_installation.py

# Default command
CMD ["python", "src/final_aerodynamic_ai.py"]
```

---

## 🔍 **Platform-Specific Instructions**

### **Windows Specific:**

#### **PowerShell Execution Policy:**
```powershell
# If you get execution policy errors:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Verify policy change:
Get-ExecutionPolicy
```

#### **Visual Studio Build Tools (if compilation needed):**
```powershell
# If you see "Microsoft Visual C++ 14.0 is required" error:
# Download and install Microsoft C++ Build Tools from:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### **Windows Path Issues:**
```powershell
# If Python not found in PATH:
# 1. Reinstall Python with "Add to PATH" option checked
# 2. Or manually add Python to PATH:
$env:PATH += ";C:\Python39;C:\Python39\Scripts"
```

### **macOS Specific:**

#### **Xcode Command Line Tools:**
```bash
# Install command line tools (needed for some packages)
xcode-select --install
```

#### **Homebrew Package Manager:**
```bash
# Install Homebrew if not available
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies via Homebrew
brew install python
brew install git
```

#### **M1/M2 Mac Compatibility:**
```bash
# For Apple Silicon Macs, use x86 compatibility if needed:
arch -x86_64 pip install neuralfoil

# Or install via Miniforge (ARM64 native):
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh
conda install neuralfoil
```

### **Linux Specific:**

#### **Ubuntu/Debian Dependencies:**
```bash
# Install required system packages
sudo apt update
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev

# Install git if not available
sudo apt install git
```

#### **CentOS/RHEL/Fedora:**
```bash
# Install development tools
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel python3-pip

# Or for older versions:
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel python3-pip
```

#### **Arch Linux:**
```bash
# Install dependencies
sudo pacman -S python python-pip git base-devel

# Install via AUR if available
yay -S python-neuralfoil  # If available in AUR
```

---

## 🧪 **Development Environment Setup**

For contributors and developers:

### **Full Development Setup:**
```bash
# Clone with development dependencies
git clone https://github.com/your-username/AI-Aerodynamic-Design-Assistant.git
cd AI-Aerodynamic-Design-Assistant

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # Linux/macOS
# dev_env\Scripts\activate    # Windows

# Install package in editable mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### **Development Dependencies:**
```bash
# Testing framework
pip install pytest pytest-cov

# Code formatting
pip install black isort

# Type checking
pip install mypy

# Code quality
pip install flake8 pylint

# Documentation
pip install sphinx sphinx-rtd-theme

# Jupyter notebooks for experimentation
pip install jupyter jupyterlab
```

### **IDE Configuration:**

#### **VS Code Setup:**
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./dev_env/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.testing.pytestEnabled": true
}
```

#### **PyCharm Setup:**
1. Open project in PyCharm
2. Go to File → Settings → Project → Python Interpreter
3. Add New Environment → Virtual Environment
4. Point to your dev_env directory
5. Install requirements from requirements-dev.txt

---

## ✅ **Verification and Testing**

### **Basic Functionality Test:**
```python
# test_basic_functionality.py
"""
Basic functionality test script
"""

def test_imports():
    """Test all critical imports"""
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        import neuralfoil as nf
        print("✅ NeuralFoil imported successfully")
        
        from src.final_aerodynamic_ai import FinalAerodynamicAI
        print("✅ AI Assistant imported successfully")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    return True

def test_neuralfoil_basic():
    """Test basic NeuralFoil functionality"""
    try:
        import neuralfoil as nf
        
        # Create simple NACA 0012 coordinates
        x = np.linspace(0, 1, 100)
        y_upper = 0.12 / 0.2 * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
        y_lower = -y_upper
        
        coordinates = np.column_stack([
            np.concatenate([x, x[::-1]]),
            np.concatenate([y_upper, y_lower[::-1]])
        ])
        
        # Test analysis
        results = nf.get_aero_from_coordinates(
            coordinates=coordinates,
            alpha=5.0,
            Re=1e6,
            M=0.1
        )
        
        print(f"✅ NeuralFoil analysis successful")
        print(f"   CL: {results['CL']:.3f}")
        print(f"   CD: {results['CD']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ NeuralFoil test failed: {e}")
        return False

def test_ai_assistant():
    """Test AI assistant initialization"""
    try:
        from src.final_aerodynamic_ai import FinalAerodynamicAI
        
        assistant = FinalAerodynamicAI()
        print("✅ AI Assistant initialized successfully")
        
        # Test airfoil database loading
        if hasattr(assistant, 'airfoils') and len(assistant.airfoils) > 0:
            print(f"✅ Airfoil database loaded: {len(assistant.airfoils)} airfoils")
        else:
            print("⚠️ Airfoil database may not be loaded properly")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Assistant test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Installation...")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_neuralfoil_basic()
    success &= test_ai_assistant()
    
    print("=" * 50)
    if success:
        print("🎉 All tests passed! Installation is successful!")
        print("You can now run: python src/final_aerodynamic_ai.py")
    else:
        print("❌ Some tests failed. Please check the installation.")
```

### **Performance Test:**
```bash
# Run performance benchmark
python tests/test_performance.py

# Expected output:
# 🚀 Performance Test Results:
# ⏱️ Single airfoil analysis: 0.08s
# ⏱️ Optimization (50 iterations): 4.2s
# ⏱️ Visualization update: 0.02s
# ✅ Performance meets requirements!
```

---

## 🚨 **Troubleshooting**

### **Common Issues and Solutions:**

#### **Issue 1: "ModuleNotFoundError: No module named 'neuralfoil'"**
```bash
# Solution 1: Verify pip installation
pip list | grep neuralfoil

# If not listed, install:
pip install neuralfoil

# Solution 2: Check Python environment
which python  # Should point to your virtual environment

# Solution 3: Reinstall with verbose output
pip install --verbose neuralfoil
```

#### **Issue 2: "ImportError: cannot import name 'get_aero_from_coordinates'"**
```bash
# This indicates an outdated NeuralFoil version
pip install --upgrade neuralfoil

# Verify version (should be 0.1.0 or higher)
python -c "import neuralfoil; print(neuralfoil.__version__)"
```

#### **Issue 3: Matplotlib backend issues**
```python
# Add to beginning of your script:
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' on Linux
import matplotlib.pyplot as plt

# Test backend:
print(matplotlib.get_backend())
```

#### **Issue 4: "Permission denied" on Windows**
```powershell
# Run as administrator or change execution policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Alternative: Use conda instead of pip:
conda install -c conda-forge neuralfoil numpy matplotlib
```

#### **Issue 5: Slow performance**
```python
# Check if running on integrated graphics:
import matplotlib
print(matplotlib.get_backend())

# Switch to faster backend:
matplotlib.use('Agg')  # For non-interactive plots
# Or install better backend:
pip install PyQt5  # Then use 'Qt5Agg'
```

### **Getting Help:**

#### **Log Collection:**
```bash
# Run with verbose logging to diagnose issues:
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from src.final_aerodynamic_ai import FinalAerodynamicAI
assistant = FinalAerodynamicAI()
" > installation_debug.log 2>&1

# Send installation_debug.log with your issue report
```

#### **System Information Collection:**
```python
# system_info.py
import sys
import platform
import pkg_resources

print("🔍 System Information:")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")

print("\n📦 Installed Packages:")
installed_packages = [d for d in pkg_resources.working_set]
for package in sorted(installed_packages, key=lambda x: x.project_name.lower()):
    print(f"{package.project_name}: {package.version}")
```

---

## 🎉 **Next Steps**

After successful installation:

1. **Run the demo:**
   ```bash
   python examples/demo_final_aerodynamics.py
   ```

2. **Try interactive mode:**
   ```bash
   python src/final_aerodynamic_ai.py
   ```

3. **Explore examples:**
   ```bash
   cd examples/
   python interactive_example.py
   python batch_processing_example.py
   ```

4. **Read the documentation:**
   - [API Reference](docs/API_REFERENCE.md)
   - [Technical Details](docs/TECHNICAL_DETAILS.md)
   - [Usage Examples](examples/)

---

**🚀 You're now ready to design aircraft with AI! Happy optimizing! ✈️**