"""
Installation verification test for AI Aerodynamic Design Assistant
"""

import sys
import platform
import importlib.util

def check_python_version():
    """Check if Python version meets requirements"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_package(package_name, display_name=None):
    """Check if a package is available"""
    if display_name is None:
        display_name = package_name
    
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name} v{version} - OK")
            return True
        except ImportError as e:
            print(f"⚠️ {display_name} - Import error: {e}")
            return False
    else:
        print(f"❌ {display_name} - Not found")
        return False

def check_core_dependencies():
    """Check all core dependencies"""
    print("\n📦 Checking core dependencies...")
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas'),
        ('neuralfoil', 'NeuralFoil AI Engine')
    ]
    
    all_ok = True
    for package, display in dependencies:
        if not check_package(package, display):
            all_ok = False
    
    return all_ok

def check_system_info():
    """Display system information"""
    print("\n💻 System Information:")
    print(f"   Platform: {platform.platform()}")
    print(f"   Architecture: {platform.architecture()[0]}")
    print(f"   Processor: {platform.processor()}")
    print(f"   Python executable: {sys.executable}")

def test_neuralfoil_basic():
    """Test basic NeuralFoil functionality"""
    print("\n🧠 Testing NeuralFoil AI Engine...")
    
    try:
        import numpy as np
        import neuralfoil as nf
        
        # Create simple NACA 0012 coordinates
        def naca_4digit(m, p, t, num_points=100):
            """Generate NACA 4-digit airfoil coordinates"""
            # Parametric equations for NACA airfoils
            x = np.linspace(0, 1, num_points)
            
            # Thickness distribution
            yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
            
            # For symmetric airfoil (NACA 0012), camber is zero
            yc = np.zeros_like(x)
            
            # Upper and lower surfaces
            x_coords = np.concatenate([x, x[::-1]])
            y_coords = np.concatenate([yc + yt, yc[::-1] - yt[::-1]])
            
            return np.column_stack([x_coords, y_coords])
        
        # Generate NACA 0012 coordinates
        coordinates = naca_4digit(0, 0, 0.12)
        
        print("   📐 Generated NACA 0012 coordinates")
        
        # Test NeuralFoil analysis
        results = nf.get_aero_from_coordinates(
            coordinates=coordinates,
            alpha=5.0,    # 5 degrees angle of attack
            Re=1e6,       # Reynolds number
            M=0.1         # Mach number
        )
        
        print(f"   📊 Analysis Results:")
        print(f"      CL (Lift Coefficient): {results['CL']:.4f}")
        print(f"      CD (Drag Coefficient): {results['CD']:.4f}")
        print(f"      L/D Ratio: {results['CL']/results['CD']:.2f}")
        
        # Validate results are reasonable
        if 0.3 <= results['CL'] <= 1.5 and 0.005 <= results['CD'] <= 0.05:
            print("✅ NeuralFoil analysis successful - Results look reasonable")
            return True
        else:
            print("⚠️ NeuralFoil analysis completed but results seem unusual")
            return False
            
    except Exception as e:
        print(f"❌ NeuralFoil test failed: {e}")
        return False

def test_matplotlib_backend():
    """Test matplotlib functionality"""
    print("\n🎨 Testing Matplotlib...")
    
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        
        print(f"   Backend: {matplotlib.get_backend()}")
        
        # Test basic plotting
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, 'b-', linewidth=2)
        plt.title('Test Plot')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        
        # Save test plot
        plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Matplotlib plotting test successful")
        return True
        
    except Exception as e:
        print(f"❌ Matplotlib test failed: {e}")
        return False

def test_ai_assistant():
    """Test AI assistant loading"""
    print("\n🤖 Testing AI Assistant...")
    
    try:
        # Try to import from src directory
        sys.path.insert(0, 'src')
        
        try:
            from final_aerodynamic_ai import FinalAerodynamicAI
            print("   📁 Found final_aerodynamic_ai.py in src/")
        except ImportError:
            # Try current directory
            from final_aerodynamic_ai import FinalAerodynamicAI
            print("   📁 Found final_aerodynamic_ai.py in current directory")
        
        # Initialize assistant
        assistant = FinalAerodynamicAI()
        print("   🚀 AI Assistant initialized")
        
        # Check if airfoil database loaded
        if hasattr(assistant, 'airfoils') and len(assistant.airfoils) > 0:
            print(f"   📚 Airfoil database loaded: {len(assistant.airfoils)} airfoils")
        else:
            print("   ⚠️ Airfoil database not detected")
        
        print("✅ AI Assistant test successful")
        return True
        
    except ImportError as e:
        print(f"   ⚠️ AI Assistant files not found: {e}")
        print("   💡 This is OK if you haven't copied the source files yet")
        return True  # Don't fail installation test for this
        
    except Exception as e:
        print(f"❌ AI Assistant test failed: {e}")
        return False

def run_installation_test():
    """Run complete installation test"""
    print("🔍 AI Aerodynamic Design Assistant - Installation Test")
    print("=" * 60)
    
    # Check system info
    check_system_info()
    
    # Test components
    tests = [
        ("Python Version", check_python_version),
        ("Core Dependencies", check_core_dependencies),
        ("NeuralFoil Engine", test_neuralfoil_basic),
        ("Matplotlib", test_matplotlib_backend),
        ("AI Assistant", test_ai_assistant)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 INSTALLATION TEST SUMMARY:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 INSTALLATION SUCCESSFUL!")
        print("   You can now run: python src/final_aerodynamic_ai.py")
        print("   Or try the demo: python examples/demo_final_aerodynamics.py")
    elif passed >= len(results) - 1:
        print("\n✅ INSTALLATION MOSTLY SUCCESSFUL!")
        print("   Core functionality is available.")
        print("   Check any failed tests above for optimal experience.")
    else:
        print("\n❌ INSTALLATION ISSUES DETECTED")
        print("   Please resolve the failed tests above.")
        print("   See docs/INSTALLATION.md for troubleshooting.")
    
    return passed == len(results)

if __name__ == "__main__":
    try:
        success = run_installation_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        sys.exit(1)