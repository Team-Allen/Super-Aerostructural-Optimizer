# Contributing to AI Aerodynamic Design Assistant

We welcome contributions to the AI Aerodynamic Design Assistant project! This document provides guidelines for contributing to the project.

## 🎯 **Ways to Contribute**

- 🐛 **Bug Reports**: Report issues or unexpected behavior
- 💡 **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve documentation, examples, or tutorials
- 🔧 **Code Contributions**: Fix bugs, implement features, or optimize performance
- 🧪 **Testing**: Add test cases or improve test coverage
- 🎨 **Visualization**: Enhance plots, dashboards, or user interface

## 📋 **Before You Start**

1. **Check existing issues**: Look through existing issues to avoid duplicates
2. **Discuss major changes**: For significant features, create an issue first to discuss the approach
3. **Follow the code style**: We use Black for formatting and follow PEP 8 guidelines

## 🚀 **Getting Started**

### **1. Fork and Clone**
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/your-username/AI-Aerodynamic-Design-Assistant.git
cd AI-Aerodynamic-Design-Assistant
```

### **2. Set Up Development Environment**
```bash
# Create virtual environment
python -m venv dev_env
source dev_env/bin/activate  # Linux/macOS
# dev_env\Scripts\activate    # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

### **3. Create a Branch**
```bash
# Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

## 🔧 **Development Guidelines**

### **Code Style**
- Use **Black** for code formatting: `black src/ tests/ examples/`
- Use **isort** for import sorting: `isort src/ tests/ examples/`
- Follow **PEP 8** guidelines
- Use **type hints** for all function parameters and return values
- Write **docstrings** for all public functions and classes

### **Testing**
- Write tests for new features using **pytest**
- Maintain or improve test coverage
- Run tests before submitting: `pytest tests/`
- Include both unit tests and integration tests

### **Documentation**
- Update docstrings for any modified functions
- Add examples for new features
- Update README.md if adding major features
- Include type hints in documentation

## 📝 **Code Style Examples**

### **Function Definition**
```python
def analyze_airfoil_performance(
    coordinates: np.ndarray,
    reynolds_number: float,
    mach_number: float,
    angle_of_attack: float
) -> AerodynamicResults:
    """
    Analyze airfoil aerodynamic performance using NeuralFoil.
    
    Args:
        coordinates: Airfoil coordinate array shape (n, 2)
        reynolds_number: Flight Reynolds number
        mach_number: Flight Mach number
        angle_of_attack: Angle of attack in degrees
    
    Returns:
        Comprehensive aerodynamic analysis results
    
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If analysis fails
    
    Example:
        >>> coords = generate_naca_coordinates("0012")
        >>> results = analyze_airfoil_performance(coords, 1e6, 0.1, 5.0)
        >>> print(f"L/D ratio: {results.lift_to_drag_ratio:.2f}")
    """
    # Implementation here
    pass
```

### **Class Definition**
```python
class AerodynamicOptimizer:
    """
    Advanced aerodynamic optimization engine.
    
    This class provides real-time airfoil optimization using genetic
    algorithms and neural network-based fitness evaluation.
    
    Attributes:
        population_size: Number of individuals in genetic algorithm
        max_generations: Maximum optimization iterations
        neuralfoil_engine: AI analysis engine instance
    
    Example:
        >>> optimizer = AerodynamicOptimizer(population_size=50)
        >>> result = optimizer.optimize(base_airfoil, requirements)
    """
    
    def __init__(self, population_size: int = 50, max_generations: int = 100):
        """Initialize optimizer with specified parameters."""
        self.population_size = population_size
        self.max_generations = max_generations
        self.neuralfoil_engine = NeuralFoilEngine()
```

## 🧪 **Testing Guidelines**

### **Test Structure**
```python
import pytest
import numpy as np
from src.final_aerodynamic_ai import FinalAerodynamicAI

class TestAerodynamicAnalysis:
    """Test suite for aerodynamic analysis functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ai_assistant = FinalAerodynamicAI()
        self.test_coordinates = self.generate_naca_0012()
    
    def test_basic_analysis(self):
        """Test basic aerodynamic analysis functionality."""
        results = self.ai_assistant.analyze_airfoil(
            self.test_coordinates,
            reynolds=1e6,
            mach=0.1,
            alpha=5.0
        )
        
        assert results is not None
        assert 0 < results.lift_coefficient < 2.0
        assert 0 < results.drag_coefficient < 0.1
        assert results.lift_to_drag_ratio > 5.0
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        with pytest.raises(ValueError):
            self.ai_assistant.analyze_airfoil([], 1e6, 0.1, 5.0)
    
    @pytest.mark.parametrize("reynolds", [1e5, 1e6, 1e7])
    def test_reynolds_sensitivity(self, reynolds):
        """Test analysis across different Reynolds numbers."""
        results = self.ai_assistant.analyze_airfoil(
            self.test_coordinates, reynolds, 0.1, 5.0
        )
        assert results.lift_coefficient > 0
```

## 📚 **Documentation Standards**

### **Docstring Format**
Use Google-style docstrings:

```python
def complex_function(param1: str, param2: int, param3: Optional[float] = None) -> Dict[str, Any]:
    """
    One-line summary of the function.
    
    More detailed description of what the function does, including
    any important algorithms or approaches used.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter  
        param3: Optional parameter with default value
    
    Returns:
        Dictionary containing results with keys:
            - 'result': Main computation result
            - 'metadata': Additional information
    
    Raises:
        ValueError: When param2 is negative
        RuntimeError: When computation fails
    
    Example:
        >>> result = complex_function("test", 42, 3.14)
        >>> print(result['result'])
        
    Note:
        This function requires significant computational resources
        for large inputs.
    """
```

## 🔄 **Submitting Changes**

### **1. Run Pre-submission Checks**
```bash
# Format code
black src/ tests/ examples/
isort src/ tests/ examples/

# Run linting
flake8 src/ tests/ examples/

# Run type checking
mypy src/

# Run tests
pytest tests/ -v

# Check test coverage
pytest tests/ --cov=src --cov-report=html
```

### **2. Commit Changes**
```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add airfoil database search functionality

- Implement intelligent scoring algorithm
- Add fuzzy matching for airfoil names  
- Include performance-based ranking
- Add comprehensive test suite

Closes #123"
```

### **3. Push and Create Pull Request**
```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
# Provide clear description of changes
# Reference any related issues
```

## 📋 **Pull Request Guidelines**

### **PR Title Format**
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding tests
- `refactor:` for code refactoring
- `perf:` for performance improvements

### **PR Description Template**
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] New tests added for new functionality

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or clearly documented)

## Screenshots/Examples
(If applicable, add screenshots or examples)

## Related Issues
Closes #(issue number)
```

## 🐛 **Bug Reports**

### **Bug Report Template**
```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g. Windows 10, macOS 12.0, Ubuntu 20.04]
- Python Version: [e.g. 3.9.7]
- Package Version: [e.g. 1.0.0]
- NeuralFoil Version: [e.g. 0.1.0]

## Additional Context
Any other context about the problem.

## Error Messages/Logs
```
Paste any error messages or logs here
```
```

## 💡 **Feature Requests**

### **Feature Request Template**
```markdown
## Feature Description
Clear description of the feature you'd like to see.

## Use Case
Describe the problem this feature would solve.

## Proposed Solution
Describe how you envision this feature working.

## Alternatives Considered
Describe alternative solutions you've considered.

## Additional Context
Any other context or screenshots about the feature request.

## Implementation Notes
(Optional) Any thoughts on how this might be implemented.
```

## 🏆 **Recognition**

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- GitHub contributors page

## 📞 **Getting Help**

- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Issues**: Use GitHub Issues for bugs and feature requests
- 📧 **Email**: Contact maintainers directly for sensitive issues

## 📜 **Code of Conduct**

Please note that this project adheres to a Code of Conduct. By participating, you are expected to uphold this code:

- Be respectful and inclusive
- Focus on constructive feedback
- Collaborate professionally
- Help create a welcoming environment for all contributors

Thank you for contributing to the AI Aerodynamic Design Assistant! 🛩️✨