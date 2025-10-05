from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-aerodynamic-design-assistant",
    version="1.0.0",
    author="MDO Laboratory",
    author_email="your-email@domain.com",
    description="AI-powered conversational aerodynamic design system for aircraft wings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/AI-Aerodynamic-Design-Assistant",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/AI-Aerodynamic-Design-Assistant/issues",
        "Documentation": "https://github.com/your-username/AI-Aerodynamic-Design-Assistant/docs",
        "Source Code": "https://github.com/your-username/AI-Aerodynamic-Design-Assistant",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "isort>=5.9.0",
            "mypy>=0.910",
            "flake8>=3.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-aerodynamic-designer=final_aerodynamic_ai:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "aerodynamics",
        "aircraft design",
        "artificial intelligence",
        "neural networks",
        "optimization",
        "CFD",
        "airfoil",
        "wing design",
        "aerospace engineering",
    ],
)