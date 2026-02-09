"""
Setup script for the Notebook ML Orchestrator.

This script configures the package for installation and distribution.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements-orchestrator.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#') and '# Built into Python' not in line
        ]

setup(
    name="notebook-ml-orchestrator",
    version="0.1.0",
    author="Notebook ML Orchestrator Team",
    author_email="team@notebook-ml-orchestrator.com",
    description="A comprehensive ML orchestration platform leveraging free notebook platforms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/notebook-ml-orchestrator/notebook-ml-orchestrator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "gui": [
            "gradio>=3.40.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
        "ml": [
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "pillow>=10.0.0",
            "librosa>=0.10.0",
            "transformers>=4.30.0",
            "torch>=2.0.0",
        ],
        "backends": [
            "modal>=0.55.0",
            "huggingface-hub>=0.16.0",
            "requests>=2.31.0",
            "aiohttp>=3.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "notebook-orchestrator=notebook_ml_orchestrator.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "notebook_ml_orchestrator": [
            "templates/*.json",
            "schemas/*.json",
            "static/*",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/notebook-ml-orchestrator/notebook-ml-orchestrator/issues",
        "Source": "https://github.com/notebook-ml-orchestrator/notebook-ml-orchestrator",
        "Documentation": "https://notebook-ml-orchestrator.readthedocs.io/",
    },
)