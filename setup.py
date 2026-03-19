"""
Setup script for Gesture Platform

Install with:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
long_description = ""
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    try:
        long_description = readme_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        long_description = readme_path.read_text(encoding="utf-16")

# Read requirements
requirements = []
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="gesture-platform",
    version="1.0.0",
    author="Gesture Platform Team",
    author_email="contact@gesture-platform.com",
    description="Real-time sign language translation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gesture-platform",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/gesture-platform/issues",
        "Documentation": "https://gesture-platform.readthedocs.io",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "apps"]),
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=6.0.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
    },
    entry_points={},
    include_package_data=True,
    zip_safe=False,
)
