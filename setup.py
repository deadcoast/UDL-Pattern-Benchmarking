"""Setup script for UDL Rating Framework."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="udl-rating-framework",
    version="0.1.0",
    author="UDL Rating Framework Team",
    description="A mathematically-grounded system for evaluating User Defined Languages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/udl-rating-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Compilers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "networkx>=3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "hypothesis>=6.0.0",
        "lark-parser>=1.1.0",
        "statsmodels>=0.14.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "sphinx>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "udl-rate=udl_rating_framework.cli.rate:main",
            "udl-train=udl_rating_framework.cli.train:main",
            "udl-compare=udl_rating_framework.cli.compare:main",
            "udl-evaluate=udl_rating_framework.cli.evaluate:main",
        ],
    },
)
