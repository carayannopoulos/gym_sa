from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gym-sa",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A simulated annealing package for optimization problems with Gymnasium environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gym-sa",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
        "gymnasium>=0.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    keywords="simulated annealing, optimization, tsp, gymnasium, reinforcement learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/gym-sa/issues",
        "Source": "https://github.com/yourusername/gym-sa",
        "Documentation": "https://github.com/yourusername/gym-sa#readme",
    },
)
