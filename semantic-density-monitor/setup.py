python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="semantic-density-monitor",
    version="0.1.0",
    author="Diego Cerda Seguel",
    author_email="geosemantica@gmail.com",
    description="Geometric phase monitoring for transformer embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geosemantica/semantic-density-monitor",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
)