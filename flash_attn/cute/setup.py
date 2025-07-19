#!/usr/bin/env python
"""
Setup script for flash_attn.cute package - CUTE: CUtlass Templates for flash attention
Pure Python package with no CUDA compilation required.
"""
from setuptools import setup, find_packages
import os

# Get the parent directory to find README
current_dir = os.path.dirname(os.path.abspath(__file__))
readme_path = os.path.join(current_dir, "..", "..", "README.md")

try:
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "CUTE: CUtlass Templates for flash attention"

setup(
    name="flash-attn-cute",
    version="0.1.0",
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="CUTE: CUtlass Templates for flash attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/flash-attention",
    packages=["flash_attn.cute"],
    package_dir={"flash_attn.cute": "."},
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "nvidia-cutlass-dsl==4.1.0.dev0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
)