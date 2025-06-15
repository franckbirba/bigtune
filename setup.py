#!/usr/bin/env python3
"""
Setup script for BigTune - LoRA Fine-tuning Pipeline
"""

from setuptools import setup, find_packages

setup(
    name="bigtune",
    version="1.0.0",
    description="LoRA Fine-tuning Pipeline CLI for RunPod training and LM Studio deployment",
    author="BigTune",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'bigtune=bigtune.cli:main',
        ],
    },
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'peft>=0.4.0',
        'requests>=2.25.0',
        'python-dotenv>=0.19.0',
        'pathlib',
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)