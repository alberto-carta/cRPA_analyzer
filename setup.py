from setuptools import setup, find_packages

setup(
    name="crpa_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "triqs_tprf",
    ],
    author="Your Name",
    description="Tools for analyzing cRPA calculations and susceptibilities",
    python_requires=">=3.7",
)
