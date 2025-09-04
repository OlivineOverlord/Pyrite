from setuptools import setup, find_packages

setup(
    name="pyrite",
    version="0.1.0",
    description="SIMS quantification",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
    ],
)