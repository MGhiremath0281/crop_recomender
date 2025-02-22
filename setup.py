from setuptools import setup, find_packages

setup(
    name="crop_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "flask",
        "fastapi",
        "uvicorn"
    ],
    author="Muktananda Hiremath",
    description="A machine learning project for crop prediction using NPK, temperature, and pH",
)
