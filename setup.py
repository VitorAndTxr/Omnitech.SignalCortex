from setuptools import find_packages, setup

setup(
    name="omnitech-signalcortex",
    version="0.1.0",
    packages=find_packages(exclude=["notebooks", "outputs"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "psycopg2-binary>=2.9.9",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "onnx>=1.15.0",
        "onnxruntime>=1.16.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.14.0",
        "joblib>=1.3.0",
        "python-dateutil",
    ],
)
