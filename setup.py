from setuptools import setup, find_packages

setup(
    name="privscale-fl",
    version="0.1.0",
    author="Anonymous Authors",
    description="Privacy Risk Scales with Effective Dimension in Federated Learning",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2",
        "torchvision>=0.17",
        "numpy>=1.24",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "transformers>=4.40",
        "peft>=0.10",
        "datasets>=2.18",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "pandas>=2.1",
        "matplotlib>=3.8",
    ],
)
