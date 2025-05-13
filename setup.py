from setuptools import setup, find_packages

setup(
    name="cognitive_models",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "joblib",
        "tqdm",
        "pandas",
        "matplotlib",
        "cloudpickle",  # Added to ensure parallel execution works out of the box
    ],
    description="A package for cognitive modeling with reinforcement learning and EM fitting",
    author="Your Name",
    author_email="your.email@example.com",
)