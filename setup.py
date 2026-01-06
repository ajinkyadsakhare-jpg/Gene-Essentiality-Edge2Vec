from setuptools import setup, find_packages

setup(
    name="help_edge2vec",
    version="0.1.0",
    description="HELP pipeline with Edge2Vec embeddings",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "optuna",
        "networkx",
        "gensim",
        "pyarrow",
        "pyyaml",
        "tqdm",
        "joblib",
    ],
)
