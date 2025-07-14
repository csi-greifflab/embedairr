from setuptools import setup, find_packages
import os


# Read the README file for long description
def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        return f.read()


setup(
    name="pypepe",
    version="1.0.0-dev1",
    author="Jahn Zhong",
    author_email="jahn.zhong@medisin.uio.no",
    description="Pipeline for Easy Protein Embedding - Extract embeddings and attention matrices from protein sequences",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/csi-greifflab/pypepe",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: Linux",
        "Operating System :: macOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "fair-esm",
        "sentencepiece",
        "numpy",
        "protobuf",
        "rjieba",
        "alive_progress",
    ],
    entry_points={
        "console_scripts": [
            "pepe = pepe.__main__:main",
            "pypepe = pepe.__main__:main",
        ]
    },
    keywords="protein embeddings bioinformatics machine-learning nlp transformers",
    project_urls={
        "Bug Reports": "https://github.com/csi-greifflab/pypepe/issues",
        "Source": "https://github.com/csi-greifflab/pypepe",
        "Documentation": "https://github.com/csi-greifflab/pypepe#readme",
    },
    include_package_data=True,
)
