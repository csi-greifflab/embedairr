from setuptools import setup, find_packages

setup(
    name="pepe",
    version="0.1",
    packages=find_packages(),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "transformers",
        "fair-esm",
        "sentencepiece",
        "numpy",
        "protobuf",
        "rjieba",
        "alive_progress",
    ],
    entry_points={"console_scripts": ["pepe = pepe.__main__:main"]},
)
