from setuptools import setup, find_packages

setup(
    name="embedairr",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "fair-esm",
        "sentencepiece",
        "numpy",
    ],
    entry_points={"console_scripts": ["embedairr = embedairr.__main__:main"]},
)
