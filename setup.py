from setuptools import setup, find_packages

setup(
    name="bfv",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
    ],
    author="Enrico Bottazzi, Yuriko Nishijima",
)
