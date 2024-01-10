from setuptools import setup, find_packages

setup(
    name="bfv_py",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
    ],
    author="Enrico Bottazzi, Yuriko Nishijima",
)
