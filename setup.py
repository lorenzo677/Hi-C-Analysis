from setuptools import setup, find_packages

setup(
    name = 'hicanalysis',
    version = '0.1.0',
    packages = find_packages(include=['hicanalysis']),
    description='A little package for spectral analysis for Hi-C matrices and visualize the results.',
    author='Lorenzo Barsotti',
    install_requires = ["networkx", "pandas", "matplotlib", "numpy", "scipy", "seaborn", "pathlib",],
    )