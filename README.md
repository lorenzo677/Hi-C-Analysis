# Hi-C-Analysis

## Introduction
This is a Python package containing some functions to process, perform spectral analysis and visualize the Hi-C matrices. 
 
In particular, with this package, it is possible extract from the full Hi-C matrix the matrices corresponding to the 24 chromosomes of the human genomes, to compute some basic distribution about the network, such as the betweenness and degree centrality, and to reconstruct the original matrix by using the projectors of the highest ranked eigenvectors.

This package is composed by two modules:

* preprocessing
* visualizegraph

with the first module it is possible to manage the data and perform spectral analysis, while with the second it is possible to visualize the results with a pre-defined style.

The repository has the following structure:
* The directory [docs](https://github.com/lorenzo677/Hi-C-Analysis/tree/main/docs) contains the documentation of the package.
* The directory [hicanalysis](https://github.com/lorenzo677/Hi-C-Analysis/tree/main/hicanalysis) contains the two modules of the package.
* The directory [images](https://github.com/lorenzo677/Hi-C-Analysis/tree/main/images)  contains some demonstrative images created with the package.
* The directory [script](https://github.com/lorenzo677/Hi-C-Analysis/tree/main/script)  contains the the script used for Complex Network project @unibo.
* The directory [tests](https://github.com/lorenzo677/Hi-C-Analysis/tree/main/tests)  contains the tests for the functions of the package.
* The directory [tutorial](https://github.com/lorenzo677/Hi-C-Analysis/tree/main/tutorial)  contains a tutorial to shows the basic usage of the package.

All the others files and directories have been used to generate the documentation.

## Installation
The easiest way to install the package is to clone the GitHub repository and install it via pip:
```
git clone https://github.com/lorenzo677/Hi-C-Analysis
cd Hi-C-Analysis
python3 -m pip install -r requirements.txt .
```
This command install also the requirements, that can be also installed before the installation of the package. The requirements are here reported:
```
matplotlib==3.6.3
networkx==3.0
numpy==1.24.2
pandas==1.5.3
seaborn==0.12.2
```
The module was build with these version of the packages but probably Hi-C-Analysis works also with previous versions (not tested).
## Running tests
The tests for the module `preprocessing`, are contained in the `tests` directory. To run them it is necessary to be in the `Hi-C-Analysis` directory and to have installed the `pytest` package.
Then it is enough to run the pytest command:
```
pytest
```
This package was built with Python `3.11.2` on `macOS 13.2.1` with `arm64` architecture and tested both on `macOS` and `Windows 10` machines.


## Documentation and Tutorial
The documentation of the modules can be found in the directory `docs` and online at the website [https://github.com/lorenzo677/Hi-C-Analysis/blob/main/_build/html/index.html](https://github.com/lorenzo677/Hi-C-Analysis/blob/main/_build/html/index.html)
In the `docs` directory there is `hi-c-analysis.pdf` that contain the documentation for all the function of the two modules.

In the directory `tutorials` it is also possible to find a tutorial that shows the basic functionality of the package.  In the directory it is present the same file in three different formats: `.html`, `.ipynb` and `.pdf`.
