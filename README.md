## Description
The Monte Carlo Distribution Function Method is applied to calculate small angle scattering curves of large hierarchical structures. This is useful for many biomolecular assemblies such as protein assemblies or protein crystals. This package includes an efficient and user friendly implementation of the MC-DFM method in python. In addition, the calculation does not require significant computational resources, so it should be able to run on ordinary laptops. We also show examples where the MC-DFM is combined with a genetic algorithm to fit experimental data and obtain structural parameters from it. Examples on calculating the scattering curves of large hierarchical structures are shown in jupyter notebooks in the Notebook folder in this repository. 


<p align="center">
  <img src="Images/RhuA1.png" width="550" height="200">
</p>

## Method
Instead of calculating the scattering curve directly from the atomic coordinates, the MC-DFM first calculates the pairwise distribution function, which is a histogram of the pairwise distances of the atoms. This is done by randomly sampling the atomic coordinates and calculating the euclidean distance between them. By coding this step entirely with matrix operations, the pairwise distribution function can quickly and efficiently be calculated. 

<p align="center">
  <img src="Images/method.png" width="700" height="200">
</p>

At its simplest form, the MC-DFM is similar to the Debye Scattering Equation as both contains a summation of sinc functions. The MC-DFM method substitutes the double summation in the Debye Equation with a single summation over the number of bins in the pairwise histogram. This allows the MC-DFM to be efficiently applied to larger systems. With a large enought sample size, the pairwise distribution function should converge to the true pairwise distribution function, and the scattering curve calculated by the MC-DFM should be equal to the true scattering curve.   

<p align="center">
  <img src="Images/Equations.png" width="700" height="400">
</p>


## Publications 

Further details on the methods used in this work can be found in the following publication:

1. Efficient analysis of small-angle scattering curves for large biomolecular assemblies using Monte Carlo methods

	[![arXiv](https://img.shields.io/badge/arXiv-2025.zktwx-b31b1b.svg)](https://chemrxiv.org/engage/chemrxiv/article-details/679a8c0181d2151a02758fba) [![DOI](https://img.shields.io/badge/DOI-10.1038/s41524.025.01822.z-blue)](https://journals.iucr.org/paper?uu5014)	

The algorithm developed in this work has been used to analyze experimental small-angle scattering data of large protein assemblies in the following publications:

1. Design of light-and chemically responsive protein assemblies through host-guest interactions

	[![arXiv](https://img.shields.io/badge/arXiv-2025.zktwx-b31b1b.svg)](https://www.osti.gov/pages/servlets/purl/2558143) [![DOI](https://img.shields.io/badge/DOI-10.1038/s41524.025.01822.z-blue)](https://www.cell.com/chem/abstract/S2451-9294(24)00652-1)	

2. Bond-centric modular design of protein assemblies

	[![arXiv](https://img.shields.io/badge/arXiv-2025.zktwx-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2024.10.11.617872v1) [![DOI](https://img.shields.io/badge/DOI-10.1038/s41524.025.01822.z-blue)](https://www.nature.com/articles/s41563-025-02297-5)	



## Installation (for powershell)
To install the package, git clone the repository.

To run the examples with jupyter notebooks, enter the Notebook directory.

```
cd .\Notebooks\
```

Create and activate a new environment (powershell for windows):

```
python -m venv venv
```
```
cd .\venv\Scripts
```
```
.\activate 
```

Go back to the MC-DFM repository and install the package with:

```
pip install . 
```

## Installation (using miniconda)
To install the package, git clone the repository.

Then enter the MC-DFM Directory

```
cd .\MC-DFM\
```

Create and activate a new environment (miniconda):

```
conda create -n "mcdfm" python=3.12 ipython 
```

```
conda activate mcdfm
```

Install the package using:

```
pip install . 
```

## Requirements 
This package is written in python. Most of the MC-DFM code is written in Pytorch and can be accelerated with a GPU using the CUDA toolkit. For the full list of libraries used see the require.txt file. The Python version is 3.12 
