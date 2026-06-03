[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pozzo-research-group/MC-DFM)

## Description
MC-DFM is a Python package for computing small-angle X-ray and neutron scattering (SAXS/SANS) curves of large hierarchical structures using the **Monte Carlo Distribution Function Method**. It is designed for systems too large for the direct Debye equation — such as protein crystals, protein tubes and sheets, nanoparticle superlattices, and fractal aggregates — while remaining fast enough to run on an ordinary laptop. GPU acceleration is supported via PyTorch when a CUDA device is available.

The package provides three main capabilities:

1. **Scattering simulation** — compute theoretical scattering curves from atomic coordinates (PDB files) or user-defined geometric models (spheres, cylinders, core-shell particles, rods, cubes, etc.)
2. **Hierarchical assembly modeling** — model large periodic or aperiodic structures using a building-block + lattice convolution approach, enabling efficient simulation of crystals, tubes, and other assemblies without explicitly enumerating every atom pair
3. **Curve fitting** — fit simulated curves to experimental data using a genetic algorithm optimizer to extract structural parameters

An additional **LLM interface** (`sas_llm/`) allows users to describe a scattering system in natural language and automatically generate the corresponding simulation script, using the AtomGPT API.

<p align="center">
  <img src="Images/RhuA1.png" width="550" height="350">
</p>


## Method
Instead of computing scattering directly from all atom pairs (as in the Debye equation), the MC-DFM randomly samples pairs of atomic coordinates and accumulates their pairwise distances into a histogram — the pair distribution function (PDF). The scattering intensity is then computed from the PDF via a single summation of sinc functions. All pairwise distance calculations are implemented as matrix operations in PyTorch, making the method both fast and scalable.

<p align="center">
  <img src="Images/method.png" width="700" height="200">
</p>

For hierarchical structures (e.g., a protein crystal), the method separates the building block from the lattice. Random samples are drawn independently from the building block coordinates and the lattice positions, combined via translation and rotation, and then used to compute the pair distribution function. This avoids explicitly constructing the full atomic model of the assembly.

<p align="center">
  <img src="Images/Equations.png" width="700" height="400">
</p>


## Repository Structure

| Directory | Contents |
|---|---|
| `Scattering_Simulator/` | Core MC-DFM implementation (PyTorch), PDB reader, genetic algorithm fitting |
| `genetic_algorithm/` | Standalone genetic algorithm for curve fitting |
| `sas_llm/` | LLM interface for generating simulation scripts from natural language |
| `Notebooks/` | Jupyter notebook examples covering proteins, nanoparticles, HOOMD simulations, and experimental data fitting |
| `sas_llm_results/` | Saved outputs from LLM-generated simulation runs |
| `Data/` | Experimental and simulated scattering data, PDB files, HOOMD trajectories |
| `LLM Examples/` | Notebooks demonstrating the LLM interface |


## Publications 

Further details on the methods used in this work can be found in the following publication:

1. Efficient analysis of small-angle scattering curves for large biomolecular assemblies using Monte Carlo methods

	[![arXiv](https://img.shields.io/badge/arXiv-2025.zktwx-b31b1b.svg)](https://chemrxiv.org/engage/chemrxiv/article-details/679a8c0181d2151a02758fba) [![DOI](https://img.shields.io/badge/DOI-10.1038/s41524.025.01822.z-blue)](https://journals.iucr.org/paper?uu5014)	

The algorithm developed in this work has been used to analyze experimental small-angle scattering data of large protein assemblies in the following publications:

1. Design of light-and chemically responsive protein assemblies through host-guest interactions

	[![arXiv](https://img.shields.io/badge/arXiv-2025.zktwx-b31b1b.svg)](https://www.osti.gov/pages/servlets/purl/2558143) [![DOI](https://img.shields.io/badge/DOI-10.1038/s41524.025.01822.z-blue)](https://www.cell.com/chem/abstract/S2451-9294(24)00652-1)	

2. Bond-centric modular design of protein assemblies

	[![arXiv](https://img.shields.io/badge/arXiv-2025.zktwx-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2024.10.11.617872v1) [![DOI](https://img.shields.io/badge/DOI-10.1038/s41524.025.01822.z-blue)](https://www.nature.com/articles/s41563-025-02297-5)

3. Role of Polymer-Protein Interactions in the Dynamics of Polymer-Integrated Protein Crystals 
  [![DOI](https://img.shields.io/badge/DOI-10.1038/s41524.025.01822.z-blue)](https://pubs.acs.org/doi/full/10.1021/jacs.6c02182)	



## Installation

Clone the repository and enter the directory:

```
git clone https://github.com/pozzo-research-group/MC-DFM.git
cd MC-DFM
```

### Option 1: venv (PowerShell)

Create and activate a virtual environment:

```
python -m venv venv
.\venv\Scripts\activate
```

Install the package:

```
pip install .
```

### Option 2: conda

Create and activate a conda environment:

```
conda create -n mcdfm python=3.12 ipython
conda activate mcdfm
```

Install the package:

```
pip install .
```

## Requirements

Python >= 3.9 is required (Python 3.12 recommended). The core scattering simulator is implemented in PyTorch and can be accelerated with a GPU via the CUDA toolkit. All dependencies are listed in `require.txt` and are installed automatically by `pip install .`:

| Package | Version |
|---|---|
| torch | 2.6.0 |
| numpy | 2.2.2 |
| scipy | 1.15.1 |
| matplotlib | 3.10.0 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| plotly | 6.0.0 |
| h5py | 3.12.1 |
| openpyxl | 3.1.5 |
| ipykernel | 6.29.5 |
| nbformat | 5.10.4 |
