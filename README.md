## Description
The Monte Carlo Distribution Function Method is applied to model the small angle scattering curves of large hierarchical structures. This is useful for many biomolecular assemblies such as protein assemblies or protein crystals. This package includes an efficient and user friendly implementation of the MC-DFM method in python. In addition, the calculation is inexpensive, so it sould be able to run on ordinary laptops. We also show examples where the MC-DFM is combined with a genetic algorithm to fit experimental data and obtain structural parameters from it. 

<p align="center">
  <img src="Images/RhuA1.png" width="700" height="200">
</p>

## Method
Instead of calculating the scattering curve directly from the atomic coordinates, the MC-DFM first calculates the pairwise distribution function, which is a histogram of the pairwise distances of the atoms. This is done by randomly sampling the atomic coordinates and calculating the euclidean distance between them. By coding this step entirely with matrix operations, the pairwise distribution function can quickly and efficiently be calculated. 

<p align="center">
  <img src="Images/method.png" width="700" height="200">
</p>

At its simplest form, the MC-DFM is similar to the Debye Scattering Equation as both contain the sinc function. The MC-DFM method substitutes the double summation in the Debye Equation with a single summation over the number of bins in the pairwise histogram. This allows the MC-DFM to be efficiently applied to larger systems. The hypothesis is that as long as the pairwise distribution function approximates the true pairwise distribution function, the scattering curve calculated by the MC-DFM should be equal to the true scattering curve. This will depend on the number of randomly sampled pairwise distances, but we show that this should not be a limitation for ordinary laptops.  

<p align="center">
  <img src="Images/Equations.png" width="700" height="400">
</p>

## Objective 
This method is designed for simulating the small angle scattering curves of large structures that cannot be approximated with a geometric model. It works best when the atomic coordinates of a structure is known beforehand or can be constructed like in the case of protein assembly. Correlation peaks are able to be simulated using this method, however caution must be taken in the high-q region of the simulations, due to the presence of artifacts. 


## Installation 
To install the package, simply git clone and follow the example from the notebooks.

## Requirements 
This package was written in python with the common libraries: <mark> pandas, numpy, matplotlib, scipy, </mark>. For the full list of libraries see the require.txt file.  
