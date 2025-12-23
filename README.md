# Numerical experiments for ED-degree and metric dependence

This repository contains the Julia code used to perform the numerical experiments
described in Section 5 of the paper

"On the Euclidean Distance Degree of Quadratic Two-Neuron Neural Networks"

by Giacomo Graziani.

The computations illustrate the dependence of the Euclidean Distance degree on the
choice of the scalar product, by comparing a generic metric with the Bombieri–Weyl metric
for the variety of symmetric 3×3 matrices of rank at most 2.

## Contents

- `ed_degree_equazioni3.jl`  
  Computes the ED-degree for a generic symmetric invertible metric on `Sym^2(C^3)`.

- `ed_degree_equazioni3BW.jl`  
  Computes the ED-degree for the Bombieri–Weyl metric.

- `Results.txt`  
  Output of representative runs showing stability of the solution counts.

## Requirements

- Julia ≥ 1.8
- Packages:
  - HomotopyContinuation.jl
  - LinearAlgebra
  - Random

## Usage

Run the scripts in Julia, for example:

```julia
julia ed_degree_equazioni3.jl
julia ed_degree_equazioni3BW.jl
