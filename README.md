# Physics-Based Machine Learning CFD Solver for Hypersonic Boundary Layers

This repository contains the CFD code and supporting scripts developed for the paper  
**“Physics-Based Machine Learning Closures and Wall Models for Hypersonic Transition–Continuum Boundary Layer Predictions”**  
by *Ashish S. Nair, Narendra Singh, Marco Panesi, Justin Sirignano, and Jonathan F. MacArt*  
([arXiv:2507.08986](https://arxiv.org/pdf/2507.08986)).

---

## 🧩 Overview

This code implements a **high-order compressible Navier–Stokes solver** for two-dimensional hypersonic boundary-layer flows in argon gas.  
It is designed for continuum and transition–continuum regimes (Kn ≈ 0.1–10) and supports machine-learning-based transport and wall models for nonequilibrium effects.

### Key Features

- Fourth-order central-difference discretization for **viscous fluxes**
- Modified **Steger–Warming flux-splitting scheme** for inviscid fluxes
- **Pseudo-time explicit Euler** advancement to steady state
- Convergence monitored via residual reduction criteria
- **Parallel execution** using MPI (supports both domain decomposition and batch-wise training)
- Iterative linear solvers for **steady adjoint equations**
- Coupled training of **ML-based anisotropic transport** and **wall distribution models**

---

## ⚙️ Numerical Methods

The Navier–Stokes equations are solved in conservative form using:
- **Viscous fluxes:** fourth-order central-difference scheme  
- **Inviscid fluxes:** modified Steger–Warming flux-splitting  
- **Time integration:** explicit Euler pseudo-time marching to steady state  
Residual convergence serves as the stopping criterion.

The solver supports both **direct flow computation** and **adjoint-based gradient evaluation** for model training.

---

## 🧠 Machine Learning Augmentations

The code allows embedding **neural network closures** within the PDE solver for:
- Anisotropic, trace-free viscosity and conductivity models
- Data-driven wall models based on **skewed-Gaussian velocity distribution functions**
- Adjoint-based optimization for parameter training using DSMC data

Parallel multi-Mach and multi-Knudsen training is supported via MPI gradient aggregation.

---
