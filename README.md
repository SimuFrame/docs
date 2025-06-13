# SimuFrame - Simulation Frame

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
![Status](https://img.shields.io/badge/Status-Under%20Development-orange)

## Overview

SimuFrame is a Python-based computational tool for **geometric nonlinear analysis** of **three-dimensional frame structures**. Implementing the **Finite Element Method (FEM)** with **von Kármán kinematics**, it provides robust solutions for **thin-walled structures** and **slender frames**, where second-order effects significantly influence structural behavior.

## Theoretical Foundation

The framework incorporates the **von Kármán strain-displacement relationship** into the element stiffness matrix, accurately capturing axial-bending coupling under **moderate rotations** while maintaining **small strain assumptions**.

Key theoretical aspects:
- **Geometric nonlinearity** via von Kármán theory
- **Large displacement** analysis capability
- **Small strain** formulation

## Core Features

- **Nonlinear solver** using the Newton–Raphson iterative method
- **3D post-processing** with PyVista and Matplotlib integration
- **Eigenvalue buckling analysis** module
- **Validation** against industry-standard software (ANSYS, Abaqus, Robot Structural Analysis)

## Validation Benchmarks

- Maximum displacement deviation: **< 0.11%** vs. commercial solutions
- Buckling load accuracy: **within 5%** of Abaqus in complex cases
- **Full agreement** with Robot Structural Analysis in standard scenarios

## Applications

Designed for both academic and professional use, SimuFrame serves:
- Civil and structural engineering simulations
- Analysis of systems sensitive to **second-order effects**
- Research and education in computational structural mechanics

## Installation

```bash
git clone https://github.com/SimuFrame/docs.git
cd docs
pip install -r requirements.txt
