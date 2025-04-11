# SimuFrame - Simulation Frame

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
![Status](https://img.shields.io/badge/Status-Under%20Development-orange)

## 📌 Overview

This Python-based program performs **geometric nonlinear analysis** of **three-dimensional frame structures**, implementing the **Finite Element Method (FEM)** with **von Kármán kinematics**. It is particularly suitable for **thin-walled structures** and **slender frames**, where second-order effects are significant.

## 🧠 Theoretical Background

The program integrates the **von Kármán strain-displacement relationship** into the element stiffness matrix to capture coupling between axial and bending effects under **moderate rotations** and **small strains**.

- **Geometric nonlinearity** modeled via von Kármán theory  
- **Large displacement** behavior enabled  
- **Small strain assumption** maintained

## ⚙️ Key Features

- 🧮 **Nonlinear solver** using the Newton–Raphson iterative scheme  
- 📊 **3D post-processing** with [PyVista](https://docs.pyvista.org/) and [Matplotlib](https://matplotlib.org/)  
- 🧠 **Eigenvalue buckling analysis** module  
- ✅ **Validation** with commercial software (ANSYS, Abaqus, Robot Structural Analysis)

## ✅ Validation Results

- Displacement deviations below **0.11%** compared to ANSYS, Abaqus, and RSA  
- Buckling load predictions within **5% of Abaqus** in complex cases  
- **Exact match** with RSA results in most scenarios

## 🧩 Applications

This open-source tool is intended for use in:

- Civil engineering simulations  
- Structural systems sensitive to **second-order effects**  
- Educational and research purposes where commercial software may be inaccessible

## 💾 Installation

```bash
git clone https://github.com/SimuFrame/docs.git
cd docs
pip install -r requirements.txt
