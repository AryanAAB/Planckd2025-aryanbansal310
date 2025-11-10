# Planck’d 2025 – Quantum Algo Track
**Author:** *[Your Name]*  
**Institution:** IIIT Bangalore  
**Event:** Qimaya @ Planck’d 2025  
**Track:** Quantum Algorithms & Simulation  

---

## Project Overview
This repository contains code, notebooks, and scripts implementing multiple problems from the **Quantum Algorithm Track** of *Planck’d 2025 (Qimaya@IIITB)*.  

It includes implementations and simulations of:
1. **Classical Random Walks (1D)**  
2. **Quantum Walks** — Pauli-X and Hadamard coin variants  
3. **Quantum Walks on Graphs** — using Grover coins and shift operators  
4. **Adiabatic Quantum Computation (AQC)** and spectral gap analysis  
5. **Quantum State Estimation / Tomography**  

The notebook code demonstrates step-by-step transitions from classical stochastic models to quantum coherent dynamics, leading up to adiabatic and QAOA-style formulations.

---

## Repository Structure

```
QuantumAlgoTrack/
│
├── QuantumAlgoTrack.ipynb           # Main notebook containing Problems 0–3
│   ├─ Problem 0 – Classical 1D Random Walk
│   ├─ Problem 1 – Pauli-X Coin Quantum Walk
│   ├─ Problem 2 – Hadamard Coin Quantum Walk
│   └─ Problem 3 – Quantum Walks on Graphs (Grover Coin)
│
├── StateEstimation.py               # Quantum State Tomography via Pauli measurements
├── SampleStateEstimationTest.txt    # 10 example 2-qubit states for reconstruction testing
│
├── aqc_and2.py                      # Adiabatic Quantum Computation for AND(2) Boolean function
│   ├─ Spectrum and gap analysis
│   ├─ Coupling matrix computation
│   └─ Time-evolution & fidelity evaluation
│
└── README.md                        # (this file)
```

---

## Problem Summaries

### **Problem 0 – Classical Random Walk**
- Simulates a 1D symmetric random walk using NumPy.  
- Each walker moves ±1 per step with equal probability.  
- Computes and plots **RMS displacement** √⟨X²⟩ vs number of steps.  
- Validates the expected diffusive scaling √t.  

### **Problem 1 – Quantum Walk with Pauli-X Coin**
- Demonstrates a deterministic “quantum” walk using a coin that flips at every step (Pauli-X).  
- Serves as a contrast: **no interference** and **no ballistic spread**.  

### **Problem 2 – Quantum Walk with Hadamard Coin**
- Implements a **coherent discrete-time quantum walk**.  
- Uses Hadamard coin + shift conditioned on coin state.  
- Shows ballistic spread (σ ∝ t).  
- Plots RMS(t) and final probability distribution.

### **Problem 3 – Quantum Walks on Graphs (Grover Coin)**
- Extends quantum walks to arbitrary graphs.  
- Defines directed edges, Grover coin, and shift operator.  
- Simulates both **quantum** and **classical** propagation.  
- Tested on 6-cycle and 4×5 grid graphs.

---

## Adiabatic Quantum Computation (AQC)
Implemented in **aqc_and2.py**.  
Simulates adiabatic evolution for the AND(2) Boolean function.

**Outputs:**
- Eigenvalue spectrum & spectral gap plots
- Fidelity vs runtime plots
- Success probability vs time plots
- Summary CSV results

---

## State Estimation / Tomography
Implemented in **StateEstimation.py**.  
Simulates 2-qubit tomography using Pauli measurements and reconstructs states via linear inversion and MLE.

---

## Dependencies

```bash
pip install numpy matplotlib qiskit qiskit_aer qiskit_ibm_runtime scipy tqdm
```

---

## Usage Guide

### ▶ Run Notebook
```bash
jupyter notebook QuantumAlgoTrack.ipynb
```

### ▶ Run AQC Simulation
```bash
python aqc_and2.py
```

### ▶ Run State Estimation Tests
```bash
python StateEstimation.py SampleStateEstimationTest.txt
```

---

## Key Results

| Module | Metric | Behavior |
|:--------|:--------|:-------------|
| Problem 0 | RMS displacement | ∝ √t |
| Problem 2 | RMS displacement | ∝ t |
| Problem 3 | Success probability | Quantum > Classical |
| AQC | Fidelity vs runtime | → 1 (adiabatic) |
| State Estimation | Fidelity | > 0.99 |
