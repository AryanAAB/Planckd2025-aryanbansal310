# Planck’d 2025 – Quantum Algo Track
**Author:** *Aryan Bansal*, *Akshat Batra*, *Abhirath Adamane*  
**Institution:** IIIT Bangalore  
**Event:** Qimaya @ Planck’d 2025  
**Track:** Quantum Algorithms & Simulation  

---

## Project Overview
This repository contains code, notebooks, and scripts implementing multiple problems from the **Quantum Algorithm Track** of *Planck’d 2025 (Qimaya@IIITB)*.  

It includes implementations and simulations of:
1. **Classical Random Walks (1D)**  
2. **Quantum Coin Flip**
3. **Superposed Walker** — using Grover coins and shift operators  
4. **Graph-Based Computation**
5. ** State Estimation**
6. **Quantum Oscillator Search**
7. **Slow and Steady Wins the Quantum Race**
8. **Bridging QAOA and Adiabatic Paths**
9. **Decoherence, Measurements, and Quantum Search Breakdown**

The notebook code demonstrates step-by-step transitions from classical stochastic models to quantum coherent dynamics, leading up to adiabatic and QAOA-style formulations.

---

## Repository Structure

```
code/
│
├── QuantumAlgoTrack.ipynb           # Main notebook containing Problems 0–3
│   ├─ Problem 0 – Classical 1D Random Walk
│   ├─ Problem 1 – Pauli-X Coin Quantum Walk
│   ├─ Problem 2 – Hadamard Coin Quantum Walk
│   ├─ Problem 3 – Quantum Walks on Graphs (Grover Coin)
|   ├─ Problem 4 – Quantum Oscillator Search
│
├── StateEstimation.py               # Quantum State Tomography via Pauli measurements
│
├── aqc_and2.py                      # Adiabatic Quantum Computation for AND(2) Boolean function
│   ├─ Spectrum and gap analysis
│   ├─ Coupling matrix computation
│   └─ Time-evolution & fidelity evaluation
|── Problem8.py
│
├── qaoa_vs_adiabatic.py
```

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

### ▶ Run qaoa_vs_adiabatic.py
```bash
python qaoa_vs_adiabatic.py
```

### ▶ Run Problem8.py
```bash
python Problem8.py
```

### ▶ Run State Estimation Tests
```bash
python StateEstimation.py SampleStateEstimationTest.txt
```

---
