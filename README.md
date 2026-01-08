# Mamba Model Predictive Control

This repository contains supplementary code for the paper:  
**Mamba Neural Networks meets Model Predictive Control**.

We demonstrate the application of Mamba networks in the context of Model Predictive Control (MPC)  across two simulated nonlinear dynamical system and one physical setup.

---

## 📁 Repository Structure

The repository contains implementations for both physical and simulated systems:

- **`Aero2/`** — Physical 2-DOF helicopter system implementation
- **`Numerical_Examples/`** — Simulated nonlinear dynamical systems
  - `Four_Tank/` — Four-tank system simulation
  - `Van_der_Pol/` — Van der Pol oscillator simulation

### Aero2 Folder Contents:

- `MambaMPC_Aero2.ipynb` — Main notebook for training and MPC simulation on the Aero2 system
- `MambaMPC/` — Core Mamba-MPC implementation
  - `mamba_model.py` — Mamba neural network architecture
  - `MambaCasadi.py` — CasADi-compatible Mamba model for MPC
  - `functions.py` — Utility functions
  - `DataProcessing.py` — Data preprocessing and handling
  - `plotting.py` — Visualization utilities
  - `pscan.py` — Parallel scan implementation
- `Data/` — Training and test datasets (`.npy` files)
- `State_dicts/` — Saved model checkpoints
- `Figures/` — Generated plots and visualizations
- `hal/`, `pal/`, `pit/`, `qvl/` — Quanser hardware interface libraries

### Numerical_Examples Folder Contents:

Each subfolder (`Four_Tank/`, `Van_der_Pol/`) includes:

- `MambaMPC_[SystemName].ipynb` — Simulation and training notebook for the respective system
- `MambaMPC/` — Mamba-MPC implementation (same structure as Aero2)
  - `mamba_model.py` — Mamba neural network architecture
  - `MambaCasadi.py` — CasADi-compatible Mamba model
  - `functions.py` — System-specific utility functions
  - `DataProcessing.py` — Data preprocessing utilities
  - `plotting.py` — Visualization tools
  - `pscan.py` — Parallel scan implementation
- `Data/` — Generated training/validation datasets
- `data_Mamba/` or `Data_Mamba/` — Mamba-specific results and data
- `data_LSTM/` — LSTM comparison results (where applicable)
- `State_dicts/` — Saved model weights
- `Figures/` — Generated plots and results
- `LSTM/` — LSTM implementation for comparison (Van der Pol only)

---

## 📦 Requirements

Python 3.11.7 is recommended for running the examples.

## 📓 Development Environment
- `jupyter` — For running and developing in Jupyter Notebooks

## 🧰 Dependencies

This project requires the following Python libraries:

### Core Libraries
- `numpy` — Numerical computing and array operations
- `scipy` — Scientific computing (integration, linear algebra)
- `matplotlib` — Plotting and visualization
- `torch` (PyTorch) — Deep learning framework for Mamba neural networks
- `casadi` — Symbolic framework for nonlinear optimization and MPC

### Mamba-Specific Dependencies 
- `einops` — Tensor operations for rearrange, repeat, and einsum

### Hardware Interface (Aero2 only)
- `pal` (Quanser Python API for Linux) — Interface for Quanser Aero2 hardware

### Standard Library
- `time` — Timing utilities
- `itertools` — Iterator building blocks
- `os` — Operating system interface
- `math` — Mathematical functions
- `dataclasses` — Data class decorators

---

## 🚀 How to Run

### For Numerical Examples:

1. Clone or download the repository

2. Navigate to one of the numerical example folders:

```bash
cd "Numerical_ Examples/Van_der_Pol"
```

or

```bash
cd "Numerical_ Examples/Four_Tank"
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook
```

4. Open the main notebook:
   - `MambaMPC_VDP.ipynb` for Van der Pol oscillator
   - `MambaMPC_FourTank.ipynb` for Four-Tank system

5. Run all code cells from top to bottom

### For Physical Aero2 Setup:

1. Ensure the Quanser Aero2 hardware is connected and the `pal` library is installed

2. Navigate to the Aero2 folder:

```bash
cd Aero2
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook
```

4. Open `MambaMPC_Aero2.ipynb` and run all code cells from top to bottom

---

## 📊 What Each Notebook Does

Each notebook performs the following steps:

- **Data Generation**: Generate or load training and validation data from the dynamical system
- **Model Training**: Train the Mamba neural network architecture with the generated data
- **MPC Simulation**: Run closed-loop Model Predictive Control using the trained Mamba model
- **Results Storage**: Save trained models to `State_dicts/` and data to `Data/`, `data_Mamba/`, or `data_LSTM/` folders
- **Visualization**: Generate plots comparing Mamba-MPC performance with baseline methods (LSTM-MPC where applicable)

**Numerical Examples** demonstrate:
- Reference tracking capabilities
- Comparison with LSTM-based MPC
- Robustness to noise (Van der Pol)
- Multi-input multi-output control (Four-Tank)

**Aero2 Example** demonstrates:
- Real-time control on physical hardware
- Data collection from the 2-DOF helicopter system
- Closed-loop tracking performance on a real system

---
<!-- 
## 📄 Citation

If you use this code in your work, please cite the following paper:

```bibtex
@misc{dejong2025deepoperatorneuralnetwork,
      title={Deep Operator Neural Network Model Predictive Control}, 
      author={Thomas Oliver de Jong and Khemraj Shukla and Mircea Lazar},
      year={2025},
      eprint={2505.18008},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2505.18008}, 
}
``` -->

---

## 📬 Contact

We welcome questions, feedback, and collaboration opportunities!

- 📧 **Primary contact**: [m.cevaal@tilburguniversity.edu](mailto:m.cevaal@tilburguniversity.edu)
- 📧 **Co-authors**:
  - [t.o.d.jong@tue.nl](mailto:t.o.d.jong@tue.nl)
  - [m.lazar@tue.nl](mailto:m.lazar@tue.nl)

Feel free to reach out via email for more information about this project or related research.





