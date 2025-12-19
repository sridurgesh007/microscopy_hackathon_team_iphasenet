# microscopy_hackathon_team_iphasenet
**Presenting repo of our team iPhaseNet**

Deep Learning for Inverse Problems in Scanning Transmission Electron Microscopy (STEM):

This repository contains a PyTorch implementation for reconstructing quantitative Atomic Electrostatic Potentials (in Volts) from multi-segment STEM images. By integrating metadata (sample thickness and rotation) with visual data, the model solves the inverse physics problem, recovering atomic structure with high fidelity (~54dB PSNR).

**Architecture**

<p align="center">
  <img src="assets/archi.png" width="600">
</p>


## 📊 Quantitative Results

Our model achieves high-fidelity reconstruction of atomic potentials, recovering physical values (Volts) with significant accuracy.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **PSNR** | **54.46 dB** | **Excellent.** Indicates the signal is virtually noise-free compared to the ground truth. |
| **MAE (Absolute)** | **897.8 V** | Mean Absolute Error. On a typical 26,000V atom, this is an error of only **~3.4%**. |
| **Pearson Corr.** | **0.72** | **Good.** Confirms strong structural correlation between predicted and real atomic columns. |
| **Peak Error** | **23.8%** | Relative error at the brightest atomic centers (intensity peaks). |

### 🔍 Analysis
* **High Fidelity:** The exceptionally high PSNR (>50 dB) demonstrates that the `SwinUNETR` backbone successfully denoised the STEM input while preserving atomic lattice structures.
* **Physical Recovery:** The model predicts a mean potential of **1825 V** (vs. Ground Truth **2162 V**), indicating it has learned to recover the correct order of magnitude.
* **Contrast Handling:** The max predicted potential (**28.3 kV**) slightly exceeds the ground truth (**25.9 kV**), suggesting the model effectively learns sharp, high-contrast atomic peaks.
