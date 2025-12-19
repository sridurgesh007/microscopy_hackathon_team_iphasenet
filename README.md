# microscopy_hackathon_team_iphasenet
**Presenting repo of our team iPhaseNet**

**Deep Learning for Inverse Problems in Scanning Transmission Electron Microscopy (STEM):**

This repository contains a PyTorch implementation for reconstructing quantitative Atomic Electrostatic Potentials (in Volts) from multi-segment STEM images. By integrating metadata (sample thickness and rotation) with visual data, the model solves the inverse physics problem, recovering atomic structure with high fidelity (~54dB PSNR).

## Architecture

<p align="center">
  <img src="assets/archi.png" width="600">
</p>


📚 Dataset & Quality
The dataset consists of high-fidelity Scanning Transmission Electron Microscopy (STEM) simulations generated using the Multislice algorithm. It is designed to rigorously test the model's ability to solve the inverse physics problem under diverse and challenging conditions.

✨ Dataset SpecificationsSource: 
Simulated physics data (Multislice method) ensuring Ground Truth accuracy.
Materials: Varied crystal structures including GaN (Gallium Nitride), LiCoO2, and Perovskite (PVSK) , MoS2, KnBO3.
Resolution: 256 $\times$ 256 pixels per sample.
Physics Variations:Thickness: diverese sample thickness [20,40,60,80,100,120,150].
Rotation: Random crystal tilts (0° - 360°) to introduce challenging projection overlaps.
Noise: Realistic Poisson/Gaussian noise injected to simulate detector imperfections while training.


### 📥 Dataset Download
The full dataset (Train/Val splits, `.npz` format) is hosted on Google Drive.

[![Download Dataset](https://img.shields.io/badge/Dataset-Download%20from%20Drive-4285F4?style=for-the-badge&logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1t1bectJp9r56jiHUq697msGyd6i7WmDj?usp=drive_link)

## Dataset sample
<p align="center">
  <img src="assets/sample_data.jpeg" width="600">
</p>

### Metadata
The model utilizes an **8-Channel Input Tensor** `(B, 8, 256, 256)` that fuses visual data with physical metadata. This "Early Fusion" strategy allows the network to condition its predictions on the sample's physical properties.

| Channel | Component | Type | Description | Normalization |
| :--- | :--- | :--- | :--- | :--- |
| **0** | **STEM Segment 1** | Visual | Detector quadrant 1 (Top-Left view). | Min-Max (0.0 - 1.0) |
| **1** | **STEM Segment 2** | Visual | Detector quadrant 2 (Top-Right view). | Min-Max (0.0 - 1.0) |
| **2** | **STEM Segment 3** | Visual | Detector quadrant 3 (Bottom-Left view). | Min-Max (0.0 - 1.0) |
| **3** | **STEM Segment 4** | Visual | Detector quadrant 4 (Bottom-Right view). | Min-Max (0.0 - 1.0) |
| **4** | **Thickness** | Metadata | Sample thickness in nanometers. | $x / 100.0$ (e.g., 50nm $\rightarrow$ 0.5) |
| **5** | **Rotation $\alpha$** | Metadata | Crystal tilt/rotation (Euler angle X). | $x / 360.0$ (Degrees $\rightarrow$ 0-1) |
| **6** | **Rotation $\beta$** | Metadata | Crystal tilt/rotation (Euler angle Y). | $x / 360.0$ (Degrees $\rightarrow$ 0-1) |
| **7** | **Rotation $\gamma$** | Metadata | Crystal tilt/rotation (Euler angle Z). | $x / 360.0$ (Degrees $\rightarrow$ 0-1) |


## 🖼️ Visual Reconstruction Results

The figure below demonstrates the model's ability to solve the inverse physics problem on a validation sample .

<p align="center">
  <img src="assets/results.jpeg" width="600" alt="Model Prediction vs Ground Truth">
  <br>
  <em>Figure: (Left) Raw Noisy STEM Input showing scanning artifacts. (Middle) Model Prediction recovering atomic columns. (Right) Ground Truth Electrostatic Potential.</em>
</p>

## 🤖 Pre-trained Models

We provide the official pre-trained weights for the **SwinUNETR** model, trained on the full dataset (6,000 samples) for 100 epochs.

| Model Version | Best PSNR | Size | Download |
| :--- | :--- | :--- | :--- |
| **STEM-Recon-v1** | **54.45 dB** |~95 MB | [![Download](https://img.shields.io/badge/Download-Weights-blue?style=flat-square&logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1K5q5mC21iD3uZS6_1AiYm2ZORb84-5-q?usp=drive_link) |

### ⚡ Quick Start
1.  **Download** the `.pt` file from the link above.
2.  **Place** the file in the `checkpoints_ddp/` folder.
3.  **Run Inference:**
    ```bash
    python inference.py --ckpt checkpoints_ddp/best_model.pt
    ```

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
