# microscopy_hackathon_team_iphasenet
**Presenting repo of our team iPhaseNet**

Deep Learning for Inverse Problems in Scanning Transmission Electron Microscopy (STEM):

This repository contains a PyTorch implementation for reconstructing quantitative Atomic Electrostatic Potentials (in Volts) from multi-segment STEM images. By integrating metadata (sample thickness and rotation) with visual data, the model solves the inverse physics problem, recovering atomic structure with high fidelity (~54dB PSNR).

**Architecture**

<p align="center">
  <img src="assets/archi.png" width="800">
</p>


## 📊 Results
Our model achieves high-fidelity reconstruction of atomic potentials, recovering physical values (Volts) with significant accuracy.MetricValueInterpretationPSNR54.46 dBExcellent. Indicates the signal is virtually noise-free compared to the ground truth.MAE (Absolute)897.8 VMean Absolute Error. On a typical 26,000V atom, this is an error of only ~3.4%.Pearson Corr.0.72Good. Confirms strong structural correlation between predicted and real atomic columns.Peak Error23.8%Relative error at the brightest atomic centers (intensity peaks).
