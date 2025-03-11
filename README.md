# Repository: Bridge-to-experiment-manuscript
Code repo for manuscript "Using machine learning to map simulated noisy and laser-limited multidimensional spectra to molecular electronic couplings"

## Overview
This package implements a **feed-forward neural network (FFNN)** for classifying electronic couplings in molecular dimers based on their **two-dimensional electronic spectra** (**2DES**). 
The package supports:
- **Machine learning model training and evaluation**
- **Sample dataset**
- **Visualization of spectra and ML results**
- **Dataset augmentation with experiment-informed pollutants (noise and resonance conditions)**
    - **Noise addition and signal-to-noise filtering**
    - **Pump pulse resonance conditions (bandwidth and center frequency)**
- **Option to scan pollutant parameters and observe influence on model performance**

## Features (contained in modules)
- **Machine Learning Pipeline**
  - Custom PyTorch dataset class for spectral data
  - Neural network training and evaluation
  - Logging of performance metrics (accuracy, F1 scores)
  - GPU support for accelerated training

- **Data Processing**
  - Loading and structuring 2D spectral data
  - Classification of datasets based on electronic couplings
  - Application of user-defined preprocessing steps

- **Data Augmentation**
  - Additive and intensity-dependent noise injection
  - Filtering of low-SNR spectra

- **Visualization**
  - Contour plots of 2D spectra
  - Line plots for ML metrics (accuracy, loss, F1 scores)

## Installation
Ensure you have the necessary dependencies installed:

pip install torch numpy pandas matplotlib scikit-learn

Clean data are loaded into a 'central dataset' (avoids re-loading data between scan points)