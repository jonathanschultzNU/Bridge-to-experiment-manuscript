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


input parameters:


example input files:

jobname = intnoise
task = noise
noise method = intensity-dependent
noise fraction = 0, 0.01
SNR filter = True
SNR threshold = 0.01
class bounds = -805, -755, -705, -655, -605, -555, -505, -455, -405, -355, -305, -255, -205, -155, -105, -55, -5, 5, 55, 105, 155, 205, 255, 305, 355, 405, 455, 505, 555, 605, 665, 705, 755, 805 
input size = 22801
hidden layer size = 100
number of epochs = 5
batch size = 100
learning rate = 0.01
dropout probability = 0.2
train-test split = 0.8
spec save interval = 1


jobname = bandwidth_center_frequency
task = bandwidth and center frequency
bandwidth = 10000, 2000, 1000
center frequency = 12000, 14500, 16500
class bounds = -805, -755, -705, -655, -605, -555, -505, -455, -405, -355, -305, -255, -205, -155, -105, -55, -5, 5, 55, 105, 155, 205, 255, 305, 355, 405, 455, 505, 555, 605, 665, 705, 755, 805 
input size = 22801
hidden layer size = 100
number of epochs = 5
batch size = 100
learning rate = 0.01
dropout probability = 0.2
train-test split = 0.8
spec save interval = 1