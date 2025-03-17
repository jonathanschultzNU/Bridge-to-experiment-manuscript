# Repository: Bridge-to-experiment-manuscript
Code repo for manuscript "Using machine learning to map simulated noisy and laser-limited multidimensional spectra to molecular electronic couplings"

## **Overview**
This package implements a **feed-forward neural network (FFNN)** for classifying electronic couplings in molecular dimers based on their **two-dimensional electronic spectra** (**2DES**).  
The package supports:
- **Machine learning model training and evaluation**
- **Processing and augmentation of spectral datasets**
- **Visualization of spectra and ML results**
- **Dataset augmentation with experiment-informed modifications:**
  - **Noise addition and signal-to-noise filtering**
  - **Pump pulse resonance conditions (bandwidth and center frequency)**
- **Parameter scanning to observe how experimental conditions affect model performance**

---

## **Features**
### **Machine Learning Pipeline**
- Custom PyTorch dataset class for spectral data
- Neural network training and evaluation
- Logging of performance metrics (accuracy, F1 scores)
- GPU support for accelerated training

### **Data Processing**
- Loading and structuring 2D spectral data
- Classification of datasets based on electronic couplings
- User-defined preprocessing steps

### **Data Augmentation**
- Additive and intensity-dependent noise injection
- Filtering of low-SNR spectra
- Pump modulation effects (bandwidth, center frequency)

### **Visualization**
- Contour plots of 2D spectra
- Line plots for ML metrics (accuracy, loss, F1 scores)

---

## **Installation**
### **Using Conda (Recommended)**
First, create and activate a **conda environment** for the project:
```bash
conda env create -f environment.yml
conda activate bridge-to-exp-env
```
This ensures that all required dependencies are installed correctly.

### **Manual Installation with Pip**
Alternatively, you can manually install dependencies:

pip install torch numpy pandas matplotlib scikit-learn gitpython

---

## **Usage**

### **Setting Up the Input File**
The model requires an input file specifying parameters for data processing and training.
Each parameter is written as a key = value pair.

### **Input Parameters**
#### **Required Parameters**
| Parameter            | Description |
|----------------------|-------------|
| `jobname`           | Name of the experiment (used for saving results). |
| `task`              | Type of dataset modification (`noise`, `bandwidth`, `center frequency`, `bandwidth and center frequency`). |
| `class bounds`      | List of numerical boundaries defining classification bins for electronic coupling values. |
| `input size`        | Number of features in the dataset (e.g., flattened spectrum size). |
| `hidden layer size` | Number of neurons in the hidden layer of the FFNN. |
| `number of epochs`  | Number of training iterations. |
| `batch size`        | Number of samples per batch in training/testing. |
| `learning rate`     | Learning rate for the optimizer. |
| `dropout probability` | Dropout rate to prevent overfitting. |
| `train-test split`  | Proportion of data allocated for training (remaining is for testing). |
| `spec save interval` | Frequency of saving intermediate 2D spectra (every N iterations). |

#### **Task-Specific Parameters**
| Task Type  | Additional Required Parameters |
|------------|--------------------------------|
| `noise`    | `noise method`, `noise fraction`, `SNR filter`, `SNR threshold` |
| `bandwidth` | `bandwidth` (list of values to scan) |
| `center frequency` | `center frequency` (list of values to scan) |
| `bandwidth and center frequency` | `bandwidth`, `center frequency` |

#### **Optional Parameters**
| Parameter            | Default Value | Description |
|----------------------|--------------|-------------|
| `save iteration outputs` | `"none"` | Whether to save ML reports per iteration. Options: `"none"`, `"partial"`, `"full"`. |
| `save training reports`  | `"false"` | Whether to save plots of training progress (accuracy, loss, F1 scores). |
| `save 2D plots`      | `"false"` | Whether to save 2D spectra during training. |
| `check-system ID`    | `11` | Specific system ID to check during debugging. |
| `t2 truncate`        | `"false"` | Whether to truncate `t2` time points. |
| `torch seed`         | `2942` | Random seed for PyTorch (ensures reproducibility). |
| `numpy seed`         | `72067` | Random seed for NumPy operations. |
| `split seed`         | `72067` | Random seed for train-test splitting. |


### **Example Input Files**

#### **Example 1: Noise Addition**
jobname = intensity_noise
task = noise
noise method = intensity-dependent
noise fraction = 0, 0.01, 0.1, 1
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

#### **Example 2: Bandwidth and Center-Frequency Scanning**
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


### **Running the Code**
#### **1. Training the Model**
Once the input file is set up, run the main script:
```bash
python main.py
```
This will:
-Load the dataset
-Apply the specified augmentation (if any)
-Train the feed-forward neural network
-Iterate through the augmentation values (if any)
-Log accuracy and F1-score results

#### **2. Viewing Results**
After running the code, the following files and directories will be generated:

##### **Required Outputs**
All outputs → [job name]_outputs/
    -This directory will be automatically generated to contain all following outputs
Performance metrics → accuracies.csv, F1scores.csv
    -You can open these files to analyze model performance.

##### **Optional Outputs**
If requested, the following files will also be generated:

2D spectra → /2D_check_plots/[iteration number].png, 2D_check_plots/[iteration number].pkl
    -Both image files and the image data are saved
Training reports → /Training_reports/[iteration number].png
ML reports → /ML_reports/[iteration number].pkl
    -Detailed results. Save options are full (training and testing), partial (testing), or none.

---

## **Citing This Work**
If you use this code in your research, please cite:
[manuscript citation info pending]

### **License**
This project is licensed under the MIT License. See LICENSE for details.