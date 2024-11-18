# Neuromorphic_computing_final_project
Neuromorphic Computing Hardware Design Course Semester Project: **SNN-based Hardware Accelerator Design for MNIST Images**

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contact](#contact)

---

## Features
- **Small Architecture**: Shallow SNN architecture.
- **Ultra-Low Precision Quantization**: Employs gird-based quantization techniques (1.5 bit) tailored for hardware-aware spiking neural networks.
- **Sparse Activations**: Reduces computational overhead by leveraging sparsity in activations.
- **Energy Efficiency**: Optimized for on-device inference, ensuring low energy consumption for edge devices.
- **Flexible Framework**: Easy integration with popular deep learning libraries like PyTorch.
- **Scalability**: Demonstrates high performance across various SNN architectures and event/static datasets.

---

## Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.12
- CUDA (Optional, for GPU acceleration)

### Install Required Packages
Clone the repository and install dependencies:

```bash
git clone https://github.com/Ahmedhasssan/IM_SNN-SpQuant_SNN.git
cd IM_SNN-SpQuant_SNN
pip install -r requirements.txt
```

## Usage

### Example Scripts
The repository includes examples for training and evaluating SNNs on popular Event datasets, **DVS-MNIST** and **DVS-CIFAR10**:

```bash
Run Bash main.sh
```

## Methodology

SNN-Hardware accelerator design introduces:

1. **Low-precision SNN Inference**:
   - Reduced the precision of model weights to 2-bit and membrane potential to as low as 1.5 bits without degrading performance.

2. **Shallow SNN Architecture**:
   - Choose a shallow architecture to achieve high performance on DVS-MNIST and DVS-CIFAR10 datasets.
   - Utilizes sparsity in spiking activations to minimize redundant computations, achieving significant energy savings.

3. **End-to-End Pipeline**:
   - Combines quantization and sparsity in a unified framework for seamless on-device inference.
   - Designed an accelerator for the event-data classification and object detection

## Results

We achieved >98% accuracy with the 2-bit model in terms of weight and membrane potential precision.

Designed a hardware accelerator using ASIC tools for real-time on-device inference.

## Contact

For any inquiries or collaboration opportunities, feel free to reach out:

- **Email**: [ah2288.@cornell.edu](mailto:ah2288@cornell.edu)
- **GitHub**: [Ahmedhasssan](https://github.com/Ahmedhasssan)

We welcome feedback, suggestions, and contributions to enhance on-device-SNN work!
