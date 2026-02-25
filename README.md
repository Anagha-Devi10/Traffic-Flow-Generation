#  Traffic Flow Generation using Latent GAN

Synthetic Traffic Sequence Generation using PeMS Sensor Data

---

##  Overview

This project implements a **Latent Space Generative Adversarial Network (GAN)** for generating realistic traffic flow time-series data using sensor readings from the **PeMS (Performance Measurement System)** dataset.

The model learns temporal patterns in traffic flow data (Veh/5 Minutes) and generates synthetic traffic sequences that preserve daily patterns and distribution characteristics.

Unlike simple forecasting models, this project focuses on **realistic sequence generation** using a hybrid architecture combining:

* LSTM-based Encoder–Decoder
* Temporal Convolutional Blocks
* Positional Embeddings
* Spectral Normalization
* Hinge Loss GAN Training

---

##  Dataset

* Source: **PeMS (California Department of Transportation)**
* Data Type: Traffic flow (Vehicles per 5 Minutes)
* Preprocessing:

  * CSV files merged
  * Min-Max scaling to [-1, 1]
  * Sequence length: 288 (represents one full day)
  * Daily segmentation using stride = 288

Each training sample represents **one full day of traffic flow**.

---

##  Model Architecture

### Phase 1 – Latent Representation Learning

**Embedder (Encoder)**

* 2-layer LSTM
* Linear projection to embedding space
* Learns compact temporal representation

**Recovery (Decoder)**

* 2-layer LSTM
* Reconstructs original sequence from latent embedding

This phase ensures meaningful latent representations before GAN training.

---

### Phase 2 – Latent GAN Training

####  Generator

* Fully connected projection from noise vector
* Positional embeddings
* Temporal Convolution Blocks (dilated)
* Outputs latent sequence representation

####  Discriminator

* Bidirectional LSTM
* Spectral-normalized Conv1D layers
* Fully connected classification head
* Hinge loss objective

---

##  Key Innovations

 * Two-phase training (Autoencoder + GAN)
 * Hinge loss for stable adversarial training
 * Spectral normalization for discriminator stability
 * Temporal convolution with dilation
 * Position-aware latent sequence modeling
 * Fully reproducible training (seed control)

---

##  Hyperparameters

| Parameter           | Value |
| ------------------- | ----- |
| Sequence Length     | 288   |
| Latent Dimension    | 250   |
| Embedding Dimension | 64    |
| Hidden Dimension    | 64    |
| AE Epochs           | 100   |
| GAN Epochs          | 2000  |
| Batch Size          | 32    |

---

##  Training Environment

* Framework: PyTorch
* Environment: Google Colab (GPU enabled)
* Optimizer: Adam
* GAN Loss: Hinge Loss
* Device: CUDA (if available)

---

##  Output

The model generates synthetic traffic flow sequences with shape:

```
(30, 288, 1)
```

Each sample represents one day of synthetic traffic data.

The generated sequences are inverse-scaled back to original traffic flow values.

---

##  How to Run

1. Install dependencies:

```
pip install torch numpy pandas scikit-learn matplotlib
```

2. Update dataset path:

```
FILE_PATH_PATTERN = "/path/to/pems/data/*.csv"
```

3. Run training script.

---

##  Applications

* Traffic simulation
* Data augmentation for forecasting models
* Smart city infrastructure research
* Anomaly detection benchmarking
* Synthetic mobility pattern generation

---

##  Future Improvements

* Wasserstein GAN with Gradient Penalty
* Transformer-based temporal modeling
* Multi-sensor spatial modeling
* Conditional GAN for peak/off-peak generation
* Quantitative evaluation (FID-style metric for time-series)
