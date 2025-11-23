# vae-anomaly-detection
1. Project Overview

This project demonstrates:

Implementation of the VAE architecture

Encoder → μ and log(σ²)

Reparameterization trick

Decoder

ELBO loss (Reconstruction + KL Divergence)

Training the VAE only on normal samples

Computing reconstruction errors to detect anomalies

Training a baseline Autoencoder (AE) with similar architecture

Comparing both models using AUC-ROC scores

This tests understanding of:

Generative modeling

Probabilistic deep learning

2. Dataset Information

Dataset used: Fashion-MNIST
Each image is:

28 × 28 pixels

Grayscale

Flattened to 784-dimensional vector in code

Normal Class (Training Data)

Class 0: T-shirt/Top

Anomaly Classes (Testing Data)

Classes 1 to 9

The model learns the distribution of class 0 and flags other classes based on reconstruction error.

3. Variational Autoencoder (VAE) — Mathematical Foundation
3.1 Evidence Lower Bound (ELBO)

The VAE optimizes:

𝐿
=
𝐸
𝑞
(
𝑧
∣
𝑥
)
[
log
⁡
𝑝
(
𝑥
∣
𝑧
)
]
−
𝐾
𝐿
(
𝑞
(
𝑧
∣
𝑥
)
  
∣
∣
  
𝑝
(
𝑧
)
)
L=E
q(z∣x)
	​

[logp(x∣z)]−KL(q(z∣x)∣∣p(z))

Rewriting as a loss to minimize:

Loss
=
Reconstruction Loss
+
𝛽
⋅
𝐾
𝐿
Loss=Reconstruction Loss+β⋅KL

Where β = 1.0 is a standard VAE.

Optimization

Unsupervised anomaly detection
