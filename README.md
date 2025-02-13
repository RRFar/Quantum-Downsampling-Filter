# Quantum-Downsampling-Filter
This repository presents the implementation of a Quantum Variational Autoencoder (Q-VAE), a hybrid model designed to enhance image reconstruction quality by combining quantum computing in the encoder with CNNs in the decoder. Traditional Classical VAEs (C-VAEs) struggle with low-resolution inputs (16x16 pixels), leading to blurred and inaccurate reconstructions. To address this, Q-VAE upsamples images from 16x16 to 32x32 during encoding, aiming to preserve key features and improve reconstruction fidelity.

Key Features:

✔️ Quantum-Classical Hybrid Architecture – Quantum-enhanced encoding with CNN-based decoding.
✔️ Resolution Enhancement – Upscales low-resolution inputs for better feature preservation.
✔️ Performance Comparison – Evaluated against C-VAE and CDP-VAE (which uses windowing pooling filters).
✔️ Benchmark Datasets – Tested on MNIST & USPS datasets.
✔️ Quantitative Evaluation – Assessed using Fréchet Inception Distance (FID) and Mean Squared Error (MSE).
✔️ Superior Performance – Q-VAE outperforms both C-VAE and CDP-VAE, demonstrating its potential for image resolution enhancement and synthetic data generation.

📄 Read the full paper on arXiv: https://arxiv.org/abs/2501.06259
