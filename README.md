# SNN

This repository contains the code for my senior project on **Spiking Neural Networks (SNNs)**.  
The project implements and compares multiple neural coding strategies—**Rate**, **Time-to-First-Spike (TTFS)**, **Phase**, and **Burst**—using a unified SNN architecture on static (frame-based) and event-based datasets.

The primary objective of this project is to study how different neural coding schemes influence **accuracy, inference latency, spike activity (energy proxy), throughput, and robustness**, using evaluation metrics that better reflect the temporal and event-driven nature of SNNs.

---

## Project Motivation

Conventional artificial neural networks (ANNs) are typically evaluated using static metrics such as classification accuracy or floating-point operations (FLOPs). However, SNNs operate through sparse, time-dependent spike dynamics, making such metrics insufficient for capturing their computational behavior.

Neural coding strategies play a central role in SNN performance, as they determine **when** and **how often** neurons emit spikes. This project aims to provide a fair and systematic comparison of common coding schemes under identical network architectures and training conditions, while emphasizing **SNN-specific performance trade-offs**.

---

## Neural Coding Schemes

The following input coding schemes are implemented and evaluated:

- **Rate coding**  
  Encodes information in the average firing rate over time. This approach is robust but typically incurs high spike activity.

- **Time-to-First-Spike (TTFS)**  
  Encodes information in the timing of the first spike. This method is temporally sparse and energy-efficient, but sensitive to noise.

- **Phase coding**  
  Encodes input magnitude using spike timing relative to a reference oscillatory phase.

- **Burst coding**  
  Encodes information using short bursts of spikes, trading off redundancy and temporal precision.

All coding schemes are evaluated using the same SNN architecture to ensure fair comparison.

---

## Model Architecture

- **Input:** 28×28 grayscale images (flattened)
- **Hidden layer:** Leaky Integrate-and-Fire (LIF) neurons
- **Readout layer:** Linear classifier with temporal low-pass integration
- **Inference:** Fixed time window with optional early decision analysis

---

## Evaluation Metrics

To capture the characteristics of spiking computation, this project evaluates:

- **Accuracy** – classification performance
- **Inference latency** – number of timesteps required to reach a stable decision
- **Energy efficiency (proxy)** – average spike count per sample
- **Throughput** – samples processed per second
- **Robustness** – accuracy under input perturbations

### Robustness Tests
- Gaussian noise (σ = 0.05, 0.10, 0.20)
- Salt-and-pepper noise (p = 0.01, 0.05, 0.10)

---

## Repository Structure
SNN/
├── encoding/ # Neural coding schemes (rate, TTFS, phase, burst)
├── models/ # LIF neurons and SNN model definition
├── train/ # Training and evaluation code
├── experiments/ # Scripts for running each coding scheme
└── README.md

---

## Running the Experiments

### Requirements
- Python ≥ 3.9
- PyTorch
- torchvision

Install dependencies:
```bash
pip install torch torchvision

- Run experiments from the repository root:
python3 -m experiments.run_rate
python3 -m experiments.run_ttfs
python3 -m experiments.run_phase
python3 -m experiments.run_burst


- To save logs:
python3 -m experiments.run_rate  > rate.txt
python3 -m experiments.run_ttfs  > ttfs.txt
