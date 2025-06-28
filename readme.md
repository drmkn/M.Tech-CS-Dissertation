# Dissertation: Causal Explanations in Deep Learning Systems

This repository contains the code, data and experiments for the M.Tech Computer Science dissertation at the Indian Statistical Institute, Kolkata, titled **"Causal Explanations in Deep Learning Systems"**.

---

## 📖 Thesis Overview

In this work, we propose and implement **Intrinsic Causal Contribution (ICC)**, a novel method to quantify the causal influence of input features on a neural network’s prediction, beyond correlations induced by other features. We:

- Model neural networks as Structural Causal Models (SCMs).
- Use **Causal Normalizing Flows (CNFs)** to learn the causal generative process of input features.
- Estimate ICC via the **Jansen estimator** for variance-based sensitivity analysis.
- Compare ICC against popular global attribution methods (SHAP, LIME, GAM, PFI, etc.) on synthetic and real-world datasets.


## 📰 Published Paper

**S. Saha, D. V. Rathore, S. Saha, U. Garain, D. Doermann**, *"On Measuring Intrinsic Causal Attributions in Deep Neural Networks"* Accepted at Causal Learning and Reasoning (CLeaR) 2025. [arXiv:2505.09660](https://arxiv.org/abs/2505.09660)

## 📦 Repository Structure

```
Dissertation/
├── .vscode/                          # Editor settings
├── assets/                           # Figures and diagrams
├── attributions/                     # Saved attribution maps
├── dags_estimated/                   # Estimated DAG structures
├── datasets/                         # Synthetic and benchmark data
├── explanations/                     # Generated explanations and examples
├── models/                           # Trained model checkpoints
├── .gitignore                        # Files and directories to ignore in Git
├── ICC.py                            # ICC estimator implementation
├── architectures.py                  # Neural network architectures 
├── attributions.ipynb                # Interactive notebook for attribution methods
├── dataloader.py                     # Data loading utilities
├── estimate_DAG.py                   # DAG estimation scripts
├── generate_attributions.py          # Pipeline to compute global attributions
├── kde_visualisation_callback.py     # KDE visualization helper
├── pl_modules.py                     # PyTorch Lightning modules
├── trainer.py                        # Training and evaluation script
└── utils.py                          # Utility functions
```

## 🔧 Key Dependencies

The main libraries used in this project are:

- `torch`: the core deep learning framework for building and training neural networks.
- `pytorch-lightning`: a high-level wrapper for organizing training loops, logging and checkpointing.
- `zuko`: provides Causal Normalizing Flow implementations (e.g., Masked Autoregressive Flows) for modeling SCMs.
- `gam`: Generalized Additive Models for summarizing local feature attributions.
- `openxai`: toolkit for generating a variety of attribution methods (Integrated Gradients, Input × Gradient, SmoothGrad, SHAP).
- `captum`: PyTorch interpretability library used here for permutation feature importance (PFI) and Lime/Splime baselines.

## ⚙️ Required Packages

Install the core libraries via pip:

```bash
pip install torch zuko openxai captum gam pytorch-lightning
```

## 🙏 Acknowledgements

I would like to express my sincere gratitude to Prof. [Utpal Garain](https://scholar.google.co.in/citations?user=4Jlqf30AAAAJ\&hl=en) for his invaluable guidance and support at the Indian Statistical Institute, Kolkata. I am also deeply thankful to [Saptarshi Saha](https://github.com/Saptarshi-Saha-1996) for his help and significant contributions to this work.





