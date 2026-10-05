# MS4N

**A Simple State Space Model Excels at Multivariate Time Series Classification**

**Authors:** Hassan Saadatmand, Geoffrey I. Webb, Hamid Rezatofighi, Mahsa Salehi

[![Paper](https://img.shields.io/badge/arXiv-2605.27406-b31b1b.svg)](https://arxiv.org/abs/2605.27406)

---

## Overview

This repository contains the implementation of **MS4N**, lightweight
state space models for both univariate and multivariate time series classification.

The models are based on the diagonal state space model **S4D** and introduce
simple and lightweight architectural modifications for effective and efficient time-series
classification.

**MS4N** extends S4D with:

- a linear input projection
- a channel-mixing mechanism
- a layer normalization to improve training stability.

The paper provides a systematic evaluation of diagonal SSMs and Mamba-based
models for time-series classification across large-scale benchmark datasets.

MS4N is designed to be **simple, tiny, and efficient**, while remaining highly **effective**:

- No dependency on specialized libraries (e.g., no custom CUDA kernels, no
  Mamba-specific packages) — built entirely with standard PyTorch operations.
- A compact parameter footprint compared to Transformer- and Mamba-based
  alternatives.
- Fast training and inference due to its lightweight diagonal SSM backbone.

Despite this simplicity, MS4N matches or outperforms far more complex models
across large-scale time-series classification benchmarks.

---

## Paper

**A Simple State Space Model Excels at Multivariate Time Series Classification**

Hassan Saadatmand, Geoffrey I. Webb, Hamid Rezatofighi, Mahsa Salehi

Paper: https://arxiv.org/abs/2605.27406

---

## Repository Structure

```text
MS4N/
│
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── .gitignore
│
├── models/
│   └── MS4N.py
|   └── RNN_Class.py  # RNN, GRU, LSTM
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   └── model_factory.py
|   └── trainer.py
│
├── main.py
```
