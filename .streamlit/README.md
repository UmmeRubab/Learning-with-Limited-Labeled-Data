
# Learning with Limited Labeled Data

An experimental study of active learning strategies for
sample-efficient text classification.

## Research Question

Can active learning reduce the number of labeled examples
required to achieve competitive text classification performance?

## Overview

Supervised machine learning depends on labeled training data.
Obtaining these labels can be expensive and time-consuming.

This project investigates whether selecting informative
training examples improves classification performance under
limited labeling budgets.

## Dataset

AG News Topic Classification

- 120,000 training examples
- 7,600 official test examples
- Four categories: World, Sports, Business, Sci/Tech

Dataset:
https://huggingface.co/datasets/fancyzhx/ag_news

## Methodology

The project compares three acquisition strategies:

1. Random Sampling
2. Uncertainty Sampling (Predictive Entropy)
3. Diversity Sampling (Cosine Similarity)

All strategies use TF-IDF representations and Logistic
Regression for text classification.

The experiments evaluate performance at increasing
labeling budgets.

## Evaluation Metrics

- Macro F1
- Accuracy
- Performance across multiple random seeds
- Learning curves

## Application Features

- AG News dataset explorer
- Configurable labeling budgets
- Active learning experiments
- Interactive learning curves
- Strategy comparisons
- CSV result export
- Research methodology and limitations

## Installation

Clone the repository:

    git clone https://github.com/UmmeRubab/learning-with-limited-labels.git

Navigate to the project directory:

    cd learning-with-limited-labels

Create a virtual environment:

    python -m venv .venv

Activate it on Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements.txt

Run the application:

    python -m streamlit run app.py

## Testing

Run:

    python -m pytest -q

## Reproducibility

Experiments use fixed random seeds and identical initial
labeled examples for all sampling strategies within each seed.

The official AG News test split is excluded from model fitting
and acquisition.

## Research Limitations

This is an experimental implementation of established active
learning methods, not a claim of a new active learning algorithm.

Annotation effort is simulated using pre-existing dataset labels.

Repeated evaluation on the official test set is intended for
exploratory comparison. Confirmatory studies should reserve
a held-out test set and tune methods using validation data.

Results may vary across datasets and experimental settings.

## Reference

Zhang, X., Zhao, J., & LeCun, Y. (2015).
Character-level Convolutional Networks for Text Classification.

https://arxiv.org/abs/1509.01626
