# rock-hardness-graph-ensemble

# Graph Ensemble Network for Rock Hardness Recognition

This repository contains the core implementation of the proposed method.

## Files

- `graph_creator.py` — builds graph datasets from multi-sensor CSV files.
- `graph_ensemble_network.py` — defines the graph ensemble network and performs cross-validation.

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- scikit-learn

Install dependencies:

```bash
pip install torch torch-geometric scikit-learn
