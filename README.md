
# README for Protein Contact Prediction Project

## Introduction

This project focuses on **predicting residue-residue contacts in proteins** using embeddings generated by the **ESM2 (Evolutionary Scale Modeling 2)** architecture. The goal is to advance protein sequence analysis by leveraging **large language models (LLMs)** to capture complex biological patterns. The proposed approach involves developing two models based on **Transformers** and **GRUs** for contact map prediction, aiming to enhance structural understanding of proteins.

---

## Features
- **Data preprocessing**: Extracts amino acid sequences, filters irrelevant DNA/RNA data, and generates contact maps from protein structures.
- **Model architectures**:
  - **Transformer-based model**: Captures long-range dependencies in sequences.
  - **GRU-based model**: Efficiently processes sequential dependencies in amino acid sequences.
- **Custom loss functions**:
  - **Weighted Binary Cross-Entropy**.
  - **Focal Loss** to handle imbalanced datasets.
- **Training utilities**: Supports padding/masking and batch processing for sequences of varying lengths.

---

## Installation

### Prerequisites
- Python 3.12 or higher
- GPU-enabled machine with CUDA (recommended for faster training)
- Required libraries:
  - `torch`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `tqdm`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/RaffiSatamyan/ProteinContactMap
   cd ProteinContactMap
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Preparation

1. **Dataset**: Download protein structures from the [RCSB PDB database](https://www.rcsb.org/).
2. **Preprocessing**:
   - Remove DNA/RNA sequences.
   - Convert PDB files to **FASTA format**.
   - Generate distance matrices and contact maps using an 8Å threshold.

3. **Custom Dataset Loader**:
   - Handles sequence padding/masking for variable-length proteins.

---

## Models

1. **Transformer-Based Model**:
   - Processes ESM-2 embeddings.
   - Captures global dependencies in sequences.
   - Outputs a square matrix with probabilities of residue-residue contacts.

2. **GRU-Based Model**:
   - Uses sequential GRU layers to process embeddings.
   - Captures short-term dependencies in sequences.

**Shared Workflow**:
- Embeddings → Pairwise Difference Computation → Feedforward Network → Sigmoid Activation.

---

## Training

### Steps
1. Monitor performance with metrics:
   - **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.
2. Save/load model parameters using custom utilities.

### Challenges Addressed
- **Class imbalance**: Resolved using weighted loss functions.
- **Variable sequence lengths**: Managed via masking and padding.

---

## Results & Limitations

### Results
- Preliminary results indicate the model struggles to identify meaningful patterns, often predicting uniform probabilities.
- Limitations in data quality and architecture likely contributed to poor performance.

### Future Work
- Incorporate structural data from similar proteins.
- Explore alternative architectures (e.g., CNNs).
- Increase dataset size and diversity.
- Optimize hyperparameters with techniques like grid search.

---

## References
- RCSB PDB: [https://www.rcsb.org/](https://www.rcsb.org/)
- FASTA format: [https://en.wikipedia.org/wiki/FASTA_format](https://en.wikipedia.org/wiki/FASTA_format)
- Protein structure analysis: [Meta AI Research](https://doi.org/10.1101/2022.07.20.500902)

---

Feel free to contribute and share feedback for further improvement!
