# Statistically Augmented Graph Representations for Omics-Based Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

This repository contains the codebase for our study:

<em><strong>On why and how to encode probability distributions on graph representations of omics data: enhancing predictive tasks and knowledge discovery</strong></em>

We propose a graph-based framework for omics data where nodes and edges are annotated with structured statistical information that captures the underlying probability distributions of molecular interactions. This approach enhances both the predictive performance and interpretability of models applied to clinical tasks such as cancer survivability and primary diagnosis.

---

## 📁 Repository Structure

```
.
├── graphs/                    # GEXF files: graph representations per cancer, layer, and target
├── results/                   # Classification and cross-validation results
├── tcgahandler/               # Utilities for downloading, processing, and managing TCGA data
├── pydge/                     # R-based DGE filtering pipeline integration
├── *.py / *.sh                # Scripts for graph generation, learning, and evaluation
├── Dockerfile                 # A preconfigured environment to execute the code
├── requirements.txt           # Python dependencies
└── requirements.r             # R dependencies
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/dmgoncal/statgraph-omics.git
cd statgraph-omics
```

### 2. Build the Docker Image

Ensure [Docker](https://www.docker.com/) is installed.

```bash
docker build -t statgraphomics .
```

Launch an interactive session:

```bash
docker run -it --rm statgraphomics bash
```

This drops you into a container where both Python and R environments are preconfigured.

---

## 🧪 Main Features

- **Graph Construction**: Encodes omics data into graphs where statistical properties of biological entities and their pairwise relationships are embedded as nodes and edge weights, respectively.
- **Predictive Modeling**: Supports benchmarking of classical ML models and the proposed graph-based approach for clinical outcome prediction.
- **Explainability**: The statistically enriched graph structure can be leveraged for the identification of putative regulatory modules.
- **Multi-omics Support**: Compatible with miRNA, mRNA, and protein expression data from TCGA (extensible).

---

## 📊 Reproducibility

Experiments can be run as follows:

```bash
# Generate graphs
bash run_generate_full_graphs.sh

# Run graph-based classification (holdout)
bash run_graph_classification.sh

# Run machine learning baseline classification (holdout)
bash run_ml_classification.sh
```

Cross-validation variants are also available, but may be slower to execute (e.g., `run_graph_classification_cv.sh`).

---

## 📦 Requirements

### Python (version 3.10.14)

Install dependencies:

```bash
pip install -r requirements.txt
```

### R (version 4.4.3)

Required R packages are listed in `requirements.r`. You can install them via:

```r
Rscript requirements.r
```

Alternatively, let the Docker image handle this for you.

---

## 🧬 Data Sources

All omics and clinical data were obtained from [The Cancer Genome Atlas (TCGA)](https://www.cancer.gov/tcga). Currently, the pipeline supports the following cancer types (extensible):

- COAD (Colon Adenocarcinoma)
- KIRC (Kidney Renal Clear Cell Carcinoma)
- LGG (Lower Grade Glioma)
- LUAD (Lung Adenocarcinoma)
- OV (Ovarian Serous Cystadenocarcinoma)

---

## 📚 Citation

*Submitted for review. Citation will be added in the future.*

---

## ✉️ Contact

For questions, suggestions, or collaboration inquiries, contact the authors:
- Daniel M. Gonçalves - dmgoncalves@tecnico.ulisboa.pt
- André Patrício - andremppatricio@tecnico.ulisboa.pt
- Rafael S. Costa - rs.costa@fct.unl.pt
- Rui Henriques - rmch@tecnico.ulisboa.pt
