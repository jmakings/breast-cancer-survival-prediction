# Breast Cancer Survival Risk Modeling — METABRIC

A comparison of classical and deep learning approaches to survival risk prediction using the METABRIC dataset. The project implements a Cox proportional hazards model and a hybrid deep learning architecture, evaluated on a held-out validation set using the concordance index and Kaplan-Meier risk group stratification.

### Model Setup and Current Rolling Results are available in *"Multi-Modal Breast Cancer Survival Analysis.pptx"*
---

## Overview

Predicting survival outcomes in breast cancer is a clinically meaningful problem with direct implications for treatment planning. This project explores two modeling paradigms on the same dataset and feature set:

- A **Cox proportional hazards model** using clinical and genomic features, implemented with the `lifelines` library
- A **hybrid deep learning model** combining a fully connected encoder for clinical and mutation data with an autoencoder for microarray gene expression, implemented in PyTorch

Both models are trained to predict a risk score and evaluated on their ability to rank patients by survival time and stratify them into clinically meaningful risk groups.

---

## Dataset

**Source:** [METABRIC dataset via Kaggle](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric)

The METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) dataset includes:

- **Microarray gene expression profiles** — genome-wide expression measurements (Z-scored)
- **Somatic mutation data** — binary mutation status across a panel of cancer-relevant genes
- **Clinical metadata** — age, tumor grade, stage, treatment, and other covariates
- **Survival outcomes** — overall survival time and event status (used for model training and evaluation)

> **Note:** The METABRIC data is not included in this repository. Download instructions are provided below.

---

## Model Architecture

### Cox proportional hazards model

Implemented using the `lifelines` library. Clinical metadata and mutation features are used as covariates. The model estimates a linear risk score based on the partial likelihood of the Cox model, with feature selection applied prior to fitting.

### Deep learning survival model

A hybrid architecture that processes the three data modalities separately before combining them:

```
Microarray expression (Z-scored)
        │
  Autoencoder encoder
  (32 latent dimensions)
        │
        ├────────────────────────┐
                                 │
Clinical metadata + mutations    │
        │                        │
Fully connected encoder          │
        │                        │
        └────────────┬───────────┘
                     │
              Concatenation
                     │
             Final encoder
                     │
               Risk score (scalar)
```

The autoencoder is pretrained to reconstruct the microarray expression input, and its encoder component is then used as a fixed or fine-tuned feature extractor within the full model. The clinical and mutation data pass through a separate fully connected encoder. The two representations are concatenated and passed through a final encoder that outputs a scalar risk score.

**Loss function:** Cox partial likelihood

```python
def cox_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss = -torch.sum((risk - log_cumsum) * event) / event.sum()
    return loss
```

---

## Preprocessing

| Data type | Preprocessing |
|---|---|
| Microarray expression | Z-scored (provided pre-normalized in METABRIC) |
| Clinical metadata | Missing values imputed with training set mean |
| Mutation data | Missing values imputed with training set mean |

All imputation parameters are fit on the training set and applied to the validation set to prevent data leakage.

---

## Evaluation

Models are evaluated on a held-out validation set using:

- **Concordance index (C-index)** — measures how well the model ranks pairs of patients by predicted survival time. A value of 0.5 indicates random performance; values above 0.65 are generally considered meaningful for survival models.
- **Kaplan-Meier risk group stratification** — patients are split into high- and low-risk groups based on the model's predicted risk score, and survival curves are plotted for each group to assess clinical separability.

### Results

| Model | Validation C-index |
|---|---|
| Cox proportional hazards | *0.70* |
| Deep learning (hybrid) | *0.66* |

Kaplan-Meier curves for both models are available in pptx with description

---

> **Note:** Notebooks are currently being cleaned up and refactored for clarity and reproducibility.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/metabric-survival
cd metabric-survival
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Purpose |
|---|---|
| `lifelines` | Cox proportional hazards model |
| `torch` | Deep learning model (PyTorch) |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Preprocessing utilities |

### 3. Download the data

Download the METABRIC dataset from Kaggle:

[https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric)

Place the downloaded files in a `data/` directory at the root of the repository:


### 4. Run the notebooks

Start with the preprocessing notebook in either `notebooks/cox_modeling.ipynb` or `notebooks/dl_modeling.ipynb` and run sequentially.

---

## Next Steps

- [ ] External validation on an independent cohort (TCGA BRCA)
- [ ] Model deployment as a lightweight inference API

---

## Acknowledgements

METABRIC data sourced from Kaggle. Original dataset described in:

> Curtis C, et al. *The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups.* Nature. 2012;486(7403):346-352.
