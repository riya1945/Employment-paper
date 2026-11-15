
# Early Fusion of Dual-Transformer Embeddings with an XGBoost Meta-Learner for Automated Resume Classification

[![IEEE Access](https://img.shields.io/badge/IEEE-Access-blue.svg)](https://ieeexplore.ieee.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official source code, benchmark implementations, and reproduction artifacts for the research paper:

> **"Early Fusion of Dual-Transformer Embeddings with an XGBoost Meta-Learner for Automated Resume Classification"**  
> *Riya Joshi, Sivakami Sundari, Nikhil, Sanjeet S. Giri*  
> Published in **IEEE Access** | Manuscript Tracking: **Access-2026-45264**

---

## 📌 Abstract & Research Overview
Automated resume classification is a fundamental front-end stage in Applicant Tracking Systems (ATS), routing high-volume unstructured resumes to appropriate occupational queues. Traditional approaches rely either on lexical keyword matching (TF-IDF), single-transformer representations, or late probability fusion. However, individual representation spaces struggle to balance fine-grained occupational terminology with broad semantic domain boundaries.

This repository implements our **Dual-Transformer Early Fusion Framework**, which horizontally concatenates:
1. **Localized contextual representations** extracted from the `[CLS]` token of a fine-tuned **DistilBERT** encoder (768-D).
2. **Global sentence semantics** extracted via a pre-trained **MPNet** encoder (`all-mpnet-base-v2`, 768-D).

The resulting 1536-dimensional concatenated feature space is classified using an **XGBoost gradient-boosted decision tree meta-learner**, effectively partitioning non-linear interactions across heterogeneous contextual and semantic subspaces.

---

## 📊 Benchmark Results (Held-Out Test Partition, $N=376$)

All models are evaluated on the exact same held-out test partition ($N=376$) across the unmerged **17 professional categories** (matching **Table 4** of the manuscript):

| Paradigm | Model Architecture | Correct / 376 | Accuracy | Macro F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Lexical IR Baseline** | TF-IDF + Cosine Similarity | 210 / 376 | 55.85% (0.5585) | 0.4426 | Phase I |
| **Lexical IR Baseline** | Keyword Co-occurrence Matching | 174 / 376 | 46.28% (0.4628) | 0.4255 | Phase I |
| **Hybrid Retrieval** | Unsupervised Hybrid (TF-IDF + MPNet, $\alpha=0.90$) | 203 / 376 | 53.99% (0.5399) | 0.5158 | Phase I |
| **Semantic Retrieval** | MPNet Dense Semantic Matching | 221 / 376 | 58.78% (0.5878) | 0.5466 | Phase I |
| **Supervised Lexical** | XGBoost (TF-IDF Features, 5000-D) | 303 / 376 | 80.59% (0.8059) | 0.7978 | Phase II |
| **Contextual Deep Learning**| DistilBERT (Fine-Tuned Linear Head) | 316 / 376 | 84.04% (0.8404) | 0.8300 | Phase II |
| **Single Transformer** | XGBoost (DistilBERT [CLS] Only, 768-D) | 316 / 376 | 84.04% (0.8404) | 0.8321 | Phase II |
| **Single Transformer** | XGBoost (MPNet Embeddings Only, 768-D) | 320 / 376 | 85.11% (0.8511) | 0.8440 | Phase II |
| **Alternative Fusion** | Late Fusion (Soft Probability Averaging) | 322 / 376 | 85.64% (0.8564) | 0.8512 | Phase II |
| **Alternative Meta-Learner**| Linear Meta-Learner (Logistic Regression) | 324 / 376 | 86.17% (0.8617) | 0.8570 | Phase II |
| **Alternative Meta-Learner**| Random Forest (1536-D, 100 Trees) | 332 / 376 | 88.30% (0.8830) | 0.8792 | Phase II |
| **Proposed Framework** | **Dual-Transformer Early Fusion + XGBoost** | **335 / 376** | **89.10% (0.8910)** | **0.8859** | **Phase II** |

### 📈 Statistical Significance & Validation
- **McNemar's Paired Test (Early Fusion vs. MPNet):** $\chi^2 = 5.000$ ($p = 0.0253$), Yates' corrected $\chi^2 = 4.356$ ($p = 0.0369$), Exact Binomial $p = 0.0362$. Relative error reduction of **26.79%** ($41$ vs. $56$ errors) with a $2:1$ discordant advantage ($n_{01}=30$ vs. $n_{10}=15$).
- **5-Fold Cross-Validation (Training Split, $N=1,504$):** Mean accuracy of **89.68% $\pm$ 2.18%**, Macro F1 of **0.8899 $\pm$ 0.0242**.
- **Representation Complementarity (CKA Analysis):** Linear CKA score of **0.412** and mean cross-cosine similarity of **0.384**, confirming that DistilBERT and MPNet span orthogonal semantic subspaces.

---

## 🗂️ Repository Structure

```plaintext
.
├── README.md                              # Comprehensive project overview & reproduction guide
├── requirements.txt                       # Verified pinned dependencies
├── LICENSE                                # MIT Open-Source License
├── prepare_data.py                        # Automated data ingestion, harmonization & leakage audit
├── evaluate.py                            # Standalone end-to-end CLI benchmark evaluation script
│
├── resume_classification_benchmark.ipynb  # Primary Master Benchmark Notebook (Self-contained, All-In-One)
├── phase1.ipynb                           # Phase I Notebook: Data Harmonization, EDA & Retrieval Baselines
├── phase2.ipynb                           # Phase II Notebook: Supervised Transformers & Early Fusion XGBoost
│
├── clean_resumes.csv                      # (Optional local cache) 1,880 harmonized candidate resumes
└── clean_job_descriptions.csv             # (Optional local cache) 42,487 harmonized JobStreet vacancies
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
Clone this repository and install dependencies in a clean Python 3.10+ virtual environment:
```bash
git clone https://github.com/your-username/resume-classification-early-fusion.git
cd resume-classification-early-fusion

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Automated Dataset Ingestion & Harmonization
Run the data curation pipeline. This script pulls raw data directly from HuggingFace, maps the 18-to-17 category taxonomy, performs the SHA-256 deduplication audit, and sets up the deterministic 80:20 split:
```bash
python prepare_data.py
```
*Note: If `clean_resumes.csv` is already present, this step is optional.*

### 3. Run the End-to-End Evaluation (CLI)
Reproduce the complete benchmark table and McNemar's test in a single terminal command:
```bash
python evaluate.py
```

### 4. Interactive Jupyter Notebooks
Alternatively, open the interactive notebooks for step-by-step experimentation and visualization:
- **`resume_classification_benchmark.ipynb`**: Complete, self-contained master reproduction notebook covering all data pipelines, baseline models, meta-learners, statistical tests, CKA analysis, and publication-ready diagnostic charts.
- **`phase1.ipynb`**: Focuses on dataset provenance, exploratory data analysis (EDA), TF-IDF retrieval, cosine similarity score distributions, and the unsupervised linear hybrid retrieval baseline ($\alpha$ grid search).
- **`phase2.ipynb`**: Focuses on supervised deep learning: fine-tuning DistilBERT, extracting MPNet embeddings, training the Early Fusion XGBoost meta-learner, and plotting per-class F1 comparisons.

---

## 📦 Dataset Provenance & Licenses

This benchmark utilizes two public recruitment datasets:
1. **Candidate Resumes Corpus**: LiveCareer career examples hosted on HuggingFace ([`opensporks/resumes`](https://huggingface.co/datasets/opensporks/resumes)), distributed under the **Creative Commons CC0-1.0 Universal** license. Contains 2,484 raw resumes across 24 original categories, curated to 1,880 resumes across 17 professional domains.
2. **Job Postings Corpus**: JobStreet market vacancies across Malaysia compiled by Mahadan ([`azrai99/job-dataset`](https://huggingface.co/datasets/azrai99/job-dataset)), released under the **Creative Commons CC BY 4.0** license. Contains 59,300 raw listings across 30 industries, curated to 42,487 postings across 17 aligned sectors.

*Data Governance & Ethics:* All personally identifiable information (PII) including email addresses, phone numbers, and external URLs was sanitized. No live human subjects or active candidate screenings were involved.

---

## ⚙️ Hardware & Compute Environment
- **Operating System:** Windows 10/11 or Ubuntu 22.04 LTS
- **Accelerator:** NVIDIA GPU with CUDA 12.x (Tested on NVIDIA RTX 3080 / A100)
- **CPU / RAM:** 16 GB RAM minimum (32 GB recommended for full job description embeddings)
- **Deterministic Seed:** All experiments pin `random_state = 42` across Python, NumPy, PyTorch, and XGBoost.

---

## 📜 Citation
If you find this code or research helpful in your work, please cite our IEEE Access paper:

```bibtex
@article{joshi2026earlyfusion,
  title={Early Fusion of Dual-Transformer Embeddings with an XGBoost Meta-Learner for Automated Resume Classification},
  author={Joshi, Riya and Sundari, Sivakami and Nikhil and Giri, Sanjeet S.},
  journal={IEEE Access},
  volume={14},
  pages={1--19},
  year={2026},
  doi={10.1109/ACCESS.2026.45264}
}
```

---

## 📄 License
This repository is licensed under the [MIT License](LICENSE).

