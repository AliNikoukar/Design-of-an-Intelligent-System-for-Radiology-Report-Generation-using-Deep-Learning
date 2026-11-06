# Deep Multimodal Architecture for Automated Chest Radiograph Report Generation

> **B.Sc. Thesis** | **Department of Electrical Engineering, Amirkabir University of Technology (Tehran Polytechnic)**  
> **Author:** Ali Nikoukar

[![CI](https://github.com/AliNikoukar/Design-of-an-Intelligent-System-for-Radiology-Report-Generation-using-Deep-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/AliNikoukar/Design-of-an-Intelligent-System-for-Radiology-Report-Generation-using-Deep-Learning/actions)
[![Tests](https://img.shields.io/badge/tests-12%20passed-brightgreen.svg)](tests/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A deep multimodal learning framework for automated clinical radiology report generation from chest radiographs (CXR). Developed and validated on the **MIMIC-CXR-JPG** benchmark (227,827 studies) and tested for cross-domain generalization on the **Indiana University Chest X-Ray (IU-Xray)** cohort.

---

## 1. Architectural Highlights

The system advances through 7 architectural generations, culminating in **Model 7** (Perceiver-Qwen Dual-Stream Framework):

```text
                                  +-------------------------------------------------+
                                  |            Chest Radiograph (CXR)               |
                                  +-------------------------------------------------+
                                                           |
                                                           v
                                  +-------------------------------------------------+
                                  |      Rad-DINO Visual Encoder (ViT-B/16)         |
                                  |        (137M Params, 16x16 Grid = 256 Patches)  |
                                  +-------------------------------------------------+
                                                           |
                                                           v
                                  +-------------------------------------------------+
                                  |           Perceiver Resampler Bridge            |
                                  |      (32 Learnable Latent Queries x 768-d)      |
                                  +-------------------------------------------------+
                                                           |
                                      +--------------------+--------------------+
                                      |                                         |
                                      v                                         v
                      +-------------------------------+         +-------------------------------+
                      |   Pathology Classifier Head   |         |   Cross-Modal Soft Projector  |
                      |    (Attention-Pool + ASL)     |         |     (768-d -> 2048-d Qwen)    |
                      +-------------------------------+         +-------------------------------+
                                      |                                         |
                                      v                                         v
                      +-------------------------------+         +-------------------------------+
                      | Diagnostic Multi-Label Loss   |         |    Qwen-2.5-3B-Instruct LLM   |
                      |   (14 CheXpert Pathologies)   |         |   (LoRA r=16, alpha=32, QLoRA)|
                      +-------------------------------+         +-------------------------------+
                                                                                ^
                                                                                |
                                  +---------------------------------------------+
                                  | Selective BM25 Exemplar Memory Bank (RAG)   |
                                  +---------------------------------------------+
```

1. **Domain-Specific Visual Representation:** Domain-adapted **Rad-DINO** (ViT-B/16 pre-trained on CXR) preserving high-frequency interstitial and parenchymal tissue patterns.
2. **Latent Query Compression:** A **Perceiver Resampler** cross-attention bottleneck compressing 256 visual patch tokens into 32 compact latent representations, eliminating quadratic attention complexity.
3. **Dual-Stream Multi-Task Guidance:** Joint optimization of language generation with a specialized multi-label diagnostic classification branch supervised via **Asymmetric Loss (ASL)** to alleviate severe clinical class imbalance.
4. **Parameter-Efficient Adaptation:** 4-bit NF4 quantized **Qwen-2.5-3B-Instruct** backbone adapted via Low-Rank Adaptation (LoRA, $r=16, \alpha=32$).
5. **Selective Retrieval Augmentation (RAG):** Contextual injection of semantically and clinically matched prior historical impressions to regularize generation vocabulary.

---

## 2. Quantitative Evaluation Benchmarks

All models were evaluated on the official MIMIC-CXR test set (N = 6,388) under identical protocol.

### A. Language Generation Metrics (MIMIC-CXR Official Test Set)

| Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | Distinct-1 | Distinct-2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A** (DenseNet-121 + LSTM) | 0.282 | 0.165 | 0.104 | 0.068 | 0.231 | 0.042 | 0.118 |
| **Baseline B** (DenseNet-121 + Additive Attention LSTM) | 0.314 | 0.191 | 0.128 | 0.089 | 0.254 | 0.058 | 0.154 |
| **Baseline C** (DenseNet-121 + DistilGPT2) | 0.339 | 0.212 | 0.141 | 0.096 | 0.268 | 0.071 | 0.189 |
| **Model 1** (Multi-Task DenseNet + DistilGPT2) | 0.351 | 0.224 | 0.152 | 0.106 | 0.279 | 0.083 | 0.215 |
| **Model 2** (DenseNet + RAG + DistilGPT2) | 0.368 | 0.239 | 0.165 | 0.117 | 0.291 | 0.098 | 0.248 |
| **Model 4** (BiomedCLIP + Qwen-2.5-3B LoRA) | 0.384 | 0.256 | 0.180 | 0.131 | 0.308 | 0.112 | 0.284 |
| **Model 5** (BiomedCLIP + Weighted Focal ASL + Qwen) | 0.392 | 0.264 | 0.188 | 0.138 | 0.315 | 0.119 | 0.297 |
| **Model 6** (Rad-DINO + AttentionPool Classifier + Qwen) | 0.407 | 0.279 | 0.201 | 0.149 | 0.329 | 0.131 | 0.326 |
| **Model 7** (Rad-DINO + Perceiver Resampler + ASL + Qwen) | **0.421** | **0.293** | **0.214** | **0.160** | **0.342** | **0.145** | **0.354** |

---

### B. Clinical Efficacy Metrics (14 CheXpert Pathologies via NegEx)

| Architecture | Macro Precision | Macro Recall | Macro F1 | Micro F1 | Mean AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline A** | 0.284 | 0.221 | 0.249 | 0.372 | 0.612 |
| **Baseline B** | 0.312 | 0.258 | 0.282 | 0.408 | 0.648 |
| **Baseline C** | 0.338 | 0.291 | 0.312 | 0.439 | 0.679 |
| **Model 4** | 0.412 | 0.386 | 0.398 | 0.521 | 0.748 |
| **Model 5** | 0.445 | 0.429 | 0.437 | 0.554 | 0.778 |
| **Model 6** | 0.478 | 0.463 | 0.470 | 0.589 | 0.812 |
| **Model 7** | **0.512** | **0.498** | **0.505** | **0.624** | **0.841** |

---

### C. Out-Of-Distribution (OOD) Robustness (IU-Xray External Cohort)

| Configuration | Test Cohort | BLEU-4 | ROUGE-L | Macro F1 | Micro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model 7 (In-Distribution)** | MIMIC-CXR (N = 6,388) | 0.160 | 0.342 | 0.505 | 0.624 |
| **Model 7 (Zero-Shot OOD)** | IU-Xray (N = 3,666) | 0.138 | 0.318 | 0.462 | 0.578 |
| **Relative Domain Degradation** | *Cross-Domain Delta* | *-13.7%* | *-7.0%* | *-8.5%* | *-7.4%* |

---

## 3. Visualizations & Explainability

### Explainable AI: Pathology-Conditioned 2D Visual Attention Heatmaps
![Attention Heatmaps](results/model7_mimic/pathology_attention_heatmaps.png)

### Multi-Label Pathology Confusion Matrices (14 CheXpert Classes)
![Confusion Matrices](results/model7_mimic/pathology_confusion_matrices.png)

### Out-Of-Distribution Evaluation (MIMIC-CXR vs IU-Xray)
<p align="center">
  <img src="results/model7_iu_xray_ood/ood_radar_comparison.png" width="48%" />
  <img src="results/model7_iu_xray_ood/ood_bar_comparison.png" width="48%" />
</p>

---

## 4. Qualitative Clinical Case Studies

| Case ID | Radiologist Ground Truth | Model 7 Generated Report | Clinical Agreement |
| :--- | :--- | :--- | :---: |
| **Study 55557761** | No acute cardiopulmonary process. | No acute cardiopulmonary process. | **Exact Match (1.0)** |
| **Study 54394630** | Mild to moderate left pleural effusion size as is adjacent left basal atelectasis. No evidence of pneumothorax. | 1. Patchy opacities in right mid and lower lung likely reflect atelectasis. 2. Small bilateral pleural effusions. No pneumothorax. | **High (Atelectasis + Effusion)** |
| **Study 54850191** | Large scale consolidation in both lower lungs, most likely severe pneumonia. Mild to moderate cardiomegaly unchanged. | 1. Moderate cardiomegaly. 2. Retrocardiac consolidation, likely combination of atelectasis and effusion. Superimposed pneumonia considered. | **High (Cardiomegaly + Consolidation)** |

---

## 5. Repository Structure

```text
├── .github/
│   └── workflows/ci.yml           # Automated GitHub Actions CI workflow
├── notebooks/                     # 24 reproducible, executed Jupyter notebooks
│   ├── 00_eda/                    # Exploratory data analysis & pathology distributions
│   ├── 01_baselines/              # Baselines A (CNN-LSTM), B (Attention), C (DistilGPT2)
│   ├── 02_multimodal_generations/ # Models 1 to 7 (Vision-Language, QLoRA, Perceiver)
│   └── 03_ood_iu_xray/            # External cohort ETL pipeline & OOD test evaluation
├── results/                       # Quantitative evaluation outputs & diagnostic artifacts
│   ├── model6/                    # Confusion matrices and attention heatmaps
│   ├── model7_mimic/              # Master MIMIC-CXR predictions, metrics & case studies
│   └── model7_iu_xray_ood/        # IU-Xray OOD predictions, radar plots, comparisons
├── scripts/                       # Production CLI tools
│   ├── evaluate.py                # Comprehensive test-set evaluation pipeline
│   ├── inference.py               # Single-case inference with 2D visual attention maps
│   └── train.py                   # Multi-task training engine with 4-bit QLoRA
├── src/                           # Modular, publication-grade Python package
│   ├── data/                      # Dataset loaders, CheXpert constants, regex extractors
│   ├── metrics/                   # Natural language & NegEx clinical metrics + bootstrap
│   ├── models/                    # Rad-DINO, Perceiver Resampler, ASL & Model 1-7
│   └── utils/                     # Device helpers, logging, visualization utilities
├── tests/                         # Automated unit test suite (12 tests)
├── .gitignore                     # Exclusion rules for clinical datasets & checkpoints
├── LICENSE                        # MIT Open-Source License
├── pyproject.toml                 # Modern PEP 517 / PEP 621 package build configuration
├── README.md                      # Project documentation
├── requirements.txt               # Pinned dependencies
└── setup.py                       # Python package configuration
```

---

## 6. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/AliNikoukar/Design-of-an-Intelligent-System-for-Radiology-Report-Generation-using-Deep-Learning.git
cd Design-of-an-Intelligent-System-for-Radiology-Report-Generation-using-Deep-Learning

# Create and activate conda environment
conda create -n radgen python=3.10 -y
conda activate radgen

# Install PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install package and dependencies
pip install -r requirements.txt
pip install -e .
```

### Running Automated Tests
```bash
python -m unittest discover tests -v
```

---

## 7. Quickstart & CLI Usage

### A. Full Evaluation Pipeline
Run official test-set evaluation on MIMIC-CXR:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/model7_qwen_best.pt \
    --features data/features_spatial_test.dat \
    --manifest data/mimic_master_split_with_rag.csv \
    --thresholds data/optimal_thresholds.json \
    --output_dir results/eval_run \
    --use_rag
```

### B. Single-Case Inference & Attention Heatmap Generation
Generate a report with 2D visual attention heatmap overlay:
```bash
python scripts/inference.py \
    --checkpoint checkpoints/model7_qwen_best.pt \
    --features data/sample_features.npy \
    --output_image results/case_study_heatmap.png \
    --num_beams 3
```

### C. Multi-Task Model Training
Train Model 7 with joint Asymmetric Loss and language generation:
```bash
python scripts/train.py \
    --manifest data/mimic_master_split_with_rag.csv \
    --features data/features_spatial_train.dat \
    --epochs 6 \
    --batch_size 16 \
    --lr 2e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --output_dir checkpoints/model7
```

---

## 8. Ethics & Data Compliance

All clinical experiments were conducted in accordance with the **PhysioNet Credentialed Health Data Use Agreement (DUA)** for the MIMIC-CXR dataset. Raw DICOM files and de-identified patient identifiers are strictly excluded from this repository under HIPAA compliance. To reproduce results from raw radiographs, researchers must complete CITI training and request access via [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).

---

## 9. License & Citation

This project is licensed under the terms of the [MIT License](LICENSE).

```bibtex
@thesis{nikoukar2026deepmultimodal,
  title  = {Deep Multimodal Architecture for Automated Chest Radiograph Report Generation},
  author = {Nikoukar, Ali},
  type   = {B.Sc. Thesis},
  school = {Department of Electrical Engineering, Amirkabir University of Technology (Tehran Polytechnic)},
  year   = {2026}
}
```
