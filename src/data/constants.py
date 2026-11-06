"""
Clinical constants, pathology taxonomy definitions, and prompt templates
for chest radiograph interpretation and report generation.
"""

import re

# Complete 14 CheXpert benchmark pathologies
CHEXPERT_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

# Core-10 conditions (prevalence > 5% in MIMIC-CXR benchmark)
CORE_10_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

# Ultra-rare conditions excluded from Core-10 macro averaging
RARE_4_PATHOLOGIES = [
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Pleural Other",
]

# Clinical keywords and synonym expansion for NegEx rule-based evaluation
CLINICAL_KEYWORDS = {
    "Atelectasis": ["atelectasis", "atelectatic", "collapse", "volume loss"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "enlarged cardiac", "cardiac enlargement"],
    "Consolidation": ["consolidation", "airspace disease", "airspace opacity"],
    "Edema": ["edema", "vascular congestion", "fluid overload", "interstitial edema"],
    "Enlarged Cardiomediastinum": ["widened mediastinum", "mediastinal widening", "cardiomediastinum"],
    "Fracture": ["fracture", "fractured", "broken rib", "healing fracture"],
    "Lung Lesion": ["lung lesion", "nodule", "mass", "nodular opacity"],
    "Lung Opacity": ["opacity", "opacities", "infiltrate", "clouding"],
    "No Finding": ["normal", "clear", "unremarkable", "no acute", "negative"],
    "Pleural Effusion": ["pleural effusion", "effusion", "blunting", "fluid in pleural"],
    "Pleural Other": ["pleural thickening", "fibrosis", "calcification"],
    "Pneumonia": ["pneumonia", "infectious process", "bronchopneumonia"],
    "Pneumothorax": ["pneumothorax", "air in pleural", "apical pneumothorax"],
    "Support Devices": ["tube", "catheter", "pacemaker", "pacer", "line", "picc", "stent"],
}

# Negation window regex patterns (35-character bidirectional window)
NEGATION_PREFIX_PATTERN = re.compile(
    r"\b(no|without|free of|negative for|absence of|denies|cleared of)\b\s*([^.;]{0,50})$",
    re.IGNORECASE,
)

NEGATION_SUFFIX_PATTERN = re.compile(
    r"^([^.;]{0,50})\s*\b(ruled out|is absent|resolved|excluded|unlikely|is not seen|not identified|has resolved|is clear)\b",
    re.IGNORECASE,
)

# Standard optimal thresholds tuned on MIMIC-CXR validation set
DEFAULT_OPTIMAL_THRESHOLDS = {
    "Atelectasis": 0.42,
    "Cardiomegaly": 0.38,
    "Consolidation": 0.35,
    "Edema": 0.40,
    "Enlarged Cardiomediastinum": 0.30,
    "Fracture": 0.25,
    "Lung Lesion": 0.28,
    "Lung Opacity": 0.45,
    "No Finding": 0.50,
    "Pleural Effusion": 0.36,
    "Pleural Other": 0.25,
    "Pneumonia": 0.32,
    "Pneumothorax": 0.28,
    "Support Devices": 0.45,
}
