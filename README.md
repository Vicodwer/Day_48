# Week 08 · Friday — Transfer Learning + ME1 Preparation

PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar

## What this notebook covers

Sub-steps 1 and 2 (Easy / Required): Loading and characterising the medical imaging metadata,
followed by feature extraction using a frozen ResNet-50 backbone.

Sub-steps 3, 4, and 5 (Medium / Required): Fine-tuning with partial unfreezing, Grad-CAM
visualisations for the two most clinically critical classes, ME1 personal synthesis on BPTT,
and classification of the 30 unlabeled images with confidence scores.

Sub-steps 6 and 7 (Hard / Optional): Three-strategy comparison (feature extraction, fine-tuning,
training from scratch) under identical evaluation conditions, and a three-tier triage protocol
for the unlabeled images with calibration analysis.

## How to run

1. Clone or download this repository. Make sure the folder structure is:

```
week-08/
  friday/
    w8_friday_transfer_learning.ipynb
    medical_imaging_meta.csv       <- download from LMS
    images/                        <- place all chest X-ray image files here
    README.md
    prompts.md
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Launch Jupyter and open the notebook:

```bash
jupyter notebook w8_friday_transfer_learning.ipynb
```

4. Run all cells in order from top to bottom. The notebook will create a `plots/` subfolder
automatically and save all figures there. Two CSV files (`unlabeled_predictions.csv` and
`triage_results.csv`) are written to the same directory.

## Python version

Python 3.10 or higher is required. The type hint syntax used in function signatures
(`tuple[float, float]`, `list`) relies on built-in generics available from 3.10 onwards.

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
grad-cam>=1.4.6
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=9.5.0
jupyter>=1.0.0
```

Install everything at once:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install grad-cam scikit-learn pandas numpy matplotlib seaborn Pillow jupyter
```

If you have a CUDA-capable GPU, replace the CPU wheel URL with the appropriate CUDA version
from pytorch.org. The notebook automatically selects `cuda` if a GPU is available.

## Dataset notes

`medical_imaging_meta.csv` must be downloaded from the LMS before running the notebook.
The images themselves should be placed in the `images/` subfolder using the filenames
as they appear in the `filename` column of the CSV. The dataset contains 520 labeled rows
across five conditions and 30 rows with no condition label (the unlabeled set for Sub-step 5).

Reference sources:
- Chest X-Ray Images (Pneumonia): kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- NIH Chest X-rays: kaggle.com/datasets/nih-chest-xrays/data

## Output files

After a full run the following files will exist alongside the notebook:

```
plots/label_distribution.png
plots/cm_feature_extraction.png
plots/cm_fine_tuning.png
plots/cm_from_scratch.png
plots/training_curves.png
plots/three_strategy_comparison.png
plots/gradcam_*.png                (one per Grad-CAM sample)
plots/triage_protocol.png
unlabeled_predictions.csv
triage_results.csv
best_feature_extraction.pth
best_fine_tuning.pth
best_from_scratch.pth
```

The `.pth` checkpoint files are excluded from version control via `.gitignore`.

## Commit history

Commits follow the task progression:

1. `add skeleton notebook with imports and seeded random state`
2. `sub-step 1: dataset characterisation, label distribution, subgroup analysis`
3. `sub-step 2: feature extraction with frozen ResNet-50, per-class evaluation`
4. `sub-step 3: fine-tuning with partial unfreeze, training curve comparison`
5. `sub-step 4: grad-cam visualisations for critical classes`
6. `sub-step 5: ME1 synthesis, unlabeled classification with confidence scores`
7. `sub-steps 6-7: from-scratch baseline, three-strategy summary, triage protocol`
