# Facial Emotion Recognition: MLP vs CNN and the Role of Label Quality

**Course:** Machine Learning, IGP-TUBS | **Supervisor:** Dr.-Ing. Mehdi Maboudi
**Author:** Atharv Patil | M.Sc. Data Science, TU Braunschweig | March 2026

## Results

| Experiment | Test Accuracy |
|---|---|
| FER-2013 MLP (hard labels) | 48.57% |
| FER-2013 CNN (hard labels) | 65.59% |
| FER+ MLP (soft labels)     | 66.90% |
| FER+ CNN (soft labels)     | **82.20%** |

**Key finding:** Label quality gain (+18.3pp) ≈ Architecture gain (+17.0pp).
FER+ MLP (66.9%) outperforms FER-2013 CNN (65.6%) — clean labels on a simpler model
beat noisy labels on a better model.

## Structure
```
notebooks/   — Jupyter notebooks (1 clean cell each, outputs included)
scripts/     — Standalone Python scripts for full reproduction
report/      — LaTeX source + compiled PDF (9 pages)
presentation/ — Final presentation slides (21 slides)
```

## Datasets

- **FER-2013:** `kaggle datasets download -d msambare/fer2013`
- **FER+:** `git clone https://github.com/microsoft/FERPlus.git`
- Raw pixels: `kaggle datasets download -d deadskull7/fer2013`

## Requirements
```
torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow kaggle
```

## How to reproduce

Run each notebook end-to-end on Google Colab with a T4 GPU.
Each notebook is self-contained (downloads data, trains, evaluates, saves plots).
Expected runtimes: MLP ~37 min | CNN ~35 min | FER+ ~60 min total.
EOF
