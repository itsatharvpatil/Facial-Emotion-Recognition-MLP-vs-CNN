# Facial Emotion Recognition: MLP vs CNN and the Role of Label Quality

**Course:** Machine Learning, IGP-TUBS  
**Supervisor:** Dr.-Ing. Mehdi Maboudi  
**Author:** Atharv Patil | M.Sc. Data Science, Technische Universität Braunschweig  
**Date:** March 2026

## Results

| Experiment                          | Test Accuracy |
|-------------------------------------|---------------|
| FER-2013 MLP (hard labels)          | 48.57%        |
| FER-2013 CNN (hard labels)          | 65.59%        |
| FER+ MLP (soft labels)              | 66.90%        |
| FER+ CNN (soft labels)              | **82.20%**    |

**Key finding:**  
Label-quality gain (+18.3 pp on MLP) ≈ Architecture gain (+17.0 pp).  
**FER+ MLP (66.90 %) outperforms FER-2013 CNN (65.59 %)** — clean labels on a simpler model beat noisy labels on a better model.

## Repository Structure

```
notebooks/   — Jupyter notebooks (1 clean cell each, outputs included)
report/      — LaTeX source + compiled PDF (9 pages)
presentation/ — Final presentation slides (21 slides)
```

## Datasets

- **FER-2013** (images + hard labels)  
  `kaggle datasets download -d msambare/fer2013`

- **FER+** (soft labels from 10 annotators)  
  `git clone https://github.com/microsoft/FERPlus.git`

- Raw pixels (alternative download)  
  `kaggle datasets download -d deadskull7/fer2013`

## Requirements

```bash
torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow kaggle

##How to Reproduce

Clone the repository.
Run each notebook end-to-end on Google Colab (T4 GPU recommended).
Every notebook is completely self-contained:
Downloads data automatically
Trains the model
Evaluates on validation/test sets
Saves all plots and results


Expected runtimes (T4 GPU):

MLP experiments ≈ 37 min
CNN experiments ≈ 35 min
Full FER+ experiments ≈ 60 min total

All results are fully reproducible with seed=42 where applicable.
