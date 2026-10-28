# UC-EEG-FM LaTeX Manuscript Package (Nature Biomedical Engineering)

This directory contains the complete, publication-grade LaTeX submission package for the manuscript:

> **A Self-Supervised Cross-Disorder EEG Foundation Model Resolves Diagnostic Collapse and Enables Few-Shot Clinical Transfer**  
> *Target Venue:* **Nature Biomedical Engineering** (Article Format)  
> *Authors:* Vishal & Senior Research Team in Computational Neuroengineering & Clinical AI  
> *Repository:* [https://github.com/Vishal442005/UC-EEG-FM](https://github.com/Vishal442005/UC-EEG-FM)  
> *Zenodo DOI:* [10.5281/zenodo.14028374](https://doi.org/10.5281/zenodo.14028374)

---

## Directory Contents

- `main.tex`: Full article manuscript with title, abstract, introduction, results, discussion, methods, tables, and figures.
- `supplementary.tex`: Supplementary Information document (detailed demographic tables, mathematical derivations, and hyperparameter tables).
- `references.bib`: Complete BibTeX bibliography containing all 30 cited references [1–30].
- `figures/`: Subfolder containing all 9 publication-grade figures at 300 DPI:
  - `Figure_1_Training_Convergence.png`
  - `Figure_2_Cross_Disorder_Latent_Manifold.png`
  - `Figure_3_Clinical_ROC_Curves.png`
  - `Figure_4_Confusion_Matrices.png`
  - `Figure_5_Few_Shot_Sample_Efficiency.png`
  - `Figure_6_Feature_Saliency_and_Neurobiology.png`
  - `Figure_7_Topographical_Scalp_Activation_Maps.png`
  - `Figure_8_Clinical_Decision_Support_System.png`
  - `Figure_9_SOTA_Literature_Benchmark_Sweep.png`

---

## How to Compile

### 1. Compile in Overleaf (Recommended)
1. Zip the entire `latex_paper/` folder (or use `uc_eeg_fm_latex_paper.zip`).
2. Go to [Overleaf.com](https://www.overleaf.com/) and click **New Project** $\to$ **Upload Project**.
3. Drag-and-drop `uc_eeg_fm_latex_paper.zip`.
4. Click **Recompile** with pdfLaTeX.

### 2. Compile Locally via TeX Live / MiKTeX
Run the following terminal commands:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
For Supplementary Information:
```bash
pdflatex supplementary.tex
pdflatex supplementary.tex
```
