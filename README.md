# ELL715: Hand Gesture-Based User Authentication 

### LEVERAGING SHAPE AND DEPTH IN USER AUTHENTICATION FROM IN-AIR HAND GESTURES  
---

📘 **GitHub Repository:** [https://github.com/shahidkhan-ai/assignment](https://github.com/assignment)

This repository implements a **gesture-based user authentication system** using depth data.  
Each gesture sequence is represented as a *temporal silhouette tunnel*, from which 14-dimensional pixel-level features are extracted and converted into compact **covariance descriptors**.  
The project includes baseline, temporal-hierarchical, and morphology-enhanced variants, and evaluates them using EER-based authentication metrics.  

---

## 📁 Repository Structure

```bash
(image_proc_env) shahidkhan@Shahid-8 assignment % tree -L 2.
├── README.md
├── __pycache__/
│   ├── *.pyc
├── analysis.py                     # Main evaluation and ablation experiments
├── covariance_matrix.py            # Covariance descriptor computation
├── environment.yml                 # Conda environment file
├── file_loader.py                  # Data loader
├── hand_morphology1.py             # Sub-silhouette (multi-channel) descriptor extraction
├── hierarchy.py                    # Temporal hierarchical covariance descriptors
├── logs.txt                        # Run-time logs and debug info
├── report/
│   ├── main.tex, preamble.tex, format.tex, references.bib
│   ├── main.pdf                    # Final LaTeX report output
│   └── problems/, out/             # Aux and intermediate LaTeX files
├── results/
│   ├── Baseline/, TemporalHierarchy/, AdditionalTunnels/
│   ├── descriptors/, features/, silhouettes/, sub_silhouettes/
│   ├── summary.txt                 # Summary of all EER results
├── results_prev/                   # Archived results from previous runs
├── ell715_assg4/                   # Dataset root (depth frames per subject), the dataset needs to be placed here
│   ├── 01/, 02/, ..., 21/          # Each folder = subject
├── silex.py, vis_msb.py, visualize_sillhouetes.py  # Visualization scripts
├── silhouette_tunnel.py            # 14D feature extraction from silhouettes
├── demo4.py                        # PCA + LDA parameter tuning and analysis
├── demo5.py                        # Final PCA + LDA evaluation script
└── results_pca60/                  # PCA + LDA improvement results (see below)
```


---

## ⚙️ Creating the Environment

To set up the project, run the following command in your terminal:

```bash
conda env create -f environment.yml
```

---

## 🧭 QUICK COMMANDS

| Action | Command |
| ------- | -------- |
| **Activate Environment** | `conda activate image_proc_env` |
| **Deactivate Environment** | `conda deactivate` |
| **List Environments** | `conda env list` |
| **Remove Environment** | `conda env remove -n image_proc_env` |

**Put the dataset file in the same repository** which can be found at [dataset link](https://csciitd-my.sharepoint.com/personal/eez227536_iitd_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Feez227536%5Fiitd%5Fac%5Fin%2FDocuments&ga=1).  
Unzip it and place it in the same directory where you run the code, or modify the path in `sanity_check_v2.py`.

---

## 🚀 How to Run the Experiments

### 1️⃣ Baseline, Temporal, and Morphology Variants

Run the ablation experiments (this computes all gesture descriptors and saves results):

```bash
python analysis.py
```

This will:

- Extract silhouette-based features for each gesture  
- Compute covariance descriptors per variant  
- Evaluate EER using cosine similarity  
- Generate histograms, DET curves, and summary logs in `results/variant`

It creates an `eer_matrix.csv` file inside `results/variant/` showing **inter-gesture vs intra-gesture** authentication performance.  
It also generates:

```
results/<Variant>/multi_gesture_hist.png
results/<Variant>/multi_gesture_eer.txt
```
The folder `results_pca60/` holds the results for the **PCA + LDA solution**, which is part of the improvement strategy.  

To reproduce these results, run:

```bash
python demo4.py
python demo5.py
```

`demo4.py` runs the code to get the **best parameters for PCA + LDA analysis**,  
while `demo5.py` generates the **final authentication outputs** and summary results.
---

| Module | Description |
| ------- | ------------ |
| `silhouette_tunnel.py` | Extracts pixel-level 14D features (x, y, t, z, 8 directions, 2 temporal) |
| `hierarchy.py` | Computes 735-D temporal hierarchical covariance descriptors |
| `hand_morphology1.py` | Computes sub-silhouette (multi-channel) features and descriptors |
| `covariance_matrix.py` | Builds upper-triangular covariance descriptors from features |
| `analysis.py` | Main evaluation: computes EER, histograms, DETs, and multi-gesture results |
| `visualize_sillhouetes.py` | Renders silhouette masks and sub-tunnel visualizations |
| `data_reading.py` | Loads and preprocesses dataset sequences and background frames |

---

### 📊 Output Structure

```
results/
├── Baseline/
│   ├── Compass_hist.png
│   ├── Piano_hist.png
│   ├── multi_gesture_hist.png
│   ├── multi_gesture_eer.txt
│   └── *.csv  (DET curves)
├── TemporalHierarchy/
├── AdditionalTunnels/
├── results_pca60/
│   ├── compass_hist.png
│   ├── piano_hist.png
│   ├── push_hist.png
│   ├── UCDO_hist.png
│   └── summary_pca60.txt
├── ablation_results_*.npz
├── eer_matrix.csv
└── summary.txt
```

```
results_pca60/
├── compass_hist.png
├── piano_hist.png
├── push_hist.png
├── UCDO_hist.png
└── summary_pca60.txt
```

Note:
- The repository includes these files from our test runs. If you clone and rerun, results will be overridden.  
- All descriptors are stored in the `results/` directory, which can become large — clean periodically if not needed.
