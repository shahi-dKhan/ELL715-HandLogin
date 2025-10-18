# ELL715: Hand Gesture-Based User Authentication 🖐️
### LEVERAGING SHAPE AND DEPTH IN USER AUTHENTICATION FROM IN-AIR HAND GESTURES

---
📘 **GitHub Repository:** [https://github.com/shahidkhan-ai/hand-gesture-authentication](https://github.com/shahidkhan-ai/hand-gesture-authentication)

# Hand Gesture Biometric Authentication using Silhouette Descriptors

📘 **GitHub Repository:** [https://github.com/shahidkhan-ai/hand-gesture-authentication](https://github.com/shahidkhan-ai/hand-gesture-authentication)

This repository implements a **gesture-based user authentication system** using depth data.
Each gesture sequence is represented as a *temporal silhouette tunnel*, from which 14-dimensional
pixel-level features are extracted and converted into compact **covariance descriptors**.
The project includes baseline, temporal-hierarchical, and morphology-enhanced variants,
and evaluates them using EER-based authentication metrics.

---

## 📁 Repository Structure

```bash
(image_proc_env) shahidkhan@Shahid-8 assignment % tree -L 2
.
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



## ⚙️ Creating the Environment

To set up the project, run the following command in your terminal:

```bash
conda env create -f environment.yml
```

---

##  QUICK COMMANDS

| Action                 | Command                                |
| ---------------------- | -------------------------------------- |
| **Activate Environment** | `conda activate image_proc_env`        |
| **Deactivate Environment** | `conda deactivate`                     |
| **List Environments** | `conda env list`                       |
| **Remove Environment** | `conda env remove -n image_proc_env`   |

**put the dataset file in the same repository which can be found at [dataset](https://csciitd-my.sharepoint.com/personal/eez227536_iitd_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Feez227536%5Fiitd%5Fac%5Fin%2FDocuments&ga=1). Unzip it and place it in the same directory where you run the code. Or modify the path in sanity_check_v2.py**


How to Run the Experiments
1️⃣ Baseline, Temporal, and Morphology Variants

Run the ablation experiments (this computes all gesture descriptors and saves results):

python analysis.py


This will:

Extract silhouette-based features for each gesture

Compute covariance descriptors per variant

Evaluate EER using cosine similarity

Generate histograms, DET curves, and summary logs in results/variant

creates a eer_matrix.csv file inside results/variant/ showing inter-gesture vs intra-gesture authentication performance.

creates results/<Variant>/multi_gesture_hist.png → histogram of genuine/impostor scores and results/<Variant>/multi_gesture_eer.txt → textual summary of EER

| Module                     | Description                                                                |
| -------------------------- | -------------------------------------------------------------------------- |
| `silhouette_tunnel.py`     | Extracts pixel-level 14D features (x, y, t, z, 8 directions, 2 temporal)   |
| `hierarchy.py`             | Computes 735-D temporal hierarchical covariance descriptors                |
| `hand_morphology1.py`      | Computes sub-silhouette (multi-channel) features and descriptors           |
| `covariance_matrix.py`     | Builds upper-triangular covariance descriptors from features               |
| `analysis.py`              | Main evaluation: computes EER, histograms, DETs, and multi-gesture results |
| `visualize_sillhouetes.py` | Renders silhouette masks and sub-tunnel visualizations                     |
| `data_reading.py`          | Loads and preprocesses dataset sequences and background frames             |



output structure 
results/
├── Baseline/
│   ├── Compass_hist.png
│   ├── Piano_hist.png
│   ├── multi_gesture_hist.png
│   ├── multi_gesture_eer.txt
│   └── *.csv  (DET curves)
├── TemporalHierarchy/
├── AdditionalTunnels/
├── ablation_results_*.npz
├── eer_matrix.csv
└── summary.txt

Note that our repository will include these files, from the test we have run on our system. If you clone the repository, and run the experiments on your own, these results will be overridden.

Also, note that this will store all the descriptors in the results directory, which will be too large in size. So, keep cleaning them unless you have to perform the same experiment again.
