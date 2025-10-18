import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_distances
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report
import matplotlib.pyplot as plt


CACHE_DIR = "grid_search_cache"       # cached descriptors from earlier run
RESULTS_DIR = "results_pca60"
os.makedirs(RESULTS_DIR, exist_ok=True)
SUMMARY_LOG = os.path.join(RESULTS_DIR, "summary_pca60.txt")

N_COMPONENTS_PCA = 60  # best configuration
LEVELS = 3             # fixed best temporal hierarchy

def load_cached_descriptors():
    results = {}
    for fname in tqdm(os.listdir(CACHE_DIR), desc="Loading cached descriptors"):
        if fname.endswith(".npz"):
            key = fname.replace(".npz", "").replace("_", "-")
            path = os.path.join(CACHE_DIR, fname)
            with np.load(path) as data:
                results[key] = data["descriptor"]
    return results


def build_per_gesture_map(results):
    per_gesture = defaultdict(lambda: defaultdict(list))
    for key, desc in results.items():
        parts = key.split('-')
        if len(parts) < 3:
            continue
        subject, gesture, test_folder = parts[0], parts[1], parts[2]
        per_gesture[gesture][subject].append(desc.ravel())
    return per_gesture

def pca_lda_transform(per_gesture_map, n_components_pca=60):
    transformed_map = defaultdict(lambda: defaultdict(list))
    for gesture, subj_map in per_gesture_map.items():
        X, y = [], []
        subjects = list(subj_map.keys())
        for i, subj in enumerate(subjects):
            for desc in subj_map[subj]:
                X.append(desc)
                y.append(i)
        X = np.vstack(X)
        y = np.array(y)

        pca = PCA(n_components=min(n_components_pca, X.shape[1]))
        X_pca = pca.fit_transform(X)
        lda = LDA(n_components=min(len(subjects)-1, X_pca.shape[1]))
        X_lda = lda.fit_transform(X_pca, y)

        idx = 0
        for i, subj in enumerate(subjects):
            n_desc = len(subj_map[subj])
            transformed_map[gesture][subj] = [X_lda[idx+j] for j in range(n_desc)]
            idx += n_desc
    return transformed_map


def compute_eer_and_plot(per_gesture_map, results_dir):
    log_lines = []
    for gesture, subj_map in per_gesture_map.items():
        genuine, impostor = [], []
        subjects = list(subj_map.keys())
        for i, subj_i in enumerate(subjects):
            for j, subj_j in enumerate(subjects):
                for desc_i in subj_map[subj_i]:
                    for desc_j in subj_map[subj_j]:
                        sim = 1 - cosine_distances(desc_i.reshape(1,-1), desc_j.reshape(1,-1))[0,0]
                        if subj_i == subj_j:
                            genuine.append(sim)
                        else:
                            impostor.append(sim)
        if not genuine or not impostor:
            continue

        eer_stats = get_eer_stats(genuine, impostor)
        eer = eer_stats.eer
        log_lines.append(f"Gesture: {gesture} | EER: {eer:.4f}\n")
        tqdm.write(f"[RESULT] Gesture: {gesture:<10} | EER: {eer:.4f}")

        # Histogram plot
        plt.figure(figsize=(6,4))
        plt.hist(genuine, bins=50, alpha=0.6, label='Genuine', density=True)
        plt.hist(impostor, bins=50, alpha=0.6, label='Impostor', density=True)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.legend()
        plt.title(f"{gesture} | PCA={N_COMPONENTS_PCA}, Levels={LEVELS} | EER={eer:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{gesture}_hist.png"), dpi=200)
        plt.close()

        # DET/ROC-style CSV
        try:
            generate_eer_report([eer_stats], ids=[gesture],
                                save_file=os.path.join(results_dir, f"{gesture}_DET.csv"))
        except Exception as e:
            tqdm.write(f"[WARN] Could not generate DET for {gesture}: {e}")

    with open(SUMMARY_LOG, "w") as f:
        f.writelines(log_lines)
    avg_eer = np.mean([float(l.split(":")[-1]) for l in log_lines])
    print(f"\n Average EER = {avg_eer:.4f}")


if __name__ == "__main__":
    print(f"\n Evaluating Best Config: PCA={N_COMPONENTS_PCA}, Levels={LEVELS} \n")
    results = load_cached_descriptors()
    per_gesture = build_per_gesture_map(results)
    per_gesture_transformed = pca_lda_transform(per_gesture, N_COMPONENTS_PCA)
    compute_eer_and_plot(per_gesture_transformed, RESULTS_DIR)
    print(f"\n All outputs saved in '{RESULTS_DIR}/' and summary_pca60.txt\n")
