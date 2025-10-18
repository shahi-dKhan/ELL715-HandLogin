import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from itertools import product
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_distances
from pyeer.eer_info import get_eer_stats
from silhouette_tunnel import silhouette_tunnel
from file_loader import load_depth_frames

CACHE_DIR = "grid_search_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cov_ut_fast_shrink(F, alpha=None):
    D, N = F.shape
    if N <= 1:
        return np.zeros((D*(D+1)//2,), dtype=np.float32)
    F = F - np.mean(F, axis=1, keepdims=True)
    S = (F @ F.T) / (N - 1)
    if alpha is None:
        alpha = min(0.5, D / (N + D))
    mu = np.trace(S) / D
    S_shrunk = (1 - alpha) * S + alpha * mu * np.eye(D)
    return S_shrunk[np.triu_indices_from(S_shrunk)].astype(np.float32)

def temporal_hierar_cov_fast(F, levels=3):
    D, N = F.shape
    descriptors = []
    for level in range(levels):
        segments = 2 ** level
        seg_len = N // segments
        for s in range(segments):
            start = s * seg_len
            end = (s + 1) * seg_len if s < segments - 1 else N
            subF = F[:, start:end]
            desc = cov_ut_fast_shrink(subF)
            descriptors.append(desc)
    return np.concatenate(descriptors, axis=0)


def compute_descriptor_cached(subject, gesture, test_folder, depth_frames, background):
    cache_file = os.path.join(CACHE_DIR, f"{subject}_{gesture}_{test_folder}.npz")
    if os.path.exists(cache_file):
        with np.load(cache_file) as data:
            return data["descriptor"]
    desc = temporal_hierar_cov_fast(silhouette_tunnel(depth_frames, background)[0])
    np.savez_compressed(cache_file, descriptor=desc)
    return desc


def build_per_gesture_map(results):
    per_gesture = defaultdict(lambda: defaultdict(list))
    for key, desc in results.items():
        subject, gesture, test_folder = key.split('-')
        per_gesture[gesture][subject].append(desc.ravel())
    return per_gesture


def pca_lda_transform(per_gesture_map, n_components_pca=50):
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


def compute_eer(per_gesture_map):
    eer_dict = {}
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
        if genuine and impostor:
            eer_stats = get_eer_stats(genuine, impostor)
            eer_dict[gesture] = eer_stats.eer
    return eer_dict


def grid_search(SUBJECTS, GESTURES, DATASET_PATH, pca_values, level_values):
    best_config = None
    best_avg_eer = 1.0
    log = []

    
    results = {}
    for subject in tqdm(SUBJECTS, desc="Caching descriptors"):
        for gesture in GESTURES:
            gesture_path = os.path.join(DATASET_PATH, subject, gesture)
            test_folders = sorted([f for f in os.listdir(gesture_path) if f.startswith("Test")])
            for test_folder in test_folders:
                frames, background = load_depth_frames(subject, gesture, test_folder)
                desc = compute_descriptor_cached(subject, gesture, test_folder, frames, background)
                results[f"{subject}-{gesture}-{test_folder}"] = desc

    per_gesture = build_per_gesture_map(results)

    
    for n_components_pca, levels in product(pca_values, level_values):
        per_gesture_transformed = pca_lda_transform(per_gesture, n_components_pca)
        eer_dict = compute_eer(per_gesture_transformed)
        avg_eer = np.mean(list(eer_dict.values()))
        log.append((n_components_pca, levels, avg_eer))
        print(f"PCA={n_components_pca}, Levels={levels} => Avg EER={avg_eer:.4f}")
        if avg_eer < best_avg_eer:
            best_avg_eer = avg_eer
            best_config = (n_components_pca, levels)

    print(f"\n Best Config: PCA={best_config[0]}, Levels={best_config[1]} => Avg EER={best_avg_eer:.4f}")
    np.savez_compressed("grid_search_results_cached.npz", log=log, best_config=best_config)
    return best_config, log


if __name__ == "__main__":
    DATASET_PATH = "./ELL715_assg4"
    SUBJECTS = [f"{i:02d}" for i in list(range(1,11))+list(range(16,22))]
    GESTURES = ["Compass","Piano","Push","UCDO"]

    pca_values = [20, 30, 40, 50, 60]
    level_values = [2,3,4]

    best_config, log = grid_search(SUBJECTS, GESTURES, DATASET_PATH, pca_values, level_values)
