import os
import cv2
import numpy as np
from tqdm import tqdm
from silhouette_tunnel import silhouette_tunnel
from hierarchy import temporal_hierar_cov
from hand_morphology1 import multichannel_descriptors
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict
import matplotlib.pyplot as plt
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report
from itertools import chain
from file_loader import load_depth_frames
# -----------------------
# CONFIG
# -----------------------
DATASET_PATH = "./ell715_assg4"
SUBJECTS = [f"{i:02d}" for i in chain(range(1,11), range(16,22))] # 01-10 and 16-21
GESTURES = ["Compass", "Piano", "Push", "UCDO"]
GALLERY_SUBJECTS = [f"{i:02d}" for i in range(1, 9)]   # 01–08
PROBE_SUBJECTS = [f"{i:02d}" for i in chain(range(9, 11), range(16,22))]    # 09–21
RESULTS_DIR = "results"
DESC_CACHE_DIR = os.path.join(RESULTS_DIR, "descriptors")
os.makedirs(DESC_CACHE_DIR, exist_ok=True)
SUMMARY_LOG = os.path.join(RESULTS_DIR, "summary.txt")

MAX_FRAMES = None
THRESHOLD = 50
FRAME_SUBSAMPLE = 1
SUB_TUNNEL_K = 3  # number of sub-silhouette bins used by multichannel_descriptors







# -----------------------
# Caching helpers
# -----------------------
def _cache_dir_for(subject, gesture, test_folder=None):
    if test_folder:
        d = os.path.join(DESC_CACHE_DIR, subject, gesture, test_folder)
    else:
        d = os.path.join(DESC_CACHE_DIR, subject, gesture)

    os.makedirs(d, exist_ok=True)
    return d



def _load_npz_safe(path, key=None):
    # helper returning data[key] if key given, else returns the whole archive mapping
    with np.load(path, allow_pickle=True) as data:
        if key is None:
            # return mapping-like object; convert to dict to avoid closed file reference
            return {k: data[k] for k in data.files}
        return data[key].copy()




# -----------------------
# Descriptor Computation with Caching
# -----------------------
def compute_descriptors_variant(depth_frames, background, variant, subject, gesture, test_folder, K=SUB_TUNNEL_K):
    """
    Computes gesture descriptors (Baseline, TemporalHierarchy, AdditionalTunnels, FirstFrame)
    and caches them. Also saves silhouette masks as PNGs for visualization.
    """

    if depth_frames.size == 0:
        return None

    desc_dir = _cache_dir_for(subject, gesture, test_folder)

    # ----- 1) Check if final descriptor is already cached -----
    final_desc_path = os.path.join(desc_dir, f"descriptor_{variant}.npz")
    if os.path.exists(final_desc_path):
        # tqdm.write(f"[CACHE] Found cached final descriptor for {variant}, Subject-{subject}, Gesture-{gesture}")
        with np.load(final_desc_path, allow_pickle=True) as data:
            return data["descriptor"]




    # ----- 2) Load or compute per-frame base descriptors -----
    frame_desc_path = os.path.join(desc_dir, "frame_descriptors.npz")
    frame_descriptors, masks = None, None

    if variant in ["Baseline", "TemporalHierarchy", "FirstFrame"]:
        if os.path.exists(frame_desc_path):
            # tqdm.write(f"[CACHE] Found per-frame descriptors for {subject}-{gesture}-{test_folder}")
            frame_descriptors = _load_npz_safe(frame_desc_path, "frame_descriptors")
        else:
            # tqdm.write(f"[EXTRACT] Computing per-frame descriptors (silhouette_tunnel) for {subject}-{gesture}-{test_folder}")
            frame_descriptors, masks = silhouette_tunnel(depth_frames, background, threshold=THRESHOLD)

            tqdm.write(f"[DEBUG] {subject}-{gesture}-{test_folder}: silhouette mean={np.mean(frame_descriptors):.4f}, "
                       f"std={np.std(frame_descriptors):.6f}, shape={frame_descriptors.shape}")

            np.savez_compressed(frame_desc_path, frame_descriptors=frame_descriptors)
            # tqdm.write(f"[SAVED] Cached per-frame descriptors for {subject}-{gesture}-{test_folder}")

            # 💾 Save silhouette visualizations
            # if masks is not None:
            #     sil_dir = os.path.join("results", "silhouettes", subject, gesture, test_folder)
            #     os.makedirs(sil_dir, exist_ok=True)
            #     for i in range(masks.shape[0]):
            #         mask_img = masks[i].astype(np.uint8) * 255
            #         cv2.imwrite(os.path.join(sil_dir, f"silhouette_{i:03d}.png"), mask_img)
            #     tqdm.write(f"[VISUAL] Saved {masks.shape[0]} silhouette masks for {subject}-{gesture}-{test_folder}")


    # ----- 3) Morphology descriptors (AdditionalTunnels) -----
    morph_desc_path = os.path.join(desc_dir, f"morphology_descriptors_K{K}.npz")
    morph_descriptors = None
    if variant == "AdditionalTunnels":
        if os.path.exists(morph_desc_path):
            tqdm.write(f"[CACHE] Found morphology descriptors K={K} for {subject}-{gesture}-{test_folder}")
            morph_descriptors = _load_npz_safe(morph_desc_path, "morphology_descriptors")
        else:
            tqdm.write(f"[EXTRACT] Computing morphology descriptors K={K} for {subject}-{gesture}-{test_folder}")
            morph_descriptors = multichannel_descriptors(subject, gesture, test_folder, depth_frames, background, desc_full=frame_descriptors, K=K)
            np.savez_compressed(morph_desc_path, morphology_descriptors=morph_descriptors)
            tqdm.write(f"[SAVED] Cached morphology descriptors for {subject}-{gesture}-{test_folder}")
            
            
    # ----- 4) Compute the final aggregated descriptor -----
    # tqdm.write(f"[INFO] Computing new descriptor for {variant}, {subject}-{gesture}-{test_folder}")
    descriptor = None

    if variant == "Baseline":
        if frame_descriptors.shape[0] != 14:
            tqdm.write(f"[WARN] Transposing frame_descriptors from {frame_descriptors.shape}")
            frame_descriptors = frame_descriptors.T

        mu = np.mean(frame_descriptors, axis=1, keepdims=True)
        F_centered = frame_descriptors - mu
        C = (F_centered @ F_centered.T) / frame_descriptors.shape[1]
        descriptor = C[np.triu_indices_from(C)].astype(np.float32)

        tqdm.write(f"[DEBUG] Covariance descriptor mean={np.mean(descriptor):.6f}, "
                   f"std={np.std(descriptor):.6f}, shape={descriptor.shape}")

    elif variant == "TemporalHierarchy":
        descriptor = np.asarray(temporal_hierar_cov(frame_descriptors), dtype=np.float32)

    elif variant == "AdditionalTunnels":
        descs_arr = [np.asarray(d, dtype=np.float32).ravel() for d in morph_descriptors]
        descriptor = np.concatenate(descs_arr).astype(np.float32)

    elif variant == "FirstFrame":
        first_frame = depth_frames[:1]
        bg_first = first_frame[0]
        F_first, mask_first = silhouette_tunnel(first_frame, bg_first)
        cov = np.cov(F_first)
        descriptor = cov[np.triu_indices_from(cov)].astype(np.float32)

        sil_dir = os.path.join("results", "silhouettes", subject, gesture, test_folder)
        os.makedirs(sil_dir, exist_ok=True)
        mask_img = mask_first[0].astype(np.uint8) * 255
        cv2.imwrite(os.path.join(sil_dir, "silhouette_first_frame.png"), mask_img)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    # 5️⃣ Cache final descriptor
    np.savez_compressed(final_desc_path, descriptor=descriptor)
    # tqdm.write(f"[SAVED] Cached final descriptor for {variant}, {subject}-{gesture}-{test_folder}")

    return descriptor





# -----------------------
# Ablation Routine
# -----------------------
def run_full_ablation():
    variants = ["Baseline", "TemporalHierarchy", "AdditionalTunnels"]  # extend later
    results = {v: {} for v in variants}

    for variant in tqdm(variants, desc="Running Variants"):
        for subject in SUBJECTS:
            for gesture in GESTURES:
                gesture_path = os.path.join(DATASET_PATH, subject, gesture)
                test_folders = sorted([f for f in os.listdir(gesture_path) if f.startswith("Test")])

                for test_folder in test_folders:
                    try:
                        frames, background = load_depth_frames(subject, gesture, test_folder, MAX_FRAMES)
                        tqdm.write(f"[DEBUG] {subject}-{gesture}-{test_folder}: frames={frames.shape}")
                        desc = compute_descriptors_variant(frames, background, variant,
                                                           subject, gesture, test_folder)
                        results[variant][f"{subject}-{gesture}-{test_folder}"] = desc
                    except Exception as e:
                        tqdm.write(f"[WARN] {subject}-{gesture}-{test_folder}: {e}")
                        continue

        # Save intermediate results
        variant_results = {k: v for k, v in results[variant].items() if v is not None}
        np.savez_compressed(os.path.join(RESULTS_DIR, f"ablation_results_{variant}.npz"), **variant_results)
        tqdm.write(f"[INFO] Saved partial results for {variant}")

    return results


# -----------------------
# Evaluation with PyEER
# -----------------------
def build_per_gesture_map(results, variant):
    per_gesture = defaultdict(lambda: defaultdict(list))
    for key, desc in results[variant].items():
        if desc is None:
            continue
        parts = key.split('-')
        if len(parts) < 3:
            continue
        subject, gesture, test_folder = parts[0], parts[1], parts[2]
        per_gesture[gesture][subject].append(np.asarray(desc).ravel())
    return per_gesture


def compute_eer_and_plot(per_gesture_map, variant):
    variant_dir = os.path.join(RESULTS_DIR, variant)
    os.makedirs(variant_dir, exist_ok=True)
    log_lines = []

    for gesture, subj_map in per_gesture_map.items():
        genuine_scores, impostor_scores = [], []

        subjects = list(subj_map.keys())
        for i, subj_i in enumerate(subjects):
            for j, subj_j in enumerate(subjects):
                for desc_i in subj_map[subj_i]:
                    for desc_j in subj_map[subj_j]:
                        sim = 1 - cosine_distances(desc_i.reshape(1, -1), desc_j.reshape(1, -1))[0, 0]
                        if subj_i == subj_j:
                            genuine_scores.append(sim)
                        else:
                            impostor_scores.append(sim)

        if not genuine_scores or not impostor_scores:
            tqdm.write(f"[WARN] Skipping {gesture}: insufficient scores.")
            continue

        eer_stats = get_eer_stats(genuine_scores, impostor_scores)
        eer = eer_stats.eer
        log_lines.append(f"Gesture: {gesture} | Variant: {variant} | EER: {eer:.4f}\n")
        tqdm.write(f"[RESULT] Gesture: {gesture} | Variant: {variant} | EER: {eer:.4f}")

        # Save DET & Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(genuine_scores, bins=50, alpha=0.6, label='Genuine', density=True)
        plt.hist(impostor_scores, bins=50, alpha=0.6, label='Impostor', density=True)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.legend()
        plt.title(f"{gesture} ({variant}) EER={eer:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(variant_dir, f"{gesture}_hist.png"), dpi=200)
        plt.close()

        try:
            generate_eer_report([eer_stats], ids=[gesture], save_file=os.path.join(variant_dir, f"{gesture}_DET.csv"))
        except Exception as e:
            tqdm.write(f"[WARN] Could not generate DET for {gesture}: {e}")

    with open(SUMMARY_LOG, "a") as f:
        f.writelines(log_lines)


# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    results = run_full_ablation()
    # variant = "Baseline"
    for variant in ["Baseline", "TemporalHierarchy", "AdditionalTunnels"]:
        gesture_map = build_per_gesture_map(results, variant)
        compute_eer_and_plot(gesture_map, variant)

    tqdm.write(f"\nAll results stored in '{RESULTS_DIR}' directory.\n")
    
    
