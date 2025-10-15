import os
import cv2
import numpy as np
from tqdm import tqdm
from sillhouette_tunnel import silhouette_tunnel
from hierarchy import temporal_hierar_cov
from hand_morphology1 import multichannel_descriptors
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict
import matplotlib.pyplot as plt
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report

# -----------------------
# CONFIG
# -----------------------
DATASET_PATH = "./ell715_assg4"
SUBJECTS = [f"{i:02d}" for i in range(1, 17)]  # 16 subjects
GESTURES = ["Compass", "Piano", "Push", "UCDO"]
GALLERY_SUBJECTS = [f"{i:02d}" for i in range(1, 9)]   # 01–08
PROBE_SUBJECTS = [f"{i:02d}" for i in range(9, 17)]    # 09–16
RESULTS_DIR = "results"
DESC_CACHE_DIR = os.path.join(RESULTS_DIR, "descriptors")
os.makedirs(DESC_CACHE_DIR, exist_ok=True)
SUMMARY_LOG = os.path.join(RESULTS_DIR, "summary.txt")

MAX_FRAMES = None
THRESHOLD = 20
FRAME_SUBSAMPLE = 1
SUB_TUNNEL_K = 3  # number of sub-silhouette bins used by multichannel_descriptors


# -----------------------
# Depth Frame Loader
# -----------------------
def load_depth_frames(subject, gesture, max_frames=None):
    gesture_path = os.path.join(DATASET_PATH, subject, gesture)
    if not os.path.isdir(gesture_path):
        return np.array([], dtype=np.float32)
    test_folders = [f for f in sorted(os.listdir(gesture_path)) if f.startswith("Test")]
    frames = []

    for test_folder in test_folders:
        lsb_path = os.path.join(gesture_path, test_folder, "LSB")
        msb_path = os.path.join(gesture_path, test_folder, "MSB")
        if not (os.path.exists(lsb_path) and os.path.exists(msb_path)):
            continue
        lsb_files = sorted([f for f in os.listdir(lsb_path) if f.endswith(".png")])
        msb_files = sorted([f for f in os.listdir(msb_path) if f.endswith(".png")])
        n = min(len(lsb_files), len(msb_files))
        if n == 0:
            continue

        for i in range(1, n + 1):
            lsb_file = os.path.join(lsb_path, f"LSB{i}.png")
            msb_file = os.path.join(msb_path, f"MSB{i}.png")
            if not os.path.exists(lsb_file) or not os.path.exists(msb_file):
                continue
            lsb = cv2.imread(lsb_file, cv2.IMREAD_UNCHANGED)
            msb = cv2.imread(msb_file, cv2.IMREAD_UNCHANGED)
            if lsb is None or msb is None:
                continue
            if lsb.ndim == 3:
                lsb = lsb[:, :, 0]
            if msb.ndim == 3:
                msb = msb[:, :, 0]
            depth = ((msb.astype(np.uint16) << 8) | lsb.astype(np.uint16)).astype(np.float32)
            frames.append(depth)
            if max_frames and len(frames) >= max_frames:
                break

        if max_frames and len(frames) >= max_frames:
            break

    if len(frames) == 0:
        return np.array([], dtype=np.float32)
    frames = np.stack(frames, axis=0).astype(np.float32)
    if FRAME_SUBSAMPLE > 1:
        frames = frames[::FRAME_SUBSAMPLE]
    return frames


def estimate_background(depth_frames, num_bg_frames=10):
    if depth_frames.size == 0:
        return None
    num_bg = min(num_bg_frames, depth_frames.shape[0])
    return np.median(depth_frames[:num_bg], axis=0).astype(np.float32)


# -----------------------
# Caching helpers
# -----------------------
def _cache_dir_for(subject, gesture):
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
def compute_descriptors_variant(depth_frames, background, variant, subject, gesture, K=SUB_TUNNEL_K):
    """
    Compute descriptor for given variant.
    
    Caching strategy:
    1. Per-frame base descriptors (from silhouette_tunnel) - frame_descriptors.npz
    2. Per-frame morphology descriptors (from multichannel_descriptors) - morphology_descriptors_K{K}.npz
    3. Final aggregated descriptor per variant - descriptor_{variant}.npz
    """
    if depth_frames.size == 0:
        return None

    desc_dir = _cache_dir_for(subject, gesture)
    
    # ----- 1) Check if final descriptor is already cached -----
    final_desc_path = os.path.join(desc_dir, f"descriptor_{variant}.npz")
    if os.path.exists(final_desc_path):
        try:
            descriptor = _load_npz_safe(final_desc_path, "descriptor")
            tqdm.write(f"[CACHE] Loaded final descriptor for {variant}, Subject-{subject}, Gesture-{gesture}")
            return descriptor
        except Exception as e:
            tqdm.write(f"[WARN] Failed loading final descriptor cache ({final_desc_path}): {e}. Will recompute.")
            try:
                os.remove(final_desc_path)
            except Exception:
                pass

    # ----- 2) Load or compute per-frame base descriptors (for Baseline, TemporalHierarchy, FirstFrame) -----
    frame_desc_path = os.path.join(desc_dir, "frame_descriptors.npz")
    frame_descriptors = None
    
    if variant in ["Baseline", "TemporalHierarchy", "FirstFrame"]:
        if os.path.exists(frame_desc_path):
            try:
                frame_descriptors = _load_npz_safe(frame_desc_path, "frame_descriptors")
                tqdm.write(f"[CACHE] Loaded per-frame descriptors for Subject-{subject}, Gesture-{gesture}")
            except Exception as e:
                tqdm.write(f"[WARN] Failed loading frame descriptors cache: {e}. Will recompute.")
                try:
                    os.remove(frame_desc_path)
                except Exception:
                    pass
        
        if frame_descriptors is None:
            tqdm.write(f"[EXTRACT] Computing per-frame descriptors (silhouette_tunnel) for Subject-{subject}, Gesture-{gesture}")
            frame_descriptors = silhouette_tunnel(depth_frames, background)
            try:
                np.savez_compressed(frame_desc_path, frame_descriptors=frame_descriptors)
                tqdm.write(f"[SAVED] Cached per-frame descriptors for Subject-{subject}, Gesture-{gesture}")
            except Exception as e:
                tqdm.write(f"[WARN] Could not save frame descriptors cache: {e}")

    # ----- 3) Load or compute morphology descriptors (for AdditionalTunnels) -----
    morph_desc_path = os.path.join(desc_dir, f"morphology_descriptors_K{K}.npz")
    morph_descriptors = None
    
    if variant == "AdditionalTunnels":
        if os.path.exists(morph_desc_path):
            try:
                morph_descriptors = _load_npz_safe(morph_desc_path, "morphology_descriptors")
                tqdm.write(f"[CACHE] Loaded morphology descriptors K={K} for Subject-{subject}, Gesture-{gesture}")
            except Exception as e:
                tqdm.write(f"[WARN] Failed loading morphology descriptors cache: {e}. Will recompute.")
                try:
                    os.remove(morph_desc_path)
                except Exception:
                    pass
        
        if morph_descriptors is None:
            tqdm.write(f"[EXTRACT] Computing morphology descriptors K={K} for Subject-{subject}, Gesture-{gesture}")
            morph_descriptors = multichannel_descriptors(depth_frames, background, K=K)
            try:
                np.savez_compressed(morph_desc_path, morphology_descriptors=morph_descriptors)
                tqdm.write(f"[SAVED] Cached morphology descriptors for Subject-{subject}, Gesture-{gesture}")
            except Exception as e:
                tqdm.write(f"[WARN] Could not save morphology descriptors cache: {e}")

    # ----- 4) Compute the final aggregated descriptor based on variant -----
    descriptor = None
    
    if variant == "Baseline":
        # Simple covariance of per-frame descriptors
        cov = np.cov(frame_descriptors)
        ut = cov[np.triu_indices_from(cov)]
        descriptor = ut.astype(np.float32)

    elif variant == "TemporalHierarchy":
        # Hierarchical temporal covariance
        desc = temporal_hierar_cov(frame_descriptors)
        descriptor = np.asarray(desc, dtype=np.float32)

    elif variant == "AdditionalTunnels":
        # Concatenate morphology descriptors
        descs_arr = [np.asarray(d, dtype=np.float32).ravel() for d in morph_descriptors]
        descriptor = np.concatenate(descs_arr).astype(np.float32)

    elif variant == "FirstFrame":
        # Use only the first frame descriptor
        first_frame = depth_frames[:1]
        bg_first = np.median(first_frame, axis=0).astype(np.float32)
        F_first = silhouette_tunnel(first_frame, bg_first)
        cov = np.cov(F_first)
        ut = cov[np.triu_indices_from(cov)]
        descriptor = ut.astype(np.float32)

    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    # ----- 5) Cache the final aggregated descriptor -----
    if descriptor is not None:
        try:
            np.savez_compressed(final_desc_path, descriptor=descriptor)
            tqdm.write(f"[SAVED] Cached final descriptor for {variant}, Subject-{subject}, Gesture-{gesture}")
        except Exception as e:
            tqdm.write(f"[WARN] Could not save final descriptor cache: {e}")
    
    return descriptor


# -----------------------
# Ablation Routine
# -----------------------
def run_full_ablation():
    variants = ["Baseline", "TemporalHierarchy", "AdditionalTunnels", "FirstFrame"]
    results = {v: {} for v in variants}

    for variant in tqdm(variants, desc="Running Variants"):
        for subject in SUBJECTS:
            for gesture in GESTURES:
                try:
                    frames = load_depth_frames(subject, gesture, MAX_FRAMES)
                    if frames.size == 0:
                        continue
                    bg = estimate_background(frames)
                    desc = compute_descriptors_variant(frames, bg, variant, subject, gesture)
                    results[variant][f"{subject}-{gesture}"] = desc
                except Exception as e:
                    tqdm.write(f"[WARN] {subject}-{gesture}: {e}")
                    continue

        # save partial results per variant (so you have them as they finish)
        try:
            variant_results = {k: v for k, v in results[variant].items() if v is not None}
            np.savez_compressed(os.path.join(RESULTS_DIR, f"ablation_results_{variant}.npz"), **variant_results)
            tqdm.write(f"[INFO] Saved partial results for {variant}")
        except Exception as e:
            tqdm.write(f"[WARN] Could not save partial results for {variant}: {e}")

    return results


# -----------------------
# Evaluation with PyEER
# -----------------------
def build_per_gesture_map(results, variant):
    per_gesture = defaultdict(lambda: defaultdict(list))
    for key, desc in results[variant].items():
        if desc is None:
            continue
        subj, gesture = key.split('-', 1)
        per_gesture[gesture][subj].append(np.asarray(desc).ravel())
    return per_gesture


def compute_eer_and_plot(per_gesture_map, variant):
    variant_dir = os.path.join(RESULTS_DIR, variant)
    os.makedirs(variant_dir, exist_ok=True)

    log_lines = []
    for gesture, subj_map in per_gesture_map.items():
        genuine_scores, impostor_scores = [], []
        gallery_templates = {s: np.mean(subj_map[s], axis=0) for s in subj_map if s in GALLERY_SUBJECTS}

        for psubj, samples in subj_map.items():
            for sample in samples:
                for gsubj, gtemp in gallery_templates.items():
                    dist = cosine_distances(sample.reshape(1, -1), gtemp.reshape(1, -1))[0, 0]
                    if psubj == gsubj:
                        genuine_scores.append(1 - dist)
                    else:
                        impostor_scores.append(1 - dist)

        if len(genuine_scores) == 0 or len(impostor_scores) == 0:
            tqdm.write(f"[WARN] Not enough scores for gesture {gesture} variant {variant} (genuine={len(genuine_scores)}, impostor={len(impostor_scores)})")
            continue

        eer_stats = get_eer_stats(genuine_scores, impostor_scores)
        eer = eer_stats.eer
        log_lines.append(f"Gesture: {gesture} | Variant: {variant} | EER: {eer:.4f}\n")
        tqdm.write(f"[RESULT] Gesture: {gesture} | Variant: {variant} | EER: {eer:.4f}")

        # Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(impostor_scores, bins=50, alpha=0.6, label='Impostor', density=True)
        plt.hist(genuine_scores, bins=50, alpha=0.6, label='Genuine', density=True)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.legend()
        plt.title(f"{gesture} ({variant}) EER={eer:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(variant_dir, f"{gesture}_hist.png"), dpi=200)
        plt.close()

        # DET Plot
        report_path = os.path.join(variant_dir, f"{gesture}_DET.png")
        try:
            generate_eer_report([eer_stats], report_path)
        except Exception as e:
            tqdm.write(f"[WARN] generate_eer_report failed for {gesture} {variant}: {e}")

    # append summary lines to summary file
    with open(SUMMARY_LOG, "a") as f:
        f.writelines(log_lines)


# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    results = run_full_ablation()

    for variant in ["Baseline", "TemporalHierarchy", "AdditionalTunnels", "FirstFrame"]:
        gesture_map = build_per_gesture_map(results, variant)
        compute_eer_and_plot(gesture_map, variant)

    tqdm.write(f"\nAll results stored in '{RESULTS_DIR}' directory.\n")