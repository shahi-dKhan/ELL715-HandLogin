#!/usr/bin/env python3
"""
efficient_test_plan.py

- Caches final descriptors only (per subject/gesture/variant).
- Reuses cached descriptors on subsequent runs.
- Computes EER and stores histograms and DET plots under results/.
"""

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
SIL_CACHE_DIR = os.path.join(RESULTS_DIR, "silhouettes")
os.makedirs(SIL_CACHE_DIR, exist_ok=True)
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
        lsb_files = sorted([f for f in os.listdir(lsb_path) if f.lower().endswith(".png")])
        msb_files = sorted([f for f in os.listdir(msb_path) if f.lower().endswith(".png")])
        n = min(len(lsb_files), len(msb_files))
        if n == 0:
            continue

        for i in range(1, n + 1):
            lsb_file = os.path.join(lsb_path, f"LSB{i}.png")
            msb_file = os.path.join(msb_path, f"MSB{i}.png")
            # fallback to sorted lists if explicitly numbered files missing
            if not os.path.exists(lsb_file):
                try:
                    lsb_file = os.path.join(lsb_path, lsb_files[i - 1])
                except Exception:
                    continue
            if not os.path.exists(msb_file):
                try:
                    msb_file = os.path.join(msb_path, msb_files[i - 1])
                except Exception:
                    continue

            lsb = cv2.imread(lsb_file, cv2.IMREAD_UNCHANGED)
            msb = cv2.imread(msb_file, cv2.IMREAD_UNCHANGED)
            if lsb is None:
                continue
            if lsb.ndim == 3:
                lsb = lsb[:, :, 0]
            if msb is None:
                depth = lsb.astype(np.float32)
            else:
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
    d = os.path.join(SIL_CACHE_DIR, subject, gesture)
    os.makedirs(d, exist_ok=True)
    return d


def _safe_npz_load(path, key=None):
    """
    Load an .npz file and return the requested key if available.
    If key is None, returns a dict of arrays.
    """
    with np.load(path, allow_pickle=True) as data:
        files = list(data.files)
        if key is None:
            return {k: data[k] for k in files}
        if key in files:
            return data[key].copy()
        # fallback: return first array if key missing
        return data[files[0]].copy()


def _safe_key_name(raw_key):
    # produce a filesystem/key-safe name for saving in npz (replace '-' with '_')
    return raw_key.replace("-", "_")


# -----------------------
# Descriptor computation & caching (final-descriptor caching only)
# -----------------------
def compute_descriptors_variant(depth_frames, background, variant, subject, gesture, K=SUB_TUNNEL_K):
    """
    Compute final descriptor for (subject,gesture,variant). If a cached final descriptor
    exists it is loaded and returned immediately.
    """
    if depth_frames.size == 0:
        tqdm.write(f"[SKIP] No frames for {subject}-{gesture}")
        return None

    sil_dir = _cache_dir_for(subject, gesture)
    desc_path = os.path.join(sil_dir, f"descriptor_{variant}.npz")

    # 0) load cached final descriptor if present
    if os.path.exists(desc_path):
        try:
            descriptor = _safe_npz_load(desc_path, "descriptor")
            tqdm.write(f"[CACHE] Loaded descriptor {variant} for {subject}-{gesture}")
            return descriptor
        except Exception as e:
            tqdm.write(f"[WARN] Failed to load descriptor cache ({desc_path}): {e}. Recomputing.")
            try:
                os.remove(desc_path)
            except Exception:
                pass

    # 1) compute base F_full (silhouette_tunnel) once - used by most variants
    tqdm.write(f"[COMPUTE] silhouette_tunnel for {subject}-{gesture} (variant={variant})")
    F_full = silhouette_tunnel(depth_frames, background)
    if F_full is None or F_full.size == 0:
        tqdm.write(f"[ERROR] silhouette_tunnel returned empty for {subject}-{gesture}")
        return None

    descriptor = None

    # 2) compute variant-specific descriptors
    if variant == "Baseline":
        cov = np.cov(F_full)
        ut = cov[np.triu_indices_from(cov)]
        descriptor = ut.astype(np.float32)

    elif variant == "TemporalHierarchy":
        descriptor = np.asarray(temporal_hierar_cov(F_full), dtype=np.float32)

    elif variant == "AdditionalTunnels":
        # compute sub_tunnels only when needed
        tqdm.write(f"[COMPUTE] multichannel_descriptors (K={K}) for {subject}-{gesture}")
        sub_tunnels = multichannel_descriptors(depth_frames, background, K=K)
        # sub_tunnels expected to be iterable/list of descriptors
        if not sub_tunnels:
            tqdm.write(f"[WARN] multichannel_descriptors returned empty for {subject}-{gesture}")
            return None
        descs_arr = [np.asarray(d, dtype=np.float32).ravel() for d in sub_tunnels]
        descriptor = np.concatenate(descs_arr).astype(np.float32)

    elif variant == "FirstFrame":
        first_frame = depth_frames[:1]
        bg_first = np.median(first_frame, axis=0).astype(np.float32)
        F_first = silhouette_tunnel(first_frame, bg_first)
        cov = np.cov(F_first)
        ut = cov[np.triu_indices_from(cov)]
        descriptor = ut.astype(np.float32)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    # 3) cache descriptor
    if descriptor is not None:
        try:
            np.savez_compressed(desc_path, descriptor=descriptor)
            tqdm.write(f"[SAVED] Cached descriptor {variant} for {subject}-{gesture}")
        except Exception as e:
            tqdm.write(f"[WARN] Could not save descriptor: {e}")

    return descriptor


# -----------------------
# Ablation routine
# -----------------------
def run_full_ablation():
    variants = ["Baseline", "TemporalHierarchy", "AdditionalTunnels", "FirstFrame"]
    results = {v: {} for v in variants}

    for variant in tqdm(variants, desc="Running Variants"):
        for subject in tqdm(SUBJECTS, desc=f"{variant} subjects", leave=False):
            for gesture in GESTURES:
                key = f"{subject}-{gesture}"
                try:
                    frames = load_depth_frames(subject, gesture, MAX_FRAMES)
                    if frames.size == 0:
                        continue
                    bg = estimate_background(frames)
                    desc = compute_descriptors_variant(frames, bg, variant, subject, gesture)
                    results[variant][key] = desc
                except Exception as e:
                    tqdm.write(f"[WARN] {key}: {e}")
                    results[variant][key] = None

        # save partial results per variant (safe-key names)
        variant_results = { _safe_key_name(k): v for k, v in results[variant].items() if v is not None }
        try:
            path = os.path.join(RESULTS_DIR, f"ablation_results_{variant}.npz")
            if variant_results:
                np.savez_compressed(path, **variant_results)
                tqdm.write(f"[INFO] Saved partial results for {variant} -> {path}")
            else:
                tqdm.write(f"[INFO] No descriptors to save for {variant}")
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
        # only include subjects that have descriptors
        gallery_templates = {s: np.mean(subj_map[s], axis=0) for s in subj_map if s in GALLERY_SUBJECTS}
        genuine_scores = []
        impostor_scores = []

        for psubj, samples in subj_map.items():
            for sample in samples:
                for gsubj, gtemp in gallery_templates.items():
                    dist = cosine_distances(sample.reshape(1, -1), gtemp.reshape(1, -1))[0, 0]
                    score = 1.0 - dist
                    if psubj == gsubj:
                        genuine_scores.append(score)
                    else:
                        impostor_scores.append(score)

        if len(genuine_scores) == 0 or len(impostor_scores) == 0:
            tqdm.write(f"[WARN] Not enough scores for gesture {gesture} variant {variant} (g={len(genuine_scores)}, i={len(impostor_scores)})")
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

        # DET Plot (use eer_stats directly)
        report_path = os.path.join(variant_dir, f"{gesture}_DET.png")
        try:
            generate_eer_report(eer_stats, report_path)
        except Exception as e:
            tqdm.write(f"[WARN] generate_eer_report failed for {gesture} {variant}: {e}")

    # append summary lines to summary file
    if log_lines:
        with open(SUMMARY_LOG, "a") as f:
            f.writelines(log_lines)


# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    tqdm.write("[START] Running ablation / descriptor extraction")
    results = run_full_ablation()

    tqdm.write("[START] Running evaluation & plotting")
    for variant in ["Baseline", "TemporalHierarchy", "AdditionalTunnels", "FirstFrame"]:
        gesture_map = build_per_gesture_map(results, variant)
        compute_eer_and_plot(gesture_map, variant)

    tqdm.write(f"[DONE] All results stored in '{RESULTS_DIR}' directory. Summary appended to {SUMMARY_LOG}")
