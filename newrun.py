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
from itertools import chain

# -----------------------
# CONFIG
# -----------------------
DATASET_PATH = "./ell715_assg4"
SUBJECTS = [f"{i:02d}" for i in chain(range(1, 3), range(9, 11))]
GESTURES = ["Compass", "Piano", "Push", "UCDO"]
GALLERY_SUBJECTS = [f"{i:02d}" for i in range(1, 3)]   # 01–08
PROBE_SUBJECTS = [f"{i:02d}" for i in range(9, 11)]    # 09–21
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
    if depth_frames.size == 0:
        return None

    desc_dir = _cache_dir_for(subject, gesture)
    
    # ----- 1) Check if final descriptor is already cached -----
    final_desc_path = os.path.join(desc_dir, f"descriptor_{variant}.npz")
    if os.path.exists(final_desc_path):
        tqdm.write(f"[CACHE] Found cached final descriptor for {variant}, Subject-{subject}, Gesture-{gesture}")
        data = np.load(final_desc_path, allow_pickle=True)
        descriptor = data["descriptor"]
        data.close()
        # print("Descriptor shape:", descriptor.shape)
        return descriptor

    # ----- 2) Load or compute per-frame base descriptors (for Baseline, TemporalHierarchy, FirstFrame) -----
    frame_desc_path = os.path.join(desc_dir, "frame_descriptors.npz")
    frame_descriptors = None

    if variant in ["Baseline", "TemporalHierarchy", "FirstFrame"]:
        if os.path.exists(frame_desc_path):
            tqdm.write(f"[CACHE] Found per-frame descriptors for Subject-{subject}, Gesture-{gesture}")
            data = np.load(frame_desc_path, allow_pickle=True)
            frame_descriptors = data["frame_descriptors"]
            data.close()
        else:
            tqdm.write(f"[EXTRACT] Computing per-frame descriptors (silhouette_tunnel) for Subject-{subject}, Gesture-{gesture}")
            frame_descriptors = silhouette_tunnel(depth_frames, background)
            
            np.savez_compressed(frame_desc_path, frame_descriptors=frame_descriptors)
            tqdm.write(f"[SAVED] Cached per-frame descriptors for Subject-{subject}, Gesture-{gesture}")

            sil_dir = os.path.join("results", "silhouettes", subject, gesture)
            os.makedirs(sil_dir, exist_ok=True)
            num_to_save = min(10, frame_descriptors.shape[0])  # Save only first 10 silhouettes
            # for i in range(num_to_save):
            #     sil_img = frame_descriptors[i]
            #     # Normalize and convert to 8-bit image for saving
            #     sil_img_norm = cv2.normalize(sil_img, None, 0, 255, cv2.NORM_MINMAX)
            #     sil_img_uint8 = sil_img_norm.astype(np.uint8)
            #     # out_path = os.path.join(sil_dir, f"silhouette_{i:03d}.png")
            #     # cv2.imwrite(out_path, sil_img_uint8)
            # tqdm.write(f"[VISUAL] Saved {num_to_save} silhouette PNGs for {subject}-{gesture} in {sil_dir}")

    # ----- 3) Load or compute morphology descriptors (for AdditionalTunnels) -----
    print("Frame descriptors shape:", frame_descriptors.shape)
    morph_desc_path = os.path.join(desc_dir, f"morphology_descriptors_K{K}.npz")
    morph_descriptors = None
    if variant == "AdditionalTunnels":
        if os.path.exists(morph_desc_path):
            tqdm.write(f"[CACHE] Found morphology descriptors K={K} for Subject-{subject}, Gesture-{gesture}")
            data = np.load(morph_desc_path, allow_pickle=True)
            morph_descriptors = data["morphology_descriptors"]
            data.close()
        else:
            tqdm.write(f"[EXTRACT] Computing morphology descriptors K={K} for Subject-{subject}, Gesture-{gesture}")
            morph_descriptors = multichannel_descriptors(depth_frames, background, K=K)
            np.savez_compressed(morph_desc_path, morphology_descriptors=morph_descriptors)
            tqdm.write(f"[SAVED] Cached morphology descriptors for Subject-{subject}, Gesture-{gesture}")

    # ----- 4) Compute the final aggregated descriptor based on variant -----
    descriptor = None
    tqdm.write(f"[INFO] Computing new descriptor for {variant}, Subject-{subject}, Gesture-{gesture}")

    if variant == "Baseline":
        cov = np.cov(frame_descriptors)
        ut = cov[np.triu_indices_from(cov)]
        descriptor = ut.astype(np.float32)

    elif variant == "TemporalHierarchy":
        desc = temporal_hierar_cov(frame_descriptors)
        descriptor = np.asarray(desc, dtype=np.float32)

    elif variant == "AdditionalTunnels":
        descs_arr = [np.asarray(d, dtype=np.float32).ravel() for d in morph_descriptors]
        descriptor = np.concatenate(descs_arr).astype(np.float32)

    elif variant == "FirstFrame":
        first_frame = depth_frames[:1]
        bg_first = np.median(first_frame, axis=0).astype(np.float32)
        F_first = silhouette_tunnel(first_frame, bg_first)
        cov = np.cov(F_first)
        ut = cov[np.triu_indices_from(cov)]
        descriptor = ut.astype(np.float32)

        # Optional: save first-frame silhouette too
        sil_dir = os.path.join("results", "silhouettes", subject, gesture)
        os.makedirs(sil_dir, exist_ok=True)
        sil_img = F_first[0]
        sil_img_norm = cv2.normalize(sil_img, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(sil_dir, "silhouette_first_frame.png"), sil_img_norm.astype(np.uint8))
        tqdm.write(f"[VISUAL] Saved first-frame silhouette for {subject}-{gesture}")

    else:
        raise ValueError(f"Unknown variant: {variant}")

    # ----- 5) Save final descriptor -----
    np.savez_compressed(final_desc_path, descriptor=descriptor)
    tqdm.write(f"[SAVED] Cached final descriptor for {variant}, Subject-{subject}, Gesture-{gesture}")

    return descriptor



# -----------------------
# Ablation Routine
# -----------------------
def run_full_ablation():
    # variants = ["Baseline", "TemporalHierarchy", "AdditionalTunnels", "FirstFrame"]
    variants = ["Baseline"]  # for quick testing
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

        # # Histogram
        # plt.figure(figsize=(6, 4))
        # plt.hist(genuine_scores, bins=min(50, len(np.unique(genuine_scores))), alpha=0.6, label='Genuine', density=True)
        # plt.hist(impostor_scores, bins=min(50, len(np.unique(impostor_scores))), alpha=0.6, label='Impostor', density=True)
        # plt.xlabel('Cosine Similarity')
        # plt.ylabel('Density')
        # plt.legend()
        # plt.title(f"{gesture} ({variant}) EER={eer:.3f}")
        # plt.tight_layout()
        # plt.savefig(os.path.join(variant_dir, f"{gesture}_hist.png"), dpi=200)
        # plt.close()
        # Histogram
    plt.figure(figsize=(6, 4))

    def safe_hist(data, label):
        if len(data) == 0:
            tqdm.write(f"[WARN] No {label} scores to plot")
            return False

        data_min, data_max = np.min(data), np.max(data)
        if not np.isfinite(data_min) or not np.isfinite(data_max):
            tqdm.write(f"[WARN] {label} scores have NaN/Inf values")
            return False

        data_range = data_max - data_min
        if data_range < 1e-6:
            tqdm.write(f"[WARN] Skipping {label} histogram: data range too small ({data_min:.6f}–{data_max:.6f})")
            return False

        unique_count = len(np.unique(data))
        num_bins = min(50, unique_count)
        plt.hist(data, bins=num_bins, alpha=0.6, label=label, density=True)
        return True


    ok1 = safe_hist(impostor_scores, "Impostor")
    ok2 = safe_hist(genuine_scores, "Genuine")

    if ok1 or ok2:
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.legend()
        plt.title(f"{gesture} ({variant}) EER={eer:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(variant_dir, f"{gesture}_hist.png"), dpi=200)
    else:
        tqdm.write(f"[SKIP] Histogram skipped for {gesture} ({variant})")
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
    variant = "Baseline"
    # for variant in ["Baseline", "TemporalHierarchy", "AdditionalTunnels", "FirstFrame"]:
    gesture_map = build_per_gesture_map(results, variant)
    compute_eer_and_plot(gesture_map, variant)

    tqdm.write(f"\nAll results stored in '{RESULTS_DIR}' directory.\n")
    
    
