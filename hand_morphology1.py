import os
import numpy as np
from scipy.ndimage import label, generate_binary_structure
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from silhouette_tunnel import compute_directional_distances, compute_temporal_directions
from hierarchy import temporal_hierar_cov
import cv2
# Configure Numba threads to use all CPU cores (Mac M2 typically shows 8)
try:
    from numba import set_num_threads, get_num_threads
    nproc = os.cpu_count() or 4
    set_num_threads(nproc)
    # print helpful info
    print(f"[INFO] numba threads set to: {get_num_threads()}")
except Exception:
    nproc = os.cpu_count() or 4
    print(f"[WARN] numba not available to set threads, using nproc={nproc}")






def save_sub_silhouette_masks(sub_tunnels, base_dir, prefix="sub"):
    """
    Save each sub-silhouette mask as PNG images for visualization.

    Args:
        sub_tunnels : list of (T,H,W) boolean arrays
        base_dir    : directory to save results in
        prefix      : optional prefix for folder names (default: 'sub')
    """
    os.makedirs(base_dir, exist_ok=True)

    for k, tunnel in enumerate(sub_tunnels):
        sub_dir = os.path.join(base_dir, f"{prefix}{k+1}")
        os.makedirs(sub_dir, exist_ok=True)
        T = tunnel.shape[0]
        for t in tqdm(range(T), desc=f"Saving {prefix}{k+1}", leave=False):
            mask_img = (tunnel[t].astype(np.uint8)) * 255
            out_path = os.path.join(sub_dir, f"frame_{t:03d}.png")
            cv2.imwrite(out_path, mask_img)







def compute_sub_silhouettes(depth_frames, silhouettes, K=3, min_region_size=20):
    T, H, W = depth_frames.shape
    sub_tunnels = [np.zeros((T, H, W), dtype=bool) for _ in range(K)]
    struct = generate_binary_structure(2, 2)
    for t in range(T):
        mask = silhouettes[t]
        if not mask.any():
            continue
        frame = depth_frames[t]
        valid_depths = frame[mask]
        if valid_depths.size < K:
            continue
        thresholds = np.quantile(valid_depths, np.linspace(0, 1, K + 1))
        if np.allclose(thresholds[0], thresholds[-1]):
            continue
        if np.any(np.diff(thresholds) == 0):
            rng = thresholds[-1] - thresholds[0]
            eps = max(rng * 1e-6, 1e-3)
            for i in range(1, len(thresholds)):
                if thresholds[i] <= thresholds[i - 1]:
                    thresholds[i] = thresholds[i - 1] + eps
        for k in range(K):
            low = thresholds[k]
            high = thresholds[k + 1]
            if k == K - 1:
                sub_mask = mask & (frame >= low) & (frame <= high)
            else:
                sub_mask = mask & (frame >= low) & (frame < high)
            if sub_mask.any():
                labeled, num = label(sub_mask, structure=struct)
                for lbl in range(1, num + 1):
                    comp = (labeled == lbl)
                    if comp.sum() < min_region_size:
                        sub_mask[comp] = False
            sub_tunnels[k][t] = sub_mask
    return sub_tunnels


def _extract_14d_features_for_mask_sequence(depth_frames, masks, dt_plus=None, dt_minus=None):
    T, H, W = depth_frames.shape
    if dt_plus is None or dt_minus is None:
        dt_plus, dt_minus = compute_temporal_directions(masks)
    feat_rows = []
    for t in range(T):
        mask = masks[t]
        if not mask.any():
            continue
        dE, dW, dN, dS, dNE, dNW, dSE, dSW = compute_directional_distances(mask)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        x_arr = xs.astype(np.float32)
        y_arr = ys.astype(np.float32)
        t_arr = np.full_like(x_arr, float(t), dtype=np.float32)
        frame = depth_frames[t]
        depth_arr = frame[ys, xs].astype(np.float32)
        dE_arr = dE[ys, xs].astype(np.float32)
        dW_arr = dW[ys, xs].astype(np.float32)
        dN_arr = dN[ys, xs].astype(np.float32)
        dS_arr = dS[ys, xs].astype(np.float32)
        dNE_arr = dNE[ys, xs].astype(np.float32)
        dSW_arr = dSW[ys, xs].astype(np.float32)
        dSE_arr = dSE[ys, xs].astype(np.float32)
        dNW_arr = dNW[ys, xs].astype(np.float32)
        dtp_arr = dt_plus[t, ys, xs].astype(np.float32)
        dtm_arr = dt_minus[t, ys, xs].astype(np.float32)
        frame_feats = np.column_stack([
            x_arr, y_arr, t_arr, depth_arr,
            dE_arr, dW_arr, dN_arr, dS_arr,
            dNE_arr, dSW_arr, dSE_arr, dNW_arr,
            dtp_arr, dtm_arr
        ])
        feat_rows.append(frame_feats)
    if len(feat_rows) == 0:
        return np.zeros((14, 0), dtype=np.float32)
    all_feats = np.vstack(feat_rows)
    F = all_feats.T.astype(np.float32)
    F_min = F.min(axis=1, keepdims=True)
    F_max = F.max(axis=1, keepdims=True)
    denom = (F_max - F_min)
    denom[denom == 0] = 1.0
    F = (F - F_min) / denom
    return F


def _process_sub_tunnel(task_args):
    k, tunnel, depth_frames, dt_plus, dt_minus = task_args
    # If tunnel empty -> zero descriptor handled by caller
    F_sub = _extract_14d_features_for_mask_sequence(depth_frames, tunnel, dt_plus=dt_plus, dt_minus=dt_minus)
    if F_sub.shape[1] == 0:
        return (k, None)
    desc_sub = temporal_hierar_cov(F_sub)
    return (k, np.asarray(desc_sub, dtype=np.float32))


def multichannel_descriptors(subject, gesture, test_folder, depth_frames, background, F_full=None, desc_full=None,
                                       K=3, min_region_size=20, silhouette_threshold=20,
                                       max_workers=None, verbose=False):

    T, H, W = depth_frames.shape

    # silhouettes
    silhouettes = np.zeros((T, H, W), dtype=bool)
    for t in range(T):
        diff = np.abs(depth_frames[t] - background)
        silhouettes[t] = diff > silhouette_threshold
    struct = generate_binary_structure(2, 2)
    for t in range(T):
        labeled, num = label(silhouettes[t], structure=struct)
        if num == 0:
            continue
        largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        silhouettes[t] = (labeled == largest)

    # full descriptor
    if desc_full is None:
        if F_full is None:
            F_full = _extract_14d_features_for_mask_sequence(depth_frames, silhouettes)
        if F_full.shape[1] == 0:
            desc_full = np.zeros(((14 * 15) // 2,), dtype=np.float32)
        else:
            desc_full = temporal_hierar_cov(F_full)
    descriptors = [np.asarray(desc_full, dtype=np.float32)]

    # sub-tunnels
    sub_tunnels = compute_sub_silhouettes(depth_frames, silhouettes, K=K, min_region_size=min_region_size)
    save_dir = os.path.join("results", "sub_silhouettes", subject, gesture, test_folder)
    save_sub_silhouette_masks(sub_tunnels, save_dir)
    dt_plus, dt_minus = compute_temporal_directions(silhouettes)

    # determine number of worker threads: keep small to avoid oversubscription
    if max_workers is None:
        max_workers = min(max(1, K), max(1, (os.cpu_count() or 4) // 2))
    max_workers = max(1, int(max_workers))

    tasks = []
    for k, tunnel in enumerate(sub_tunnels):
        tasks.append((k, tunnel, depth_frames, dt_plus, dt_minus))

    results = [None] * K
    # Use ThreadPoolExecutor so Numba kernels (which release GIL) can run concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        if verbose:
            # submit all then process as completed with tqdm
            futures = {ex.submit(_process_sub_tunnel, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Sub-tunnels"):
                k, desc = fut.result()
                results[k] = desc
        else:
            futures = [ex.submit(_process_sub_tunnel, t) for t in tasks]
            for fut in as_completed(futures):
                k, desc = fut.result()
                results[k] = desc

    # assemble descriptors list in order (0..K-1)
    for k in range(K):
        desc = results[k]
        if desc is None:
            descriptors.append(np.zeros_like(descriptors[0]))
        else:
            descriptors.append(desc)

    return descriptors
