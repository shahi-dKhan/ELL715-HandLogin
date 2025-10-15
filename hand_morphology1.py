import numpy as np
from scipy.ndimage import label, generate_binary_structure
from tqdm import tqdm

# Import your optimized functions
from sillhouette_tunnel import silhouette_tunnel, compute_directional_distances, compute_temporal_directions
from covariance_matrix import covariance_descriptor
from hierarchy import temporal_hierar_cov


def compute_sub_silhouettes(depth_frames, silhouettes, K=3):
    """
    Compute K sub-silhouette tunnels based on quantile thresholds of depth
    within each silhouette frame. Keeps identical behavior as original logic.
    """
    T, H, W = depth_frames.shape
    sub_tunnels = [np.zeros((T, H, W), dtype=bool) for _ in range(K)]
    struct = generate_binary_structure(2, 2)

    for t in range(T):
        mask = silhouettes[t]
        if not mask.any():
            continue

        frame = depth_frames[t]
        valid_depths = frame[mask]
        thresholds = np.quantile(valid_depths, np.linspace(0, 1, K + 1))

        for k in range(K):
            # Last bin includes max value (same subtlety as your comment)
            if k == K - 1:
                sub_mask = mask & (frame >= thresholds[k]) & (frame <= thresholds[k + 1])
            else:
                sub_mask = mask & (frame >= thresholds[k]) & (frame < thresholds[k + 1])

            labeled, num = label(sub_mask, structure=struct)
            # Remove small connected regions (<20 pixels)
            for lbl in range(1, num + 1):
                if np.sum(labeled == lbl) < 20:
                    sub_mask[labeled == lbl] = False

            sub_tunnels[k][t] = sub_mask

    return sub_tunnels


def multichannel_descriptors(depth_frames, background, K=3):
    """
    Compute descriptors for multiple cases:
        - 1 full silhouette descriptor
        - K sub-silhouette descriptors
    Each descriptor is a 735-D vector (14x14 upper-triangular × 7 segments).
    Returns list: [full_silhouette_desc, sub1_desc, sub2_desc, sub3_desc]
    """
    T, H, W = depth_frames.shape
    silhouettes = np.zeros((T, H, W), dtype=bool)

    # Step 1: Full silhouette mask sequence
    for t in range(T):
        diff = np.abs(depth_frames[t] - background)
        silhouettes[t] = diff > 20  # same threshold as before

    # Step 2: Full silhouette descriptor
    F_full = silhouette_tunnel(depth_frames, background)
    desc_full = temporal_hierar_cov(F_full)

    # Step 3: Compute sub-silhouette tunnels once
    sub_tunnels = compute_sub_silhouettes(depth_frames, silhouettes, K)
    dt_plus, dt_minus = compute_temporal_directions(silhouettes)

    # Step 4: Extract features for each sub-silhouette (Numba-accelerated)
    descriptors = [desc_full]

    for k, tunnel in enumerate(sub_tunnels):
        feats = []
        for t in tqdm(range(T), desc=f"Sub-silhouette {k+1}", leave=False):
            mask = tunnel[t]
            if not mask.any():
                continue

            dE, dW, dN, dS, dNE, dNW, dSE, dSW = compute_directional_distances(mask)
            indices = np.argwhere(mask)
            frame = depth_frames[t]

            for (y, x) in indices:
                f = [
                    x, y, t / T, frame[y, x],
                    dE[y, x], dW[y, x], dN[y, x], dS[y, x],
                    dNE[y, x], dSW[y, x], dSE[y, x], dNW[y, x],
                    dt_plus[t, y, x], dt_minus[t, y, x],
                ]
                feats.append(f)

        if len(feats) == 0:
            descriptors.append(np.zeros(735))
            continue

        F = np.array(feats).T
        # Normalize same as your original
        F_min = F.min(axis=1, keepdims=True)
        F_max = F.max(axis=1, keepdims=True)
        F = (F - F_min) / (F_max - F_min + 1e-6)

        if F.shape[0] < 14:
            raise ValueError(f"Expected 14 features per pixel, got {F.shape[0]} instead.")

        desc = temporal_hierar_cov(F)
        descriptors.append(desc)

    return descriptors
