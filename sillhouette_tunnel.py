import numpy as np
from scipy.ndimage import label, generate_binary_structure
from tqdm import tqdm
from numba import njit, prange


# ===============================================================
# --- 1️⃣ Directional Distance Computation (Numba-accelerated) ---
# ===============================================================
@njit(parallel=True)
def compute_directional_distances(mask):
    H, W = mask.shape
    dE = np.zeros((H, W), np.int32)
    dW = np.zeros((H, W), np.int32)
    dN = np.zeros((H, W), np.int32)
    dS = np.zeros((H, W), np.int32)
    dNE = np.zeros((H, W), np.int32)
    dNW = np.zeros((H, W), np.int32)
    dSE = np.zeros((H, W), np.int32)
    dSW = np.zeros((H, W), np.int32)

    # EAST
    for y in prange(H):
        for x in range(W - 2, -1, -1):
            if mask[y, x]:
                if mask[y, x + 1]:
                    dE[y, x] = dE[y, x + 1] + 1
                else:
                    dE[y, x] = 1

    # WEST
    for y in prange(H):
        for x in range(1, W):
            if mask[y, x]:
                if mask[y, x - 1]:
                    dW[y, x] = dW[y, x - 1] + 1
                else:
                    dW[y, x] = 1

    # NORTH
    for y in range(1, H):
        for x in prange(W):
            if mask[y, x]:
                if mask[y - 1, x]:
                    dN[y, x] = dN[y - 1, x] + 1
                else:
                    dN[y, x] = 1

    # SOUTH
    for y in range(H - 2, -1, -1):
        for x in prange(W):
            if mask[y, x]:
                if mask[y + 1, x]:
                    dS[y, x] = dS[y + 1, x + 0] + 1
                else:
                    dS[y, x] = 1

    # DIAGONALS (NE, NW, SE, SW)
    for y in range(1, H):
        for x in range(W - 2, -1, -1):  # NE
            if mask[y, x]:
                if mask[y - 1, x + 1]:
                    dNE[y, x] = dNE[y - 1, x + 1] + 1
                else:
                    dNE[y, x] = 1

    for y in range(1, H):
        for x in range(1, W):  # NW
            if mask[y, x]:
                if mask[y - 1, x - 1]:
                    dNW[y, x] = dNW[y - 1, x - 1] + 1
                else:
                    dNW[y, x] = 1

    for y in range(H - 2, -1, -1):
        for x in range(W - 2, -1, -1):  # SE
            if mask[y, x]:
                if mask[y + 1, x + 1]:
                    dSE[y, x] = dSE[y + 1, x + 1] + 1
                else:
                    dSE[y, x] = 1

    for y in range(H - 2, -1, -1):
        for x in range(1, W):  # SW
            if mask[y, x]:
                if mask[y + 1, x - 1]:
                    dSW[y, x] = dSW[y + 1, x - 1] + 1
                else:
                    dSW[y, x] = 1

    return dE, dW, dN, dS, dNE, dNW, dSE, dSW


# ======================================================
# --- 2️⃣ Temporal Directional Distances (dt+, dt-) ---
# ======================================================
@njit(parallel=True)
def compute_temporal_directions(silhouettes):
    T, H, W = silhouettes.shape
    dt_plus = np.zeros((T, H, W), np.int32)
    dt_minus = np.zeros((T, H, W), np.int32)

    for y in prange(H):
        for x in range(W):
            # forward (dt+)
            for t in range(T - 2, -1, -1):
                if silhouettes[t, y, x]:
                    if silhouettes[t + 1, y, x]:
                        dt_plus[t, y, x] = dt_plus[t + 1, y, x] + 1
                    else:
                        dt_plus[t, y, x] = 1

            # backward (dt-)
            for t in range(1, T):
                if silhouettes[t, y, x]:
                    if silhouettes[t - 1, y, x]:
                        dt_minus[t, y, x] = dt_minus[t - 1, y, x] + 1
                    else:
                        dt_minus[t, y, x] = 1
    return dt_plus, dt_minus


# ==============================================
# --- 3️⃣ Main Silhouette Tunnel Extraction ---
# ==============================================
def silhouette_tunnel(depth_frames, background, threshold=20):
    """
    Compute silhouette tunnel and 14D features (same logic, now Numba-accelerated).
    Returns normalized feature matrix F of shape (14, N).
    """
    T, H, W = depth_frames.shape
    silhouettes = np.zeros((T, H, W), dtype=bool)

    # --- Step 1: Foreground mask ---
    for i in range(T):
        diff = np.abs(depth_frames[i] - background)
        silhouettes[i] = diff > threshold

    # --- Step 2: Keep largest connected component per frame ---
    struct = generate_binary_structure(2, 2)
    for i in range(T):
        labeled, num = label(silhouettes[i], structure=struct)
        if num == 0:
            continue
        largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        silhouettes[i] = (labeled == largest)

    # --- Step 3: Temporal run-lengths ---
    dt_plus, dt_minus = compute_temporal_directions(silhouettes)

    # --- Step 4: Feature extraction ---
    features = []

    for i in tqdm(range(T), desc="Feature extraction", unit="frame"):
        mask = silhouettes[i]
        indices = np.argwhere(mask)
        if len(indices) == 0:
            continue

        # Compute directional distances for this frame
        dE, dW, dN, dS, dNE, dNW, dSE, dSW = compute_directional_distances(mask)

        for y, x in indices:
            f = [
                x, y, i, depth_frames[i, y, x],
                dE[y, x], dW[y, x], dN[y, x], dS[y, x],
                dNE[y, x], dSW[y, x], dSE[y, x], dNW[y, x],
                dt_plus[i, y, x], dt_minus[i, y, x],
            ]
            features.append(f)

    # --- Step 5: Normalize features ---
    F = np.array(features).T
    F_min = F.min(axis=1, keepdims=True)
    F_max = F.max(axis=1, keepdims=True)
    F = (F - F_min) / (F_max - F_min + 1e-6)
    return F
