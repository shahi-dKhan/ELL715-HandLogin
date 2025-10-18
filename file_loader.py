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
def load_depth_frames(subject, gesture, test_folder, max_frames=None, base_path="./ell715_assg4"):
   

    gesture_path = os.path.join(base_path, subject, gesture)
    test_path = os.path.join(gesture_path, test_folder)
    msb_dir = os.path.join(test_path, "MSB")
    lsb_dir = os.path.join(test_path, "LSB")

        
        
    msb_files = sorted([f for f in os.listdir(msb_dir) if f.endswith(".png")])
    lsb_files = sorted([f for f in os.listdir(lsb_dir) if f.endswith(".png")])
    n = min(len(msb_files), len(lsb_files))
        
        
    ### Here we get the backround
    bg_msb_path = os.path.join(msb_dir, msb_files[0])
    bg_lsb_path = os.path.join(lsb_dir, lsb_files[0])
    bg_msb = cv2.imread(bg_msb_path, cv2.IMREAD_UNCHANGED)
    bg_lsb = cv2.imread(bg_lsb_path, cv2.IMREAD_UNCHANGED)
    if bg_msb.ndim == 3: bg_msb = bg_msb[:, :, 0]
    if bg_lsb.ndim == 3: bg_lsb = bg_lsb[:, :, 0]
    background = ((bg_msb.astype(np.uint16) << 8) | bg_lsb.astype(np.uint16)).astype(np.float32)

    # Loading the remaining frames
    frames = []
    for i in range(n):  # start from frame 2
        msb_path = os.path.join(msb_dir, msb_files[i])
        lsb_path = os.path.join(lsb_dir, lsb_files[i])
        msb = cv2.imread(msb_path, cv2.IMREAD_UNCHANGED)
        lsb = cv2.imread(lsb_path, cv2.IMREAD_UNCHANGED)
        if msb is None or lsb is None:
            tqdm.write(f"[WARN] Missing frame in {test_folder}, index {i}")
            continue
        if msb.ndim == 3: msb = msb[:, :, 0]
        if lsb.ndim == 3: lsb = lsb[:, :, 0]
        depth = ((msb.astype(np.uint16) << 8) | lsb.astype(np.uint16)).astype(np.float32)
        frames.append(depth)
        if max_frames and len(frames) >= max_frames:
            break

    frames = np.stack(frames, axis=0)
    # tqdm.write(f"[DONE] {subject}-{gesture}-{test_folder}: {frames.shape[0]} frames loaded, background ready.")
    return frames, background