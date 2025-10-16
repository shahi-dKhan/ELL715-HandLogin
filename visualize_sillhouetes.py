import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from silhouette_tunnel import silhouette_tunnel


def load_depth_sequence(msb_dir, lsb_dir):
    """
    Load 16-bit depth frames from corresponding MSB and LSB directories.
    Converts RGBA PNGs to single-channel grayscale depth maps.
    """
    msb_files = sorted([f for f in os.listdir(msb_dir) if f.endswith('.png')])
    lsb_files = sorted([f for f in os.listdir(lsb_dir) if f.endswith('.png')])

    if len(msb_files) != len(lsb_files):
        raise ValueError(f"MSB and LSB folder size mismatch: {len(msb_files)} vs {len(lsb_files)}")

    frames = []
    frame_paths = []  # store tuples of (MSB_path, LSB_path)

    for msb_name, lsb_name in zip(msb_files, lsb_files):
        msb_path = os.path.join(msb_dir, msb_name)
        lsb_path = os.path.join(lsb_dir, lsb_name)

        msb = cv2.imread(msb_path, cv2.IMREAD_UNCHANGED)
        lsb = cv2.imread(lsb_path, cv2.IMREAD_UNCHANGED)

        if msb is None or lsb is None:
            raise IOError(f"Error reading: {msb_path} or {lsb_path}")

        # Handle 4-channel RGBA → use only one channel
        if msb.ndim == 3 and msb.shape[2] > 1:
            msb = msb[:, :, 0]
        if lsb.ndim == 3 and lsb.shape[2] > 1:
            lsb = lsb[:, :, 0]

        # Combine MSB + LSB to form 16-bit depth image
        depth = (msb.astype(np.uint16) << 8) + lsb.astype(np.uint16)
        frames.append(depth.astype(np.float32))
        frame_paths.append((msb_path, lsb_path))

    return np.stack(frames, axis=0), frame_paths  # (T, H, W), [(MSB, LSB), ...]


def visualize_silhouettes(base_dir, subject='01', gesture='Push', test='Test001', threshold=50, num_samples=6):
    """
    Visualize the silhouette tunnel for a given subject, gesture, and test.
    Each subplot will show the full path of the corresponding MSB and LSB images.
    """

    msb_dir = os.path.join(base_dir, subject, gesture, test, 'MSB')
    lsb_dir = os.path.join(base_dir, subject, gesture, test, 'LSB')

    if not os.path.exists(msb_dir) or not os.path.exists(lsb_dir):
        raise FileNotFoundError(f"MSB or LSB folder not found for {subject}/{gesture}/{test}")

    print(f"Loading depth sequence for {subject}/{gesture}/{test}...")
    frames, frame_paths = load_depth_sequence(msb_dir, lsb_dir)
    print(f"Loaded {frames.shape[0]} frames of size {frames.shape[1:]}")

    # Compute background (median)
    background = np.broadcast_to(frames[0], frames.shape)  # Using first frame as background

    # Compute silhouettes
    F, silhouettes = silhouette_tunnel(frames, background, threshold)
    print(f"Silhouettes computed: shape = {silhouettes.shape}")

    # Pick evenly spaced frames for visualization
    total_frames = silhouettes.shape[0]
    sample_idxs = [0,1,24,25,26,27]

    plt.figure(figsize=(12, 5))
    for i, idx in enumerate(sample_idxs):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(silhouettes[idx], cmap='gray')
        plt.axis('off')

        # Full file paths for MSB and LSB
        msb_path, lsb_path = frame_paths[idx]
        full_path_text = f"{os.path.abspath(msb_path)}\n{os.path.abspath(lsb_path)}"
        # plt.title(full_path_text, fontsize=6, loc='center', pad=5)
        plt.title(f"Frame {idx}", fontsize=6)

    plt.suptitle(f"{subject} - {gesture} - {test}", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    


if __name__ == "__main__":
    base_dir = "./ell715_assg4"  # Change path if necessary
    visualize_silhouettes(base_dir, subject="01", gesture="Compass", test="Test001", threshold=50)
