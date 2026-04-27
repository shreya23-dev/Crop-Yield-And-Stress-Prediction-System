import numpy as np
from pathlib import Path
import tensorflow as tf

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SAT_DIR = DATA_DIR / "satellite_images"
SAT_MONTHS = [6, 7, 8, 9, 10, 11]
IMAGE_H, IMAGE_W = 32, 32

def fetch_satellite_images(district: str, year: int) -> np.ndarray:
    """
    Since we don't have Earth Engine credentials on the server,
    we load the previously exported numpy arrays.
    Returns: numpy array shape (1, 6, H, W, 1)
    """
    images = np.zeros((1, 6, IMAGE_H, IMAGE_W, 1), dtype=np.float32)
    district = district.strip()
    
    for m, month in enumerate(SAT_MONTHS):
        fpath = SAT_DIR / f"{district}_{year}_{month:02d}.npy"
        if fpath.exists():
            arr = np.load(fpath).astype(np.float32)
            if arr.ndim == 3: 
                arr = arr[:, :, 0]
            if arr.shape != (IMAGE_H, IMAGE_W):
                arr = tf.image.resize(arr[..., np.newaxis], [IMAGE_H, IMAGE_W]).numpy()[..., 0]
            images[0, m, :, :, 0] = arr
            
    return images

def extract_cnn_features(images: np.ndarray, cnn_model) -> np.ndarray:
    if cnn_model is None:
        return np.zeros((1, 6, 128), dtype=np.float32)

    sat_min   = images.min()
    sat_range = max(images.max() - sat_min, 1e-6)
    imgs_norm = np.clip((images - sat_min) / sat_range, 0.0, 1.0)

    feats = np.zeros((1, 6, 128), dtype=np.float32)
    for m in range(6):
        feats[0, m, :] = cnn_model.predict(imgs_norm[0:1, m, :, :, :], verbose=0)
    return feats
