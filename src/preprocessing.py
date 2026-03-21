"""
Tratar a imagem para mitigar sombras e variações de luz.~~
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def normalize_image(image_chw: np.ndarray, p_lower=2, p_upper=98) -> np.ndarray:
    """Normalizes bands using specified percentiles."""
    img = np.moveaxis(image_chw, 0, -1).astype(np.float32)
    normalized = np.zeros_like(img, dtype=np.uint8)

    for i in range(img.shape[2]):
        band = img[:, :, i]
        mask = band > 0
        if not np.any(mask): continue

        p2, p98 = np.percentile(band[mask], [p_lower, p_upper])
        band_norm = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
        normalized[:, :, i] = band_norm.astype(np.uint8)

    logger.info(f"Image normalized (p{p_lower}-{p_upper}).")
    return normalized

def apply_clahe(img_hwc: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Applies CLAHE with dynamic parameters."""
    # OpenCV expects tuple for grid size
    grid = tuple(tile_grid_size)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)

    channels = [clahe.apply(img_hwc[:, :, i]) for i in range(img_hwc.shape[2])]
    result = np.stack(channels, axis=-1)

    logger.info(f"CLAHE applied (limit={clip_limit}, tile={grid}).")
    return result

def preprocess_image(image_chw: np.ndarray, config: dict) -> np.ndarray:
    """Preprocessing pipeline driven by config dict."""
    logger.info("Starting preprocessing...")

    # Extract params from config
    norm_cfg = config.get('normalization', {})
    clahe_cfg = config.get('clahe', {})

    img_norm = normalize_image(
        image_chw,
        p_lower=norm_cfg.get('p_lower', 2),
        p_upper=norm_cfg.get('p_upper', 98)
    )

    img_pre = apply_clahe(
        img_norm,
        clip_limit=clahe_cfg.get('clip_limit', 2.0),
        tile_grid_size=clahe_cfg.get('tile_grid_size', [8, 8])
    )

    return img_pre
