"""
Isolar o verde das mudas do restante da imagem (solo).
"""
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

def segmentation_RGB(rgb_image: np.ndarray, threshold: int = 80, kernel_size: int = 10):
    """
    Segmenta cada banda RGB da imagem usando threshold Otsu
    e aplica morfologia matemática (open) para limpeza.
    """
    logger.info(f"Starting RGB segmentation with threshold={threshold}, kernel_size={kernel_size}")
    band_names = ['Red (B1)', 'Green (B2)', 'Blue (B3)']

    if rgb_image.shape[0] == 3:
        rgb_image = np.moveaxis(rgb_image, 0, -1)

    logger.debug(f"Image shape for segmentation: {rgb_image.shape}")

    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    valid      = np.any(rgb_image > 0, axis=-1)

    masks = {}  # guarda as máscaras para a próxima célula

    for i, name in enumerate(band_names):
        logger.info(f"Processing band: {name}")
        band = rgb_image[:, :, i]

        # Morfologia — remove ruído
        mask_threshold = (band > threshold).astype(np.uint8) * 255
        opened = cv2.morphologyEx(mask_threshold, cv2.MORPH_OPEN, kernel)
        opened[~valid] = 255

        masks[name] = opened

    logger.info(f"RGB segmentation completed. {len(masks)} masks generated.")
    return masks

def calculate_index(rgb_image: np.ndarray):
    """Calculates common vegetation indices."""
    logger.info("Calculating vegetation indices (ExG, Smolka, Vwg, LGI)...")
    r = rgb_image[:, :, 0].astype(float)
    g = rgb_image[:, :, 1].astype(float)
    b = rgb_image[:, :, 2].astype(float)

    # Avoid division by zero
    g_safe = np.where(g == 0, 1e-6, g)
    denom_vwg = (2 * g + r + b)
    denom_vwg_safe = np.where(denom_vwg == 0, 1e-6, denom_vwg)

    indices = {
        'ExG':    4 * g - r - b,
        'Smolka': (g - np.maximum(r, b) ** 2) / g_safe,
        'Vwg':    (2 * g - (r + b)) / denom_vwg_safe,
        'LGI':    -0.884 * r + 1.262 * g - 0.311 * b,
    }
    logger.debug(f"Indices calculated: {list(indices.keys())}")
    return indices

def segment_index(rgb_image: np.ndarray, threshold_manual: int = 130, kernel_size: int = 10):
    logger.info(f"Starting green index segmentation (ExG + Smolka) with manual threshold={threshold_manual}")

    if rgb_image.ndim == 3 and rgb_image.shape[0] in (3, 4):
        rgb_image = np.moveaxis(rgb_image, 0, -1)
    rgb_image = rgb_image[:, :, :3]

    valid   = np.any(rgb_image > 0, axis=-1)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    indices = calculate_index(rgb_image)

    selected = {k: indices[k] for k in ('ExG', 'Smolka')}

    results = {}

    for name, index in selected.items():
        logger.info(f"Segmenting index: {name}")
        index_norm = cv2.normalize(index, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if name == 'ExG':
            index_norm = cv2.bitwise_not(index_norm)

        _, mask_man  = cv2.threshold(index_norm, threshold_manual, 255, cv2.THRESH_BINARY)
        thresh_otsu, mask_otsu = cv2.threshold(index_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask_man  = cv2.morphologyEx(mask_man,  cv2.MORPH_OPEN, kernel)
        mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel)

        mask_man[~valid]  = 255
        mask_otsu[~valid] = 255

        results[f"{name}_Manual"] = cv2.bitwise_not(mask_man)
        results[f"{name}_Otsu"]   = cv2.bitwise_not(mask_otsu)

        logger.info(f"  {name}_Manual threshold applied.")
        logger.info(f"  {name}_Otsu: threshold={thresh_otsu:.0f} (auto)")

    logger.info(f"Green index segmentation done. {len(results)} masks generated.")

    return results
