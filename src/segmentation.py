"""
Isola o verde das mudas do restante da imagem (solo).
"""
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)


def segmentation_RGB(rgb_image: np.ndarray, threshold: int = 80, kernel_size: int = 10):
    """
    Segmenta cada banda RGB da imagem usando threshold manual
    e aplica morfologia matemática (open) para limpeza.
    """
    logger.info(f"Iniciando segmentação RGB | threshold={threshold}, kernel_size={kernel_size}")
    band_names = ['Red (B1)', 'Green (B2)', 'Blue (B3)']

    if rgb_image.shape[0] == 3:
        rgb_image = np.moveaxis(rgb_image, 0, -1)

    h, w = rgb_image.shape[:2]
    logger.debug(f"Dimensões da imagem: {w}x{h}")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    valid  = np.any(rgb_image > 0, axis=-1)
    logger.debug(f"Pixels válidos: {valid.sum()} de {h * w} ({100 * valid.sum() / (h * w):.1f}%)")

    masks = {}

    for i, name in enumerate(band_names):
        band = rgb_image[:, :, i]
        mask_threshold = (band > threshold).astype(np.uint8) * 255
        opened = cv2.morphologyEx(mask_threshold, cv2.MORPH_OPEN, kernel)
        opened[~valid] = 255
        masks[name] = opened

        px_segmentados = (opened == 0).sum()
        logger.debug(f"  Banda {name}: {px_segmentados} pixels segmentados ({100 * px_segmentados / valid.sum():.1f}% dos válidos)")

    logger.info(f"Segmentação RGB concluída: {len(masks)} máscara(s) gerada(s).")
    return masks


def calculate_index(rgb_image: np.ndarray):
    """Calcula índices de vegetação comuns (ExG, Smolka, Vwg, LGI)."""
    logger.info("Calculando índices de vegetação...")
    r = rgb_image[:, :, 0].astype(float)
    g = rgb_image[:, :, 1].astype(float)
    b = rgb_image[:, :, 2].astype(float)

    g_safe     = np.where(g == 0, 1e-6, g)

    indices = {
        'ExG':    4 * g - r - b,
        'Smolka': (g - np.maximum(r, b) ** 2) / g_safe,
    }

    for name, idx in indices.items():
        logger.debug(f"  {name}: min={idx.min():.3f}, max={idx.max():.3f}, média={idx.mean():.3f}")

    logger.info(f"Índices calculados: {list(indices.keys())}")
    return indices


def segment_index(rgb_image: np.ndarray, threshold_manual: int = 130, kernel_size: int = 10):
    """
    Segmenta a imagem usando índices de vegetação (ExG e Smolka),
    aplicando threshold manual e Otsu com morfologia de limpeza.
    """
    logger.info(f"Iniciando segmentação por índice de verde | threshold_manual={threshold_manual}, kernel_size={kernel_size}")

    if rgb_image.ndim == 3 and rgb_image.shape[0] in (3, 4):
        rgb_image = np.moveaxis(rgb_image, 0, -1)
    rgb_image = rgb_image[:, :, :3]

    h, w = rgb_image.shape[:2]
    valid  = np.any(rgb_image > 0, axis=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    logger.debug(f"Dimensões: {w}x{h} | Pixels válidos: {valid.sum()} ({100 * valid.sum() / (h * w):.1f}%)")

    indices  = calculate_index(rgb_image)
    selected = {k: indices[k] for k in ('ExG', 'Smolka')}
    results  = {}

    for name, index in selected.items():
        logger.info(f"Segmentando índice: {name}")
        index_norm = cv2.normalize(index, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  #type: ignore[call-overload]

        if name == 'ExG':
            index_norm = cv2.bitwise_not(index_norm)
            logger.debug(f"  {name}: inversão de bits aplicada antes do threshold.")

        _, mask_man        = cv2.threshold(index_norm, threshold_manual, 255, cv2.THRESH_BINARY)
        thresh_otsu, mask_otsu = cv2.threshold(index_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask_man  = cv2.morphologyEx(mask_man,  cv2.MORPH_OPEN, kernel)
        mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel)

        mask_man[~valid]  = 255
        mask_otsu[~valid] = 255

        final_man  = cv2.bitwise_not(mask_man)
        final_otsu = cv2.bitwise_not(mask_otsu)

        px_man  = (final_man  == 255).sum()
        px_otsu = (final_otsu == 255).sum()
        logger.debug(f"  {name}_Manual : {px_man}  pixels positivos ({100 * px_man  / valid.sum():.1f}% dos válidos)")
        logger.debug(f"  {name}_Otsu   : {px_otsu} pixels positivos ({100 * px_otsu / valid.sum():.1f}% dos válidos) | threshold automático={thresh_otsu:.0f}")

        results[f"{name}_Manual"] = final_man
        results[f"{name}_Otsu"]   = final_otsu

    logger.info(f"Segmentação por índice concluída: {len(results)} máscara(s) gerada(s).")
    return results
