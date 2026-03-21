"""
Trata a imagem para mitigar sombras e variações de iluminação.
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def normalize_image(image_chw: np.ndarray, p_lower=2, p_upper=98) -> np.ndarray:
    """
    Normaliza as bandas usando percentis especificados.
    """

    logger.info(f"Normalizando imagem com percentis p{p_lower}-p{p_upper}...")
    img = np.moveaxis(image_chw, 0, -1).astype(np.float32)
    normalized = np.zeros_like(img, dtype=np.uint8)
    n_bands = img.shape[2]

    for i in range(n_bands):
        band = img[:, :, i]
        mask = band > 0
        valid_px = np.sum(mask)

        if not np.any(mask):
            logger.warning(f"Banda {i} ignorada: todos os pixels são zero.")
            continue

        p2, p98 = np.percentile(band[mask], [p_lower, p_upper])
        band_norm = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
        normalized[:, :, i] = band_norm.astype(np.uint8)
        logger.debug(f"  Banda {i}: {valid_px} pixels válidos | p{p_lower}={p2:.2f}, p{p_upper}={p98:.2f}")

    logger.info(f"Normalização concluída: {n_bands} banda(s) processada(s).")
    return normalized


def apply_clahe(img_hwc: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Aplica CLAHE com parâmetros dinâmicos.
    """
    grid = tuple(tile_grid_size)
    n_channels = img_hwc.shape[2]
    logger.info(f"Aplicando CLAHE: limite={clip_limit}, tile={grid}, canais={n_channels}...")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    channels = []
    for i in range(n_channels):
        ch = clahe.apply(img_hwc[:, :, i])
        before = img_hwc[:, :, i].mean()
        after  = ch.mean()
        logger.debug(f"  Canal {i}: brilho médio {before:.2f} → {after:.2f}")
        channels.append(ch)

    result = np.stack(channels, axis=-1)
    logger.info("CLAHE concluído.")
    return result


def preprocess_image(image_chw: np.ndarray, config: dict) -> np.ndarray:
    """
    Pipeline de pré-processamento configurado via dicionário.
    """
    h, w = image_chw.shape[1], image_chw.shape[2]
    n_bands = image_chw.shape[0]
    logger.info(f"Iniciando pré-processamento | imagem: {w}x{h}, {n_bands} banda(s).")

    norm_cfg  = config.get('normalization', {})
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

    logger.info("Pré-processamento finalizado.")
    return img_pre
