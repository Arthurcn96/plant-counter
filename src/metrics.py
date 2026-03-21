"""
Gerar indicadores de performance e produtividade.
"""
import json
import logging
import numpy as np
import geopandas as gpd
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_stats(confirmed: list, gdf_polygons: gpd.GeoDataFrame, valid: np.ndarray, gsd: float, crs: str, output_path: Path) -> dict:
    """
    Calcula estatísticas do resultado da detecção e salva em JSON.
    """
    logger.info("Iniciando cálculo de estatísticas...")
    logger.debug(f"  Parâmetros | GSD={gsd:.4f}m, CRS={crs}, plantas confirmadas={len(confirmed)}")

    # Área total válida
    pixel_area_m2 = gsd ** 2
    total_pixels  = valid.sum()
    total_area_ha = (total_pixels * pixel_area_m2) / 10_000
    logger.debug(f"  Pixels válidos: {total_pixels} | Área por pixel: {pixel_area_m2:.6f} m² | Área total: {total_area_ha:.4f} ha")

    # Densidade de plantas
    total_plants  = len(confirmed)
    plants_per_ha = total_plants / total_area_ha if total_area_ha > 0 else 0
    logger.debug(f"  Densidade: {total_plants} plantas / {total_area_ha:.4f} ha = {plants_per_ha:.2f} plantas/ha")

    # Homogeneidade via polígonos
    if gdf_polygons is not None and len(gdf_polygons) > 0 and 'area_m2' in gdf_polygons.columns:
        areas_m2  = gdf_polygons['area_m2'].values
        mean_area = float(np.mean(areas_m2))
        std_area  = float(np.std(areas_m2))
        cv        = (std_area / mean_area * 100) if mean_area > 0 else 0
        logger.debug(f"  Polígonos: {len(areas_m2)} | Área média: {mean_area:.4f} m² | DP: {std_area:.4f} m² | CV: {cv:.2f}%")
    else:
        mean_area = std_area = cv = 0.0
        logger.warning("  Nenhum polígono disponível — métricas de homogeneidade definidas como zero.")

    interpretacao = "homogêneo" if cv < 25 else "heterogêneo"

    stats = {
        "total_plants":  total_plants,
        "area_ha":       round(total_area_ha, 4),
        "plants_per_ha": round(plants_per_ha, 2),
        "crs":           str(crs),
        "gsd_meters":    round(gsd, 6),
        "homogeneity": {
            "mean_crown_area_m2": round(mean_area, 4),
            "std_crown_area_m2":  round(std_area, 4),
            "cv_percent":         round(cv, 2),
            "interpretation":     interpretacao,
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"  Total de plantas : {total_plants}")
    logger.info(f"  Área válida      : {total_area_ha:.4f} ha")
    logger.info(f"  Densidade        : {plants_per_ha:.2f} plantas/ha")
    logger.info(f"  CV               : {cv:.2f}% ({interpretacao})")
    logger.info(f"  Estatísticas salvas em: {output_path}")

    return stats
