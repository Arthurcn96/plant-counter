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
    logger.info("Calculating statistics...")

    # Área total da imagem em hectares (só pixels válidos)
    pixel_area_m2 = gsd ** 2
    total_pixels  = valid.sum()
    total_area_ha = (total_pixels * pixel_area_m2) / 10_000

    # Total de plantas — usa confirmed como fonte principal
    total_plants  = len(confirmed)
    plants_per_ha = total_plants / total_area_ha if total_area_ha > 0 else 0

    # Homogeneidade — usa áreas reais dos polígonos se disponível
    if gdf_polygons is not None and len(gdf_polygons) > 0 and 'area_m2' in gdf_polygons.columns:
        areas_m2  = gdf_polygons['area_m2'].values
        mean_area = float(np.mean(areas_m2))
        std_area  = float(np.std(areas_m2))
        cv        = (std_area / mean_area * 100) if mean_area > 0 else 0
        logger.info(f"  Homogeneity based on {len(areas_m2)} polygons")
    else:
        mean_area = 0.0
        std_area  = 0.0
        cv        = 0.0
        logger.warning("  No polygons available — homogeneity metrics set to 0")

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
            "interpretation":     "homogeneous" if cv < 25 else "heterogeneous"
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"  Total plants:  {total_plants}")
    logger.info(f"  Area:          {total_area_ha:.4f} ha")
    logger.info(f"  Density:       {plants_per_ha:.2f} plants/ha")
    logger.info(f"  CV:            {cv:.2f}% ({stats['homogeneity']['interpretation']})")
    logger.info(f"  Stats saved to {output_path}")

    return stats
