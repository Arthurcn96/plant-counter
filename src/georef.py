"""
Converter coordenadas de imagem em coordenadas geográficas reais.
"""
import logging
import json
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, Polygon
from pyproj import Transformer
from rasterio.transform import xy as rasterio_xy
import rasterio

logger = logging.getLogger(__name__)


def pixels_to_geo(
    cx: int, cy: int,
    transform,
    transformer: Transformer,
) -> tuple[float, float]:
    """
    Converte coordenadas de pixel (cx, cy) para (lon, lat) em WGS84.
    """
    x, y = rasterio_xy(transform, cy, cx)
    lon, lat = transformer.transform(x, y)
    return lon, lat


def export_points_geojson(
    confirmed: list,
    transform,
    crs,
    valid: np.ndarray,
    output_path: Path,
) -> gpd.GeoDataFrame:
    """
    Exporta pontos confirmados como GeoJSON georreferenciado.
    """
    logger.info("Exporting points GeoJSON...")

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    h, w = valid.shape

    features = []
    skipped  = 0

    for cx, cy in confirmed:
        if not (0 <= cx < w and 0 <= cy < h):
            skipped += 1
            continue
        if not valid[cy, cx]:
            skipped += 1
            continue

        lon, lat = pixels_to_geo(cx, cy, transform, transformer)

        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            skipped += 1
            continue

        features.append({
            'geometry': Point(lon, lat),
            'pixel_x':  cx,
            'pixel_y':  cy,
            'lon':      round(lon, 8),
            'lat':      round(lat, 8),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    gdf.to_file(str(output_path), driver="GeoJSON")

    logger.info(f"  Exported: {len(gdf)} points | Skipped: {skipped}")
    return gdf


def export_polygons_geojson(
    contours_list: list,
    transform,
    crs,
    gsd: float,
    output_path: Path,
) -> gpd.GeoDataFrame:

    logger.info("Exporting polygons GeoJSON...")
    logger.info(f"  Contours received: {len(contours_list)}")

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    features = []
    skipped  = 0

    for item in contours_list:
        contour = item['contour']
        area_px = item['area_px']
        source  = item['source']

        coords_geo = []
        for pt in contour[:, 0]:
            cx, cy   = int(pt[0]), int(pt[1])
            lon, lat = pixels_to_geo(cx, cy, transform, transformer)
            coords_geo.append((lon, lat))

        if len(coords_geo) < 3:
            skipped += 1
            continue

        features.append({
            'geometry': Polygon(coords_geo),
            'area_m2':  round(area_px * gsd ** 2, 4),
            'source':   source,
        })

    logger.info(f"  Features geradas: {len(features)} | Skipped: {skipped}")

    # Retorna GeoDataFrame vazio se não houver features
    if not features:
        logger.warning("Nenhum polígono gerado — retornando GeoDataFrame vazio.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(features, geometry='geometry', crs="EPSG:4326")
    gdf.to_file(str(output_path), driver="GeoJSON")

    logger.info(f"  Exported: {len(gdf)} polygons")
    return gdf
