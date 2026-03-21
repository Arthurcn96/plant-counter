"""
Centralizar a leitura e escrita de arquivos
"""
import rasterio
import logging
import json
import yaml
import cv2
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """
    Loads YAML configuration file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config

def create_output_dir(base_path, prefix):
    """
    Creates a unique output directory like output/exp1, output/exp2...
    """
    base_dir = Path(base_path)
    base_dir.mkdir(exist_ok=True)

    i = 1
    while True:
        run_dir = base_dir / f"{prefix}{i}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True)

            # cria a subpasta "image"
            image_dir = run_dir / "imagens"
            image_dir.mkdir()

            logger.info(f"Results directory created: {run_dir}")
            return run_dir
        i += 1

def read_tiff(file_path: str):
    """
    Reads georeferenced TIFF and logs profile.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    logger.info(f"Reading TIF: {path}")

    with rasterio.open(path) as src:
        # Lê as 3 primeiras bandas (RGB) — ignora NIR se existir
        image = src.read([1, 2, 3])

        gsd     = abs(src.transform.a)
        unidade = "degrees" if src.crs and src.crs.is_geographic else "meters"

        area_ha = None
        if src.crs and src.crs.is_projected:
            area_ha = round(src.width * src.height * gsd ** 2 / 10_000, 4)

        meta = {
            "driver":    src.driver,
            "width":     src.width,
            "height":    src.height,
            "bands":     src.count,
            "dtype":     str(src.dtypes[0]),
            "crs":       str(src.crs),
            "transform": src.transform,
            "bounds":    src.bounds,
            "gsd":       round(gsd, 6),
            "gsd_unit":  unidade,
            "area_ha":   area_ha,
        }

        logger.info(f"  Dimensions:  {meta['width']} x {meta['height']} px")
        logger.info(f"  Bands:       {meta['bands']}")
        logger.info(f"  CRS:         {meta['crs']}")
        logger.info(f"  GSD:         {meta['gsd']} {meta['gsd_unit']}/px")
        if area_ha:
            logger.info(f"  Area:        {area_ha} ha")
    return image, meta

def save_image(image: np.ndarray, output_path: Path):
    """
    Saves HWC RGB image as BGR via OpenCV.
    """
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_image)
    logger.info(f"Image saved to {output_path}")


def save_json(data: dict, output_path: Path):
    """
    Saves dict as JSON with basic cleaning for non-serializable objects."""
    clean_data = {}
    for k, v in data.items():
        if k == 'crs': clean_data[k] = str(v)
        elif k == 'bounds': clean_data[k] = list(v)
        elif k == 'transform': clean_data[k] = [v.a, v.b, v.c, v.d, v.e, v.f]
        else: clean_data[k] = v

    with open(output_path, 'w') as f:
        json.dump(clean_data, f, indent=4)
    logger.info(f"Metadata saved to {output_path}")

def plot_masks(masks_dict: dict, title: str, output_path: Path, cols: int = 3):
    """
    Plots a grid of binary masks or images.
    """
    n = len(masks_dict)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (name, mask) in zip(axes, masks_dict.items()):
        ax.imshow(mask, cmap='gray')
        ax.set_title(name)
        ax.axis('off')

    # Hide empty subplots
    for ax in axes[n:]:
        ax.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Plot saved to {output_path}")

def draw_detections(image: np.ndarray, centroids: list, output_path: Path):
    """
    Draws dots on the image at each centroid and saves it.
    Expects image in HWC RGB format.
    """
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)

    # Normalize to uint8 if necessary
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert to BGR for OpenCV drawing
    output_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for pt in centroids:
        center = (int(pt['x']), int(pt['y']))
        cv2.circle(output_img, center, 3, (0, 0, 255), -1) # Red dots

    cv2.imwrite(str(output_path), output_img)
    logger.info(f"Detections visualized in {output_path}")

def plot_detections_grid(image: np.ndarray, detected_points: dict, output_path: Path):
    """
    Plots a grid (2x2) of the original image dimmed,
    with red dots for detected plant centroids.
    """
    # Ensure HWC RGB
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)

    # Normalize to uint8 if necessary
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Dim the image for better visualization of dots
    img_display = np.clip(image.astype(float) * 0.5, 0, 255).astype(np.uint8)

    combos = list(detected_points.keys())
    n = len(combos)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 14))
    axes = axes.flatten()

    for i, name in enumerate(combos):
        ax = axes[i]
        points = detected_points[name]

        ax.imshow(img_display)
        if points:
            x_coords = [p['x'] for p in points]
            y_coords = [p['y'] for p in points]
            ax.scatter(x_coords, y_coords, s=20, color='red', alpha=0.7)

        ax.set_title(f"{name} — {len(points)} plantas", fontsize=14)
        ax.axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Final Detection Comparison", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Detections grid plot saved to {output_path}")

def plot_confirmed_rejected(image: np.ndarray, confirmed: list, rejected: list, show_rejected: bool, output_path: Path):
    """
    Plots confirmed (cyan) and optionally rejected (red) points on the original image.
    Saves the final result to the output_path.
    """
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)

    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(np.clip(image * 0.7, 0, 255).astype(np.uint8))

    # Confirmed
    if confirmed:
        xs = [p[0] for p in confirmed]
        ys = [p[1] for p in confirmed]
        ax.scatter(xs, ys, s=30, color='cyan', alpha=0.9,
                   label=f'Confirmadas ({len(confirmed)})')

    # Rejected
    if show_rejected and rejected:
        xs = [p[0] for p in rejected]
        ys = [p[1] for p in rejected]
        ax.scatter(xs, ys, s=20, color='red', alpha=0.6,
                   label=f'Rejeitadas ({len(rejected)})')

    ax.legend(loc='lower right', fontsize=11, markerscale=2,
              facecolor='black', labelcolor='white')
    ax.set_title(
        f"Detecção final — {len(confirmed)} confirmadas"
        + (f" | {len(rejected)} rejeitadas" if show_rejected else ""),
        fontsize=13
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Final plot (Confirmed={len(confirmed)}, Rejected={len(rejected) if show_rejected else 'hidden'}) saved to {output_path}")

def get_valid_mask(image: np.ndarray) -> np.ndarray:
    """
    Retorna máscara de pixels válidos.
    """
    if image.shape[0] in (3, 4):
        image = np.moveaxis(image, 0, -1)
    return np.any(image > 0, axis=-1)


def plot_analysis(
    confirmed: list,
    gdf_polygons: gpd.GeoDataFrame,
    image: np.ndarray,
    gsd: float,
    output_path: Path,
):
    """
    Gera gráficos de análise final da detecção.
    """
    from scipy.stats import gaussian_kde
    from scipy.spatial import KDTree

    logger.info(f"Iniciando analise dos dados...")
    # Garante formato HWC
    if image.shape[0] in (3, 4):
        image = np.moveaxis(image, 0, -1)
    image = image[:, :, :3]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- 1. Histograma de áreas das copas ---
    if gdf_polygons is not None and len(gdf_polygons) > 0:
        areas = gdf_polygons['area_m2'].values
        axes[0].hist(areas, bins=30, color='green', alpha=0.7)
        axes[0].axvline(areas.mean(), color='red', linestyle='--',
                        label=f'média={areas.mean():.2f}m²')
        axes[0].set_xlabel('Área (m²)')
        axes[0].set_ylabel('Frequência')
        axes[0].set_title('Distribuição das áreas das copas')
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'Sem polígonos disponíveis',
                     ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Distribuição das áreas das copas')

    # --- 2. Mapa de densidade ---
    xs = [p[0] for p in confirmed]
    ys = [p[1] for p in confirmed]

    axes[1].imshow(np.clip(image * 0.5, 0, 255).astype(np.uint8))
    if len(xs) > 1:
        xy_stack     = np.vstack([xs, ys])
        kde          = gaussian_kde(xy_stack)
        density      = kde(xy_stack)
        density_norm = (density - density.min()) / (density.max() - density.min() + 1e-6)
        sc = axes[1].scatter(xs, ys, c=density_norm, cmap='hot', s=30, alpha=0.8, vmin=0, vmax=1)
        plt.colorbar(sc, ax=axes[1], label='Densidade relativa')
    axes[1].set_title("Mapa de densidade de plantas")
    axes[1].axis('off')

    # --- 3. Histograma de espaçamento entre vizinhos ---
    if len(confirmed) > 1:
        pts      = np.array(confirmed)[:, :2]
        tree     = KDTree(pts)
        dists, _ = tree.query(pts, k=2)
        nn_dists = dists[:, 1] * gsd

        axes[2].hist(nn_dists, bins=30, color='steelblue', alpha=0.7)
        axes[2].axvline(nn_dists.mean(), color='red', linestyle='--',
                        label=f'média={nn_dists.mean():.2f}m')
        axes[2].set_xlabel('Distância ao vizinho mais próximo (m)')
        axes[2].set_ylabel('Frequência')
        axes[2].set_title('Espaçamento entre plantas')
        axes[2].legend()

        logger.info(f"  Espaçamento médio:  {nn_dists.mean():.2f}m")
        logger.info(f"  Espaçamento mínimo: {nn_dists.min():.2f}m")
        logger.info(f"  Espaçamento máximo: {nn_dists.max():.2f}m")
    else:
        axes[2].text(0.5, 0.5, 'Pontos insuficientes',
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Espaçamento entre plantas')

    plt.suptitle("Análise final da detecção", fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Analysis plot saved to {output_path}")
