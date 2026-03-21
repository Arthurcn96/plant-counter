import logging
import sys
import numpy as np
from src.io_utils import (
    load_config, create_output_dir, read_tiff, save_image, plot_analysis,
    save_json, plot_masks, plot_confirmed_rejected, get_valid_mask
)
from src.preprocessing import preprocess_image
from src.segmentation import segment_index, segmentation_RGB
from src.detection import detect_plants, vote_points, contours_from_masks
from src.georef import export_points_geojson, export_polygons_geojson
from src.metrics import calculate_stats


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def main():
    try:
        # Carrega configuração
        config = load_config("config.yaml")
        path_cfg  = config['paths']
        pre_cfg   = config['preprocessing']
        seg_cfg   = config['segmentation']
        detec_cfg = config['detection']

        # Cria diretório de saída
        output_dir = create_output_dir(path_cfg['output_base'], path_cfg['output_prefix'])

        # Lê a imagem e metadados
        image, meta = read_tiff(path_cfg['input_tif'])
        valid    = get_valid_mask(image)

        # Pré-processamento
        preprocessed_img = preprocess_image(image, pre_cfg)

        # Segmentação
        logger.info("Iniciando segmentação...")
        band_masks = segmentation_RGB(
            image,
            threshold=seg_cfg['bands']['threshold'],
            kernel_size=seg_cfg['bands']['kernel_size']
        )
        green_masks = segment_index(
            preprocessed_img,
            threshold_manual=seg_cfg['green_index']['threshold_manual'],
            kernel_size=seg_cfg['green_index']['kernel_size']
        )

        # Detecção e votação — RGB
        logger.info("Iniciando detecção de plantas e votação...")
        detected_rgb            = detect_plants(band_masks, detec_cfg['min_area'], detec_cfg['max_area'])
        confirmed_rgb, rejected = vote_points(detected_rgb, detec_cfg['eps'], detec_cfg['min_votos'])

        plot_confirmed_rejected(
            image,
            confirmed_rgb,
            rejected,
            show_rejected=detec_cfg['show_rejected'],
            output_path=output_dir / "imagens/final_detections_rgb.jpg"
        )

        # Detecção e votação — Índice verde
        detected_green            = detect_plants(green_masks, detec_cfg['min_area'], detec_cfg['max_area'])
        confirmed_green, rejected = vote_points(detected_green, detec_cfg['eps'], detec_cfg['min_votos'])

        plot_confirmed_rejected(
            image,
            confirmed_green,
            rejected,
            show_rejected=detec_cfg['show_rejected'],
            output_path=output_dir / "imagens/final_detections_green.jpg"
        )

        # Detecção combinada para extração de contornos
        all_masks     = {**band_masks, **green_masks}
        contours_list = contours_from_masks(all_masks)

        # Exportação geoespacial
        export_points_geojson(
            confirmed_rgb, meta['transform'], meta['crs'], valid,
            output_dir / "plantas.geojson"
        )
        gdf_polygons = export_polygons_geojson(
            contours_list, meta['transform'], meta['crs'], meta['gsd'],
            output_dir / "plantas_poligonos.geojson"
        )

        # Salva imagens e plots
        original_hwc = np.moveaxis(image, 0, -1)
        save_image(original_hwc,     output_dir / "imagens/original.jpg")
        save_image(preprocessed_img, output_dir / "imagens/preprocessed.jpg")

        plot_masks(band_masks,  "Segmentação por Bandas RGB (Limiar Manual)",       output_dir / "imagens/plot_bands.png",         cols=3)
        plot_masks(green_masks, "Segmentação por Índices de Verde (Limiar Otsu)",   output_dir / "imagens/plot_green_indices.png", cols=2)

        # Salva metadados e configuração
        save_json(meta,   output_dir / "metadata.json")
        save_json(config, output_dir / "config_used.json")

        confirmed_final = confirmed_rgb

        # Estatísticas
        calculate_stats(
            confirmed    = confirmed_final,
            gdf_polygons = gdf_polygons,
            valid        = valid,
            gsd          = meta['gsd'],
            crs          = meta['crs'],
            output_path  = output_dir / "stats.json"
        )

        # Análise
        logger.info(f"Processo finalizado. Resultados em: {output_dir}")
        plot_analysis(
            confirmed    = confirmed_final,
            gdf_polygons = gdf_polygons,
            image        = image,
            gsd          = meta['gsd'],
            output_path  = output_dir / "analysis.png"
        )

        logger.info(f"Processo finalizado. Resultados em: {output_dir}")

    except Exception as e:
        logger.error(f"Falha na execução: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
