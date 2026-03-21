import logging
import sys
import numpy as np
from src.io_utils import (
    load_config, create_output_dir, read_tiff, save_image, get_valid_mask,
    save_json, plot_masks, plot_confirmed_rejected, load_meta
)
from src.preprocessing import preprocess_image
from src.segmentation import segment_index, segmentation_RGB
from src.detection import detect_plants, vote_points, contours_from_masks
from src.georef import export_points_geojson, export_polygons_geojson
from src.metrics import calculate_stats



# Global logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def main():
    try:
        # 1. Load Configuration
        config = load_config("config.yaml")
        path_cfg = config['paths']
        pre_cfg = config['preprocessing']
        seg_cfg = config['segmentation']
        detec_cfg = config['detection']

        # 2. Setup Output Directory
        output_dir = create_output_dir(path_cfg['output_base'], path_cfg['output_prefix'])

        # 3. Read Input
        image, meta = read_tiff(path_cfg['input_tif'])

        # 4. Preprocessing
        preprocessed_img = preprocess_image(image, pre_cfg)

        # 5. Segmentation
        logger.info("Starting segmentation...")
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

        # 6. Detection & Voting
        logger.info("Starting plant detection and voting...")
        detected_rgb   = detect_plants(band_masks, detec_cfg['min_area'], detec_cfg['max_area'])
        detected_green = detect_plants(green_masks, detec_cfg['min_area'], detec_cfg['max_area'])

        # Realizando votação combinada
        all_detections_rgb = {**detected_rgb}
        confirmed_rgb, rejected = vote_points(all_detections_rgb, detec_cfg['eps'], detec_cfg['min_votos'])

        # Plota os resultados finais
        plot_confirmed_rejected(
            image,
            confirmed_rgb,
            rejected,
            show_rejected=detec_cfg['show_rejected'],
            output_path=output_dir / "image/final_detections_rgb.jpg"
        )

        # Realizando votação combinada
        all_detections_green = {**detected_green}
        confirmed_green, rejected = vote_points(all_detections_green, detec_cfg['eps'], detec_cfg['min_votos'])

        # Plota os resultados finais
        plot_confirmed_rejected(
            image,
            confirmed_green,
            rejected,
            show_rejected=detec_cfg['show_rejected'],
            output_path=output_dir / "image/final_detections_green.jpg"
        )

        # Detecta contornos
        # Poderia gerar 2 contornos, mas no final apenas um detalhe.
        all_masks      = {**band_masks, **green_masks}
        contours_list  = contours_from_masks(all_masks)

        detected       = detect_plants(all_masks)
        confirmed, _   = vote_points(detected)
        contours_list  = contours_from_masks(all_masks)

        meta = load_meta(path_cfg['input_tif'])
        valid       = get_valid_mask(image)

        _ = export_points_geojson(confirmed, meta['transform'], meta['crs'], valid, output_dir / "plantas.geojson")
        gdf_polygons = export_polygons_geojson(contours_list, meta['transform'], meta['crs'], meta['gsd'], output_dir / "plantas_poligonos.geojson")

        # 7. Saving Other Results
        original_hwc = np.moveaxis(image, 0, -1)
        save_image(original_hwc, output_dir / "image/original.jpg")
        save_image(preprocessed_img, output_dir / "image/preprocessed.jpg")

        plot_masks(band_masks, "Segmentation by RGB Bands (Manual Threshold)", output_dir / "image/plot_bands.png", cols=3)
        plot_masks(green_masks, "Segmentation by Green Indices (Otsu Threshold)", output_dir / "image/plot_green_indices.png", cols=2)

        save_json(meta, output_dir / "metadata.json")
        save_json(config, output_dir / "config_used.json")
        # save_json({"counts": {"confirmed": len(confirmed_rgb), "rejected": len(rejected)}}, output_dir / "summary_counts_rgb.json")
        # save_json({"counts": {"confirmed": len(confirmed_green), "rejected": len(rejected)}}, output_dir / "summary_counts_green.json")
        #


        # TODO: Adicionar a possibilidade de trocar entre qual confirmado escolher
        _ = calculate_stats(
            confirmed    = confirmed_rgb,
            gdf_polygons = gdf_polygons,
            valid        = valid,
            gsd          = meta['gsd'],
            crs          = meta['crs'],
            output_path  = output_dir / "stats.json"
        )

        logger.info(f"Process finished. Results in: {output_dir}")

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
