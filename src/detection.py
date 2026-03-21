"""
Contar e localizar as mudas individualmente.
"""
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)


def detect_plants(masks: dict, min_area: int = 60, max_area: int = 3000) -> dict:
    """
    Detecta centroides de plantas a partir de um dicionário de máscaras binárias.
    """
    logger.info(f"Iniciando detecção de plantas | {len(masks)} máscara(s), área válida: [{min_area}, {max_area}] px²")

    detected_points = {}

    for name, mask in masks.items():
        inverted = cv2.bitwise_not(mask)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            inverted, connectivity=8
        )

        total_components = num_labels - 1  # exclui o background
        points    = []
        rejected  = 0

        for j in range(1, num_labels):
            area = stats[j, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                cx, cy = int(centroids[j][0]), int(centroids[j][1])
                points.append((cx, cy))
            else:
                rejected += 1

        detected_points[name] = points
        logger.debug(f"  {name}: {total_components} componentes | {len(points)} aceitos | {rejected} rejeitados por área")

    total = sum(len(v) for v in detected_points.values())
    logger.info(f"Detecção concluída: {total} ponto(s) detectado(s) no total.")
    return detected_points


def vote_points(detected_points: dict, eps: int = 15, min_votes: int = 2) -> tuple[list, list]:
    """
    Compara pontos detectados em múltiplas máscaras e retorna apenas os
    que aparecem em pelo menos min_votes fontes diferentes.
    """
    logger.info(f"Iniciando votação | eps={eps}px, min_votes={min_votes}")

    all_points  = []
    all_sources = []

    for name, points in detected_points.items():
        all_points.extend(points)
        all_sources.extend([name] * len(points))

    logger.debug(f"  Total de pontos para votação: {len(all_points)} em {len(detected_points)} fonte(s)")

    confirmed = []
    rejected  = []
    used      = set()

    for i, (pi, si) in enumerate(zip(all_points, all_sources)):
        if i in used:
            continue

        neighbors        = []
        neighbor_sources = {si}

        for j, (pj, sj) in enumerate(zip(all_points, all_sources)):
            if i == j or j in used:
                continue
            dist = np.sqrt((pi[0] - pj[0])**2 + (pi[1] - pj[1])**2)
            if dist <= eps:
                neighbors.append(j)
                neighbor_sources.add(sj)

        if len(neighbor_sources) >= min_votes:
            group = [pi] + [all_points[j] for j in neighbors]
            cx = int(np.mean([p[0] for p in group]))
            cy = int(np.mean([p[1] for p in group]))
            confirmed.append((cx, cy))
            used.add(i)
            used.update(neighbors)
        else:
            rejected.append(pi)
            used.add(i)

    taxa = len(confirmed) / (len(confirmed) + len(rejected)) * 100 if all_points else 0
    logger.info(f"Votação concluída | confirmados: {len(confirmed)} | rejeitados: {len(rejected)} | taxa de aprovação: {taxa:.1f}%")
    return confirmed, rejected


def contours_from_masks(masks: dict, min_area: int = 60, max_area: int = 3000) -> list:
    """
    Extrai contornos de plantas a partir de um dicionário de máscaras binárias.
    """
    logger.info(f"Extraindo contornos | {len(masks)} máscara(s), área válida: [{min_area}, {max_area}] px²")

    contours_list = []

    for name, mask in masks.items():
        inverted = cv2.bitwise_not(mask)
        contours, _ = cv2.findContours(
            inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        count    = 0
        rejected = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                contours_list.append({
                    'contour': contour,
                    'area_px': area,
                    'source':  name,
                })
                count += 1
            else:
                rejected += 1

        logger.debug(f"  {name}: {count} contorno(s) aceito(s) | {rejected} rejeitado(s) por área")

    logger.info(f"Extração concluída: {len(contours_list)} contorno(s) no total.")
    return contours_list
