"""Fonctions géométriques pour le calcul de trajectoires et détection de franchissements."""

from typing import Tuple, Optional


def compute_bbox_center(box: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """Calcule le centre géométrique d'une bounding box (x1, y1, x2, y2).

    Args:
        box: Coordonnées (x1, y1, x2, y2)

    Returns:
        Tuple (cx, cy) des coordonnées entières du centre.
    """
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    return (cx, cy)


def _ccw(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> bool:
    """Vérifie si les points a, b, c sont orientés dans le sens trigonométrique (anti-horaire)."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    q1: Tuple[int, int],
    q2: Tuple[int, int]
) -> bool:
    """Détermine si deux segments [p1, p2] et [q1, q2] se croisent.

    Args:
        p1: Première extrémité du premier segment (ex: position précédente de l'objet)
        p2: Seconde extrémité du premier segment (ex: position actuelle de l'objet)
        q1: Première extrémité du second segment (ex: début ligne virtuelle)
        q2: Seconde extrémité du second segment (ex: fin ligne virtuelle)

    Returns:
        True si les segments se coupent, False sinon.
    """
    return (
        _ccw(p1, q1, q2) != _ccw(p2, q1, q2)
        and _ccw(p1, p2, q1) != _ccw(p1, p2, q2)
    )


def point_side_of_line(
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
    point: Tuple[int, int]
) -> float:
    """Calcule le produit vectoriel pour déterminer de quel côté de la ligne orientée se trouve le point.

    Valeur > 0 : à 'gauche' de la ligne orientée (Sens A)
    Valeur < 0 : à 'droite' de la ligne orientée (Sens B)
    Valeur = 0 : colinéaire
    """
    x1, y1 = line_start
    x2, y2 = line_end
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def check_crossing_direction(
    prev_pt: Tuple[int, int],
    curr_pt: Tuple[int, int],
    line_start: Tuple[int, int],
    line_end: Tuple[int, int]
) -> Optional[str]:
    """Vérifie le franchissement d'une ligne entre deux positions successives.

    Returns:
        'IN' si franchissement dans le sens A -> B
        'OUT' si franchissement dans le sens B -> A
        None si aucun croisement
    """
    if not segments_intersect(prev_pt, curr_pt, line_start, line_end):
        return None

    side_prev = point_side_of_line(line_start, line_end, prev_pt)
    side_curr = point_side_of_line(line_start, line_end, curr_pt)

    # Franchissement d'un côté vers l'autre
    if side_prev < 0 and side_curr >= 0:
        return "IN"
    elif side_prev > 0 and side_curr <= 0:
        return "OUT"

    return "IN"  # Traversée validée par défaut
