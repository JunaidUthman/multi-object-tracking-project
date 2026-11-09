"""Sélection interactive de la ligne de comptage via OpenCV."""

import os
from typing import Tuple, Optional
import cv2
import numpy as np


class LineDrawer:
    """Gestionnaire des événements de souris pour tracer la ligne de comptage."""

    def __init__(self, base_image: np.ndarray):
        self.base_image = base_image.copy()
        self.display_image = base_image.copy()
        self.points = []
        self.confirmed = False

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                self._redraw()

    def reset(self):
        self.points = []
        self.display_image = self.base_image.copy()

    def _redraw(self):
        self.display_image = self.base_image.copy()
        if len(self.points) >= 1:
            cv2.circle(self.display_image, self.points[0], 6, (0, 0, 255), -1)
            cv2.putText(
                self.display_image,
                "Point 1",
                (self.points[0][0] + 10, self.points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        if len(self.points) == 2:
            cv2.circle(self.display_image, self.points[1], 6, (0, 0, 255), -1)
            cv2.putText(
                self.display_image,
                "Point 2",
                (self.points[1][0] + 10, self.points[1][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.line(self.display_image, self.points[0], self.points[1], (0, 255, 255), 3)


def select_counting_line(
    first_frame: np.ndarray,
    default_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Ouvre une fenêtre interactive pour permettre à l'utilisateur de cliquer deux points.

    Args:
        first_frame: Première image de la vidéo.
        default_line: Ligne de secours si l'environnement n'a pas d'interface graphique.

    Returns:
        Tuple ((x1, y1), (x2, y2)) des deux extrémités de la ligne de comptage.
    """
    height, width = first_frame.shape[:2]

    # Ligne par défaut (milieu horizontal de l'image)
    if default_line is None:
        default_line = ((0, int(height * 0.6)), (width, int(height * 0.6)))

    # Vérification d'un environnement d'affichage graphique (X11 / Wayland)
    has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if not has_display:
        print(f"[INFO] Aucun écran graphique détecté ($DISPLAY non défini).")
        print(f"[INFO] Utilisation de la ligne par défaut : {default_line}")
        return default_line

    drawer = LineDrawer(first_frame)
    window_name = "Configuration - Cliquez 2 points pour la ligne de comptage"

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, width), min(720, height))
        cv2.setMouseCallback(window_name, drawer.on_mouse)

        print("\n--- CONFIGURATION INTERACTIVE DE LA LIGNE ---")
        print("1. Cliquez avec le bouton gauche pour définir le Point 1.")
        print("2. Cliquez une seconde fois pour définir le Point 2.")
        print("3. Appuyez sur [ENTREE] ou [ESPACE] pour valider.")
        print("4. Appuyez sur [R] pour recommencer le tracé.")
        print("5. Appuyez sur [ECHAP] pour utiliser la ligne par défaut.\n")

        while True:
            banner = drawer.display_image.copy()
            help_text = "Cliquez 2 points | 'r': Reset | ENTREE/ESPACE: Valider | ECHAP: Par defaut"
            cv2.rectangle(banner, (0, 0), (width, 40), (0, 0, 0), -1)
            cv2.putText(banner, help_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(window_name, banner)
            key = cv2.waitKey(20) & 0xFF

            # Validation
            if key in (13, 32):  # Enter ou Espace
                if len(drawer.points) == 2:
                    break
                else:
                    print("[ATTENTION] Veuillez sélectionner 2 points avant de valider.")

            # Reset
            elif key == ord('r') or key == ord('R'):
                drawer.reset()

            # Annulation / Échap
            elif key == 27:
                print(f"[INFO] Sélection annulée. Utilisation de la ligne par défaut : {default_line}")
                drawer.points = list(default_line)
                break

        cv2.destroyWindow(window_name)

        if len(drawer.points) == 2:
            pt1, pt2 = drawer.points[0], drawer.points[1]
            print(f"[INFO] Ligne de comptage configurée : {pt1} -> {pt2}")
            return (pt1, pt2)

    except cv2.error as e:
        print(f"[AVERTISSEMENT] Erreur lors de l'ouverture de la fenêtre graphique : {e}")
        print(f"[INFO] Utilisation de la ligne par défaut : {default_line}")

    return default_line
