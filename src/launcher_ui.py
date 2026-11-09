"""Interface graphique moderne pour la sélection des algorithmes de Détection et de Suivi."""

import os
import sys
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


class LauncherDialog:
    """Boîte de dialogue interactive et moderne pour configurer le pipeline."""

    def __init__(
        self,
        default_model: str = "yolov8l.pt",
        default_tracker: str = "configs/custom_botsort.yaml"
    ):
        self.selected_model = default_model
        self.selected_tracker = default_tracker
        self.confirmed = False
        self.canceled = False

        # Dimensions de la fenêtre
        self.width = 720
        self.height = 560

        # Définition des options disponibles
        self.detector_options = [
            {
                "id": "yolov8n.pt",
                "title": "YOLOv8n (Nano)",
                "desc": "Ultra-rapide (6 Mo) - Usage standard / CPU",
                "badge": "Rapide"
            },
            {
                "id": "yolov8l.pt",
                "title": "YOLOv8l (Large)",
                "desc": "Haute précision (84 Mo) - Robuste aux foules et occultations",
                "badge": "Précis (Recommandé)"
            }
        ]

        self.tracker_options = [
            {
                "id": "configs/custom_botsort.yaml",
                "title": "BoT-SORT (Re-ID Optimisé)",
                "desc": "Re-ID visuel (vêtements) + mémoire 90f - Anti ID Switches",
                "badge": "Recommandé"
            },
            {
                "id": "configs/custom_bytetrack.yaml",
                "title": "ByteTrack (Cinématique)",
                "desc": "Filtre de Kalman + mémoire 90f (léger et fluide)",
                "badge": "Rapide"
            }
        ]

        # Zones cliquables (bounding boxes) [x1, y1, x2, y2]
        self.hitboxes: Dict[str, Tuple[int, int, int, int]] = {}
        self.hovered_key: Optional[str] = None

    def on_mouse(self, event, x, y, flags, param):
        """Gère le survol et les clics souris sur les composants de l'UI."""
        # Détection du survol
        current_hover = None
        for key, (x1, y1, x2, y2) in self.hitboxes.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                current_hover = key
                break
        self.hovered_key = current_hover

        # Clic gauche
        if event == cv2.EVENT_LBUTTONDOWN:
            for key, (x1, y1, x2, y2) in self.hitboxes.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if key.startswith("det_"):
                        self.selected_model = key.replace("det_", "")
                    elif key.startswith("trk_"):
                        self.selected_tracker = key.replace("trk_", "")
                    elif key == "btn_start":
                        self.confirmed = True
                    elif key == "btn_cancel":
                        self.canceled = True
                    break

    def render(self) -> np.ndarray:
        """Génère l'image de l'interface avec un thème sombre moderne."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (28, 22, 20)  # Fond sombre élégant (Dark slate)

        # 1. En-tête / Header avec dégradé subtil
        cv2.rectangle(canvas, (0, 0), (self.width, 80), (45, 34, 30), -1)
        cv2.line(canvas, (0, 80), (self.width, 80), (85, 65, 55), 1)

        cv2.putText(
            canvas,
            "CONFIGURATION DU PIPELINE VIDEO",
            (30, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            "Choisissez vos algorithmes de detection IA et de suivi de trajectoire",
            (30, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 170, 160),
            1,
            cv2.LINE_AA
        )

        # 2. Section 1 : Algorithmes de Détection
        det_y = 105
        cv2.putText(
            canvas,
            "1. ALGORITHME DE DETECTION (YOLO)",
            (30, det_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA
        )

        card_w = self.width - 60
        card_h = 60
        start_y = det_y + 15

        for i, opt in enumerate(self.detector_options):
            cy = start_y + i * (card_h + 10)
            is_selected = (self.selected_model == opt["id"])
            box_key = f"det_{opt['id']}"
            self.hitboxes[box_key] = (30, cy, 30 + card_w, cy + card_h)

            self._draw_radio_card(
                canvas,
                x=30,
                y=cy,
                w=card_w,
                h=card_h,
                title=opt["title"],
                desc=opt["desc"],
                badge=opt["badge"],
                is_selected=is_selected,
                is_hovered=(self.hovered_key == box_key)
            )

        # 3. Section 2 : Algorithmes de Suivi (Tracking)
        trk_y = start_y + 2 * (card_h + 10) + 20
        cv2.putText(
            canvas,
            "2. ALGORITHME DE SUIVI (TRACKER)",
            (30, trk_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA
        )

        start_trk_y = trk_y + 15
        for i, opt in enumerate(self.tracker_options):
            cy = start_trk_y + i * (card_h + 10)
            is_selected = (self.selected_tracker == opt["id"])
            box_key = f"trk_{opt['id']}"
            self.hitboxes[box_key] = (30, cy, 30 + card_w, cy + card_h)

            self._draw_radio_card(
                canvas,
                x=30,
                y=cy,
                w=card_w,
                h=card_h,
                title=opt["title"],
                desc=opt["desc"],
                badge=opt["badge"],
                is_selected=is_selected,
                is_hovered=(self.hovered_key == box_key)
            )

        # 4. Boutons d'Action (En bas)
        btn_y = self.height - 75
        btn_start_w = 320
        btn_start_h = 45
        btn_start_x = self.width - 30 - btn_start_w

        self.hitboxes["btn_start"] = (btn_start_x, btn_y, btn_start_x + btn_start_w, btn_y + btn_start_h)
        is_btn_hover = (self.hovered_key == "btn_start")

        # Bouton Démarrer (vert / émeraude vif)
        btn_color = (70, 190, 80) if not is_btn_hover else (90, 220, 100)
        cv2.rectangle(
            canvas,
            (btn_start_x, btn_y),
            (btn_start_x + btn_start_w, btn_y + btn_start_h),
            btn_color,
            -1
        )
        cv2.rectangle(
            canvas,
            (btn_start_x, btn_y),
            (btn_start_x + btn_start_w, btn_y + btn_start_h),
            (140, 255, 150),
            1
        )

        btn_text = "LANCER LE PROGRAMME  [ENTREE]"
        (tw, th), _ = cv2.getTextSize(btn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx = btn_start_x + (btn_start_w - tw) // 2
        ty = btn_y + (btn_start_h + th) // 2
        cv2.putText(canvas, btn_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 30, 10), 2, cv2.LINE_AA)

        # Raccourci d'annulation
        cv2.putText(
            canvas,
            "Appuyez sur [ECHAP] pour quitter",
            (35, self.height - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (140, 130, 120),
            1,
            cv2.LINE_AA
        )

        return canvas

    def _draw_radio_card(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str,
        desc: str,
        badge: str,
        is_selected: bool,
        is_hovered: bool
    ):
        """Dessine une carte d'option avec un radio-button moderne."""
        # Couleur du fond de la carte
        if is_selected:
            bg_color = (55, 42, 38)
            border_color = (0, 215, 255)  # Bordure dorée / cyan pour l'actif
            border_thick = 2
        elif is_hovered:
            bg_color = (48, 38, 35)
            border_color = (120, 100, 90)
            border_thick = 1
        else:
            bg_color = (38, 30, 27)
            border_color = (65, 52, 48)
            border_thick = 1

        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, border_thick)

        # Radio button graphique (cercle extérieur + point central si coché)
        radio_cx = x + 25
        radio_cy = y + h // 2
        radio_r = 9

        if is_selected:
            cv2.circle(canvas, (radio_cx, radio_cy), radio_r, (0, 215, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (radio_cx, radio_cy), 5, (0, 215, 255), -1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(canvas, (radio_cx, radio_cy), radio_r, (120, 110, 100), 1, lineType=cv2.LINE_AA)

        # Titre
        title_color = (255, 255, 255) if is_selected else (210, 205, 200)
        cv2.putText(
            canvas,
            title,
            (x + 48, y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            title_color,
            2 if is_selected else 1,
            cv2.LINE_AA
        )

        # Description
        cv2.putText(
            canvas,
            desc,
            (x + 48, y + 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 150, 140),
            1,
            cv2.LINE_AA
        )

        # Badge à droite
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        badge_x = x + w - bw - 22
        badge_y = y + 16
        cv2.rectangle(
            canvas,
            (badge_x - 6, badge_y - 2),
            (badge_x + bw + 6, badge_y + bh + 6),
            (55, 45, 40) if not is_selected else (0, 130, 160),
            -1
        )
        cv2.putText(
            canvas,
            badge,
            (badge_x, badge_y + bh + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220) if not is_selected else (255, 255, 255),
            1,
            cv2.LINE_AA
        )


def launch_algorithm_selector(
    default_model: str = "yolov8l.pt",
    default_tracker: str = "configs/custom_botsort.yaml"
) -> Tuple[str, str]:
    """Affiche l'interface de sélection graphique ou terminal en cas d'absence d'écran.

    Returns:
        Tuple (modele_choisi, tracker_choisi)
    """
    has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")

    # Si pas d'affichage X11/Wayland, fallback élégant en console
    if not has_display:
        print("\n" + "=" * 60)
        print("          CHOIX DES ALGORITHMES (MODE CONSOLE)")
        print("=" * 60)
        print("Algorithmes de Détection disponibles :")
        print("  [1] YOLOv8n (Nano)  - Rapide, faible consommation")
        print("  [2] YOLOv8l (Large) - Haute précision (Recommandé)")
        choice_det = input("Votre choix [1/2] (défaut: 2) : ").strip()
        model = "yolov8n.pt" if choice_det == "1" else "yolov8l.pt"

        print("\nAlgorithmes de Suivi (Tracking) disponibles :")
        print("  [1] BoT-SORT  - Re-ID visuel + mémoire 90f (Recommandé)")
        print("  [2] ByteTrack - Suivi cinématique rapide")
        choice_trk = input("Votre choix [1/2] (défaut: 1) : ").strip()
        tracker = "configs/custom_bytetrack.yaml" if choice_trk == "2" else "configs/custom_botsort.yaml"

        print("=" * 60 + "\n")
        return (model, tracker)

    # Fenêtre graphique OpenCV moderne
    dialog = LauncherDialog(default_model=default_model, default_tracker=default_tracker)
    window_name = "Configuration - Selection des Algorithmes IA"

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, dialog.on_mouse)

        print("\n[INFO] Ouverture du sélecteur d'algorithmes graphique...")

        while True:
            frame = dialog.render()
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(25) & 0xFF

            # Validation (Entrée / Espace)
            if key in (13, 32) or dialog.confirmed:
                break

            # Annulation (Échap / Q)
            elif key in (27, ord('q'), ord('Q')) or dialog.canceled:
                print("[INFO] Lancement annulé par l'utilisateur.")
                cv2.destroyAllWindows()
                sys.exit(0)

            # Raccourcis clavier
            elif key == ord('1'):
                dialog.selected_model = "yolov8n.pt"
            elif key == ord('2'):
                dialog.selected_model = "yolov8l.pt"
            elif key == ord('3'):
                dialog.selected_tracker = "configs/custom_botsort.yaml"
            elif key == ord('4'):
                dialog.selected_tracker = "configs/custom_bytetrack.yaml"

        cv2.destroyWindow(window_name)

        print(f"[CHOIX VALIDE] Détection : {dialog.selected_model} | Suivi : {dialog.selected_tracker}")
        return (dialog.selected_model, dialog.selected_tracker)

    except cv2.error as e:
        print(f"[AVERTISSEMENT] Erreur d'ouverture d'interface graphique : {e}")
        print("[INFO] Utilisation des options par défaut : YOLOv8l + BoT-SORT")
        return (default_model, default_tracker)
