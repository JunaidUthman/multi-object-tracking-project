"""Pipeline de traitement vidéo séquentiel : Détection (YOLOv8), Suivi (ByteTrack),
Calcul de trajectoires, Comptage de flux et Rendu vidéo.
"""

from collections import defaultdict, deque
from typing import Tuple, Dict, Set, Optional
import os
import cv2
import numpy as np
from ultralytics import YOLO

from src.geometry import compute_bbox_center, check_crossing_direction
from src.interactive import select_counting_line

# ByteTrack est l'algorithme de tracking unique et exclusif du projet
TRACKER_TYPE = "bytetrack.yaml"


class VideoTrackingPipeline:
    """Pipeline d'analyse vidéo séquentiel inspiré d'une chaîne de production."""

    def __init__(
        self,
        video_source_path: str,
        video_output_path: Optional[str] = None,
        model_name: str = "yolov8l.pt",
        tracker_type: str = "configs/custom_botsort.yaml",
        history_len: int = 90,
        target_classes: Optional[list] = None,
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.65,
        imgsz: int = 1280,
        counting_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        show_live: bool = True
    ):
        """Initialise le pipeline, les modèles et les descripteurs de fichiers vidéo.

        Args:
            video_source_path: Chemin vers le fichier vidéo d'entrée.
            video_output_path: Chemin vers le fichier vidéo généré en sortie.
            model_name: Poids YOLOv8 (ex: 'yolov8n.pt', 'yolov8l.pt').
            tracker_type: Configuration tracker (ex: 'configs/custom_botsort.yaml').
            history_len: Nombre de points historiques retenus par trajectoire (défaut: 90).
            target_classes: Classes COCO à filtrer (défaut: [0] pour 'person').
            conf_threshold: Seuil de confiance minimal (défaut: 0.15 pour personnes à moitié cachées).
            iou_threshold: Seuil de recouvrement NMS (défaut: 0.65 pour séparer les personnes proches).
            imgsz: Taille de redimensionnement de détection (défaut: 1280 pixels).
            counting_line: Coordonnées prédéfinies de la ligne ((x1, y1), (x2, y2)).
            show_live: Afficher la fenêtre de prévisualisation en direct.
        """
        self.video_source_path = video_source_path
        self.model_name = model_name
        self.tracker_type = tracker_type
        self.history_len = history_len
        self.target_classes = target_classes if target_classes is not None else [0]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.show_live = show_live

        # Détermination du nom de fichier vidéo de sortie incluant les modèles
        if video_output_path is None or os.path.isdir(video_output_path) or video_output_path.endswith(("/", "\\")):
            det_slug = os.path.splitext(os.path.basename(self.model_name))[0]
            trk_slug = os.path.splitext(os.path.basename(self.tracker_type))[0].replace("custom_", "")
            filename = f"{det_slug}_{trk_slug}_result.mp4"
            target_dir = video_output_path if (video_output_path and os.path.isdir(video_output_path)) else "output"
            self.video_output_path = os.path.join(target_dir, filename)
        else:
            self.video_output_path = video_output_path

        # --- Étape 1 : Initialisation du Modèle IA ---
        print(f"[1/4] Chargement du modèle YOLO : {self.model_name}...")
        self.model = YOLO(self.model_name)

        # --- Initialisation de la capture vidéo source ---
        if not os.path.exists(self.video_source_path):
            raise FileNotFoundError(f"Fichier vidéo source introuvable : {self.video_source_path}")

        self.cap = cv2.VideoCapture(self.video_source_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir le flux vidéo : {self.video_source_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Vidéo source : {self.width}x{self.height} @ {self.fps:.1f} FPS ({self.total_frames} frames)")

        # --- Préparation de l'écrivain vidéo de destination ---
        os.makedirs(os.path.dirname(os.path.abspath(self.video_output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.video_output_path, fourcc, self.fps, (self.width, self.height)
        )

        # --- Étape 2 : Configuration de la ligne virtuelle ---
        if counting_line is not None:
            self.counting_line = counting_line
        else:
            ret, first_frame = self.cap.read()
            if not ret or first_frame is None:
                raise RuntimeError("Impossible de lire la première image de la vidéo.")
            self.counting_line = select_counting_line(first_frame)
            # Revenir au début de la vidéo pour ne pas rater la première frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # --- Structures de données pour le suivi et les indicateurs ---
        # Mémoire temporelle : ID -> deque de (cx, cy) avec maxlen=30
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.history_len))

        # Suivi des franchissements pour éviter les multi-comptages par le même ID
        self.crossed_ids: Set[int] = set()

        # Indicateurs statistiques
        self.count_in = 0
        self.count_out = 0
        self.count_total = 0
        self.all_seen_ids: Set[int] = set()

    def process(self):
        """Exécute la boucle principale de traitement séquentiel frame par frame."""
        trk_name = "BOT-SORT" if "botsort" in self.tracker_type.lower() else "BYTETRACK"
        print(f"[3/4] Démarrage du traitement vidéo avec [{self.model_name}] + [{trk_name}]...")
        print(f"      • Paramètres : conf={self.conf_threshold} | iou={self.iou_threshold} | imgsz={self.imgsz}")
        print(f"      • Config Tracker : {self.tracker_type}")
        frame_idx = 0

        window_name = "Video Processing Pipeline - YOLOv8 + Tracker"
        if self.show_live and os.environ.get("DISPLAY"):
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(1280, self.width), min(720, self.height))

        try:
            while True:
                # 1. Extraction : Lecture de l'image
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("\n[INFO] Fin de la séquence vidéo atteinte.")
                    break

                frame_idx += 1

                # 2. Cerveau IA : Détection & Tracking via Ultralytics
                # persist=True assure la continuité des IDs entre les images successives
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_type,
                    classes=self.target_classes,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    verbose=False
                )

                current_frame_tracks = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    if boxes.id is not None:
                        ids = boxes.id.int().cpu().tolist()
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()

                        for track_id, box, conf in zip(ids, xyxy, confs):
                            current_frame_tracks.append((track_id, box, conf))
                            self.all_seen_ids.add(track_id)

                # 3. Logique Métier : Historique et Vérification du franchissement
                current_active_ids = set()
                line_p1, line_p2 = self.counting_line

                for track_id, box, _ in current_frame_tracks:
                    current_active_ids.add(track_id)
                    center = compute_bbox_center(box)
                    history = self.track_history[track_id]

                    # Si on a au moins une position précédente, on vérifie la trajectoire
                    if len(history) > 0:
                        prev_pos = history[-1]
                        curr_pos = center

                        # Vérification de franchissement si l'ID n'a pas déjà été comptabilisé
                        if track_id not in self.crossed_ids:
                            direction = check_crossing_direction(prev_pos, curr_pos, line_p1, line_p2)
                            if direction is not None:
                                self.crossed_ids.add(track_id)
                                self.count_total += 1
                                if direction == "IN":
                                    self.count_in += 1
                                else:
                                    self.count_out += 1

                    # Mise à jour de la mémoire temporelle (conserve max 30 points)
                    history.append(center)

                # Nettoyage progressif des vieux IDs disparus pour libérer la mémoire
                if frame_idx % 150 == 0:
                    for old_id in list(self.track_history.keys()):
                        if old_id not in current_active_ids and len(self.track_history[old_id]) == 0:
                            del self.track_history[old_id]

                # 4. Rendu Visuel : Incrustation graphique
                annotated_frame = self._render_frame(
                    frame=frame,
                    tracks=current_frame_tracks,
                    active_occupancy=len(current_active_ids),
                    frame_idx=frame_idx
                )

                # 5. Sauvegarde : Écriture dans le fichier vidéo de destination
                self.writer.write(annotated_frame)

                # Affichage optionnel en direct
                if self.show_live and os.environ.get("DISPLAY"):
                    cv2.imshow(window_name, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):
                        print("\n[INFO] Arrêt prématuré demandé par l'utilisateur.")
                        break

                if frame_idx % 30 == 0 or frame_idx == self.total_frames:
                    progress = (frame_idx / self.total_frames) * 100 if self.total_frames > 0 else 0
                    print(
                        f"\rProgression : {frame_idx}/{self.total_frames} frames ({progress:.1f}%) "
                        f"| Flux Total : {self.count_total} (IN: {self.count_in}, OUT: {self.count_out}) "
                        f"| Occupation : {len(current_active_ids)}",
                        end="",
                        flush=True
                    )

        finally:
            self._cleanup()

    def _render_frame(
        self,
        frame: np.ndarray,
        tracks: list,
        active_occupancy: int,
        frame_idx: int
    ) -> np.ndarray:
        """Dessine les boîtes, IDs, trajectoires, ligne de flux et tableau de bord."""
        annotated = frame.copy()
        line_p1, line_p2 = self.counting_line

        # 1. Tracé de la ligne virtuelle de comptage
        cv2.line(annotated, line_p1, line_p2, (0, 0, 255), 3)
        cv2.circle(annotated, line_p1, 6, (0, 0, 255), -1)
        cv2.circle(annotated, line_p2, 6, (0, 0, 255), -1)

        # 2. Tracé des trajectoires et des boîtes englobantes
        for track_id, box, conf in tracks:
            x1, y1, x2, y2 = map(int, box)

            # Couleur unique par ID (palette dynamique basée sur l'ID)
            color = self._get_id_color(track_id)

            # Dessin de la bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Étiquette : ID + Confiance
            label = f"ID: {track_id} ({conf:.2f})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + lw, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

            # Dessin de la trajectoire haute qualité (anti-aliasing, lissage et traînée progressive)
            history = self.track_history[track_id]
            raw_points = [p for p in history if p is not None]
            smoothed_points = self._smooth_trajectory(raw_points)
            n_pts = len(smoothed_points)

            for j in range(1, n_pts):
                # Progression douce de l'épaisseur (1px pour la queue, jusqu'à 3-4px vers la tête)
                progress = j / float(n_pts)
                thickness = max(1, int(1 + 3 * progress))
                cv2.line(
                    annotated,
                    smoothed_points[j - 1],
                    smoothed_points[j],
                    color,
                    thickness,
                    lineType=cv2.LINE_AA
                )

            # Point de tête lumineux haute définition au centre actuel
            if n_pts > 0:
                head_pos = smoothed_points[-1]
                cv2.circle(annotated, head_pos, 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(annotated, head_pos, 3, color, -1, lineType=cv2.LINE_AA)

        # 3. Tableau de bord / HUD en haut de l'image
        self._draw_dashboard(annotated, active_occupancy, frame_idx)

        return annotated

    def _draw_dashboard(self, frame: np.ndarray, occupancy: int, frame_idx: int):
        """Affiche les métriques clés (Flux, Occupation, FPS, Frame)."""
        overlay = frame.copy()
        # Bandeau semi-transparent
        cv2.rectangle(overlay, (0, 0), (self.width, 65), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Textes des métriques
        text_flux = f"FLUX TOTAL : {self.count_total}  (ENTREES: {self.count_in} | SORTIES: {self.count_out})"
        text_occupancy = f"OCCUPATION : {occupancy} pers."
        trk_name = "BOT-SORT" if "botsort" in self.tracker_type.lower() else "BYTETRACK"
        det_name = self.model_name.replace(".pt", "").upper()
        text_meta = f"Frame : {frame_idx}/{self.total_frames} | {det_name} + {trk_name}"

        cv2.putText(frame, text_flux, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text_occupancy, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 2, cv2.LINE_AA)
        cv2.putText(frame, text_meta, (self.width - 450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    @staticmethod
    def _smooth_trajectory(points: list) -> list:
        """Applique un filtre de lissage par moyenne pondérée pour supprimer les saccades de détection."""
        if len(points) < 3:
            return points

        smoothed = [points[0]]
        for i in range(1, len(points) - 1):
            sx = int(0.25 * points[i - 1][0] + 0.5 * points[i][0] + 0.25 * points[i + 1][0])
            sy = int(0.25 * points[i - 1][1] + 0.5 * points[i][1] + 0.25 * points[i + 1][1])
            smoothed.append((sx, sy))
        smoothed.append(points[-1])
        return smoothed

    @staticmethod
    def _get_id_color(track_id: int) -> Tuple[int, int, int]:
        """Génère une couleur BGR pseudo-aléatoire et constante pour chaque ID."""
        np.random.seed(int(track_id))
        color = np.random.randint(50, 255, size=3).tolist()
        return (int(color[0]), int(color[1]), int(color[2]))

    def _cleanup(self):
        """Libère les ressources vidéo et détruit les fenêtres OpenCV."""
        print(f"\n\n[4/4] Finalisation et clôture des fichiers...")
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()

        print(f"==================================================")
        print(f"RÉSUMÉ DU TRAITEMENT :")
        print(f"- Vidéo générée       : {self.video_output_path}")
        print(f"- Flux total traversé : {self.count_total} (IN: {self.count_in}, OUT: {self.count_out})")
        print(f"- Total IDs uniques   : {len(self.all_seen_ids)}")
        print(f"==================================================")
