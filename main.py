#!/usr/bin/env python3
"""Point d'entrée principal de l'application de détection, suivi et analyse vidéo.

Permet de choisir interactivement les algorithmes IA (YOLOv8n / YOLOv8l et ByteTrack / BoT-SORT)
via une interface graphique moderne au démarrage.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

from src.launcher_ui import launch_algorithm_selector
from src.pipeline import VideoTrackingPipeline


def parse_line_coordinates(line_str: Optional[str]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Parse une chaîne 'x1,y1,x2,y2' en coordonnées de ligne."""
    if not line_str:
        return None
    try:
        parts = [int(p.strip()) for p in line_str.split(",")]
        if len(parts) != 4:
            raise ValueError
        return ((parts[0], parts[1]), (parts[2], parts[3]))
    except Exception:
        print(f"[ERREUR] Format de ligne invalide '{line_str}'. Format attendu: x1,y1,x2,y2")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Application de Détection & Suivi d'objets avec sélection d'algorithmes (YOLOv8 + ByteTrack / BoT-SORT)."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/test_video.mp4",
        help="Chemin de la vidéo d'entrée (défaut: data/test_video.mp4)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Chemin de la vidéo de sortie (défaut automatique: output/<modele>_<tracker>_result.mp4)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        choices=["yolov8n.pt", "yolov8l.pt", "yolov8s.pt", "yolov8m.pt", "yolov8x.pt"],
        help="Modèle YOLOv8 (si omis, la boîte de sélection graphique s'affiche)"
    )
    parser.add_argument(
        "--tracker", "-t",
        type=str,
        default=None,
        help="Algorithme de tracking ('botsort', 'bytetrack', ou chemin vers un fichier YAML)"
    )
    parser.add_argument(
        "--line",
        type=str,
        default=None,
        help="Coordonnées prédéfinies de la ligne de comptage : 'x1,y1,x2,y2' (évite le clic souris)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Seuil de confiance pour les détections YOLO (défaut: 0.15 pour personnes à moitié cachées)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.65,
        help="Seuil de recouvrement NMS pour séparer les personnes proches (défaut: 0.65)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Résolution d'inférence en pixels pour séparer les pixels des foules (défaut: 1280)"
    )
    parser.add_argument(
        "--history",
        type=int,
        default=90,
        help="Longueur de l'historique des positions pour tracer les trajectoires (défaut: 90 frames = ~3 sec)"
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Désactiver l'interface graphique de sélection des algorithmes au démarrage"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Désactiver l'affichage temps réel pendant le traitement (plus rapide)"
    )

    args = parser.parse_args()

    # Vérification de l'existence de la vidéo
    if not os.path.exists(args.input):
        print(f"\n[ERREUR] Le fichier d'entrée '{args.input}' n'existe pas.")
        print(f"👉 Veuillez déposer votre vidéo dans le dossier 'data/' (ex: data/test_video.mp4)")
        print(f"   Ou spécifier son chemin avec : python main.py --input chemin/vers/video.mp4\n")
        sys.exit(1)

    # --- Étape 0 : Sélection graphique des algorithmes si non spécifiés en CLI ---
    model_name = args.model
    tracker_name = args.tracker

    if (model_name is None or tracker_name is None) and not args.no_ui:
        chosen_model, chosen_tracker = launch_algorithm_selector(
            default_model="yolov8l.pt",
            default_tracker="configs/custom_botsort.yaml"
        )
        model_name = model_name or chosen_model
        tracker_name = tracker_name or chosen_tracker
    else:
        # Valeurs par défaut si --no-ui sans arguments
        model_name = model_name or "yolov8l.pt"
        tracker_name = tracker_name or "configs/custom_botsort.yaml"

    # Résolution des alias vers les fichiers de configuration optimisés
    tracker_aliases = {
        "botsort": "configs/custom_botsort.yaml",
        "botsort.yaml": "configs/custom_botsort.yaml",
        "bytetrack": "configs/custom_bytetrack.yaml",
        "bytetrack.yaml": "configs/custom_bytetrack.yaml",
    }
    tracker_name = tracker_aliases.get(tracker_name, tracker_name)

    # Génération automatique du nom de vidéo de sortie selon la combinaison de modèles
    det_slug = os.path.splitext(os.path.basename(model_name))[0]
    trk_slug = os.path.splitext(os.path.basename(tracker_name))[0].replace("custom_", "")
    default_filename = f"{det_slug}_{trk_slug}_result.mp4"

    if args.output is None:
        output_path = os.path.join("output", default_filename)
    elif os.path.isdir(args.output):
        output_path = os.path.join(args.output, default_filename)
    else:
        output_path = args.output

    counting_line = parse_line_coordinates(args.line)

    trk_label = "BoT-SORT (Re-ID)" if "botsort" in tracker_name.lower() else "ByteTrack"

    print("=" * 60)
    print("      PIPELINE DE TRAITEMENT VIDÉO & ANALYSE DE TRAJECTOIRES")
    print("=" * 60)
    print(f"• Vidéo source       : {args.input}")
    print(f"• Vidéo de sortie    : {output_path}")
    print(f"• Modèle Détecteur   : {model_name}")
    print(f"• Algorithme Tracker : {trk_label} ({tracker_name})")
    print(f"• Paramètres NMS     : conf={args.conf} | iou={args.iou} | imgsz={args.imgsz}")
    print(f"• Historique points  : {args.history} frames")
    if counting_line:
        print(f"• Ligne manuelle     : {counting_line[0]} -> {counting_line[1]}")
    else:
        print(f"• Ligne              : Définition interactive (clic à l'écran)")
    print("=" * 60 + "\n")

    pipeline = VideoTrackingPipeline(
        video_source_path=args.input,
        video_output_path=output_path,
        model_name=model_name,
        tracker_type=tracker_name,
        history_len=args.history,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        counting_line=counting_line,
        show_live=not args.no_show
    )

    pipeline.process()


if __name__ == "__main__":
    main()
