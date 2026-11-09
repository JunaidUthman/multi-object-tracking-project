#!/usr/bin/env python3
"""Point d'entrée principal de l'application de détection, suivi et comptage vidéo.

Usage :
    python main.py --input data/test_video.mp4 --output output/resultat.mp4
"""

import argparse
import os
import sys
from typing import Optional, Tuple

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
        description="Application de Détection & Suivi d'objets avec YOLOv8 et ByteTrack."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/test_video.mp4",
        help="Chemin de la vidéo d'entrée (ex: data/test_video.mp4)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/tracked_output.mp4",
        help="Chemin de la vidéo de sortie annotée (ex: output/tracked_output.mp4)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov8n.pt",
        help="Modèle YOLOv8 à charger (ex: yolov8n.pt, yolov8s.pt, yolov8m.pt)"
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
        default=0.3,
        help="Seuil de confiance pour les détections YOLO (défaut: 0.3)"
    )
    parser.add_argument(
        "--history",
        type=int,
        default=200,
        help="Longueur de l'historique des positions pour tracer les trajectoires (défaut: 90 frames = ~3 sec)"
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

    counting_line = parse_line_coordinates(args.line)

    print("=" * 60)
    print("      PIPELINE DE TRAITEMENT VIDÉO & ANALYSE DE TRAJECTOIRES")
    print("=" * 60)
    print(f"• Vidéo source       : {args.input}")
    print(f"• Vidéo de sortie    : {args.output}")
    print(f"• Modèle IA          : {args.model}")
    print(f"• Algorithme Tracker : ByteTrack")
    print(f"• Historique points  : {args.history} frames")
    if counting_line:
        print(f"• Ligne manuelle     : {counting_line[0]} -> {counting_line[1]}")
    else:
        print(f"• Ligne              : Définition interactive (clic à l'écran)")
    print("=" * 60 + "\n")

    pipeline = VideoTrackingPipeline(
        video_source_path=args.input,
        video_output_path=args.output,
        model_name=args.model,
        history_len=args.history,
        conf_threshold=args.conf,
        counting_line=counting_line,
        show_live=not args.no_show
    )

    pipeline.process()


if __name__ == "__main__":
    main()
