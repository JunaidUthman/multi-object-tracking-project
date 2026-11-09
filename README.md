# Pipeline de Détection, Suivi et Analyse Vidéo (YOLOv8 + BoT-SORT / ByteTrack)

Application de traitement vidéo séquentiel en flux continu pour l'analyse de trajectoires, le comptage de flux (entrées / sorties) et l'estimation de l'occupation dans des lieux publics (ex: séquences MOT17).

---

## 🏗️ Architecture du Projet

```text
computer_vision_project/
├── configs/
│   ├── custom_botsort.yaml   # BoT-SORT avec Re-ID visuel (vêtements), buffer 90f et appearance_thresh 0.25
│   └── custom_bytetrack.yaml # ByteTrack optimisé avec buffer 90f
├── data/                     # Dossier pour placer vos vidéos de test (ex: test_video.mp4)
├── output/                   # Vidéos générées avec incrustations graphiques
├── src/
│   ├── __init__.py
│   ├── geometry.py           # Calculs géométriques (centres de boîtes, intersection de segments, sens)
│   ├── interactive.py        # Sélection interactive de la ligne de comptage par clic souris
│   ├── launcher_ui.py        # Interface moderne de sélection des algorithmes IA (Détecteur + Tracker)
│   └── pipeline.py           # Pipeline séquentiel : YOLOv8 + Tracker + Historique + Rendu
├── main.py                   # Script d'exécution en ligne de commande
├── yolov8n.pt                # Poids YOLOv8 Nano (6 Mo - rapide)
├── yolov8l.pt                # Poids YOLOv8 Large (84 Mo - haute précision)
├── requirements.txt          # Dépendances requises
└── .gitignore                # Exclusion des vidéos, poids et caches
```

---

## ⚙️ Paramètres d'Inférence Optimisés Anti-Occultation & Anti-Swaps

Pour résoudre la perte de personnes très proches ou les permutations d'identifiants (ID switches) :
1. **`conf=0.15`** : Seuil très sensible captant les silhouettes masquées à 50% ou plus.
2. **`iou=0.65`** : Tolère jusqu'à 65% de chevauchement avant que le NMS ne supprime une boîte (crucial pour les personnes marchant côte à côte).
3. **`imgsz=1280`** : Résolution d'inférence haute définition pour séparer nettement les pixels dans les foules denses.
4. **`with_reid: True` + `appearance_thresh: 0.25`** : BoT-SORT extrait les vecteurs d'apparence visuelle (couleur des habits) pour distinguer deux personnes qui se croisent.
5. **`track_buffer: 90`** : Mémorise l'identité pendant 3 secondes d'occultation complète.

---

## 🚀 Utilisation

### 1. Déposer une vidéo
Placez votre vidéo brute dans le dossier [`data/`](file:///home/junaid_uth/projects/computer_vision_project/data) nommée `data/test_video.mp4`.

### 2. Lancer l'application
```bash
python main.py
```

1. **Écran 1 (Sélecteur d'Algorithmes)** :
   Une interface graphique s'affiche avec des boutons radio pour choisir :
   - **Détecteur** : `YOLOv8l (Large)` *(recommandé)* ou `YOLOv8n`
   - **Tracker** : `BoT-SORT (Re-ID Optimisé)` *(recommandé)* ou `ByteTrack`
   - Cliquez sur **[LANCER LE PROGRAMME]** ou appuyez sur **[ENTRÉE]**.

2. **Écran 2 (Ligne de Comptage)** :
   - Cliquez 2 fois avec la souris pour positionner le segment virtuel de flux.
   - Appuyez sur **[ENTRÉE]** pour démarrer le traitement temps réel.

---

### 3. Options avancées en ligne de commande

| Option | Rôle | Valeur par défaut |
| :--- | :--- | :--- |
| `--model, -m` | Modèle YOLOv8 | `yolov8l.pt` |
| `--tracker, -t` | Fichier ou nom du tracker | `configs/custom_botsort.yaml` |
| `--conf` | Seuil de confiance minimal | `0.15` |
| `--iou` | Tolérance de chevauchement NMS | `0.65` |
| `--imgsz` | Résolution d'inférence | `1280` |
| `--history` | Mémoire des trajectoires tracées | `90` |
| `--no-ui` | Sauter l'interface de sélection au lancement | `False` |
| `--no-show` | Traitement rapide en tâche de fond | `False` |

Exemple direct sans interface graphique :
```bash
python main.py --model yolov8l.pt --tracker botsort --conf 0.15 --iou 0.65 --imgsz 1280
```
