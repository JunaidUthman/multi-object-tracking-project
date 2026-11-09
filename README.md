# Pipeline de Détection, Suivi et Analyse Vidéo (YOLOv8 + ByteTrack / BoT-SORT)

Application de traitement vidéo séquentiel en flux continu pour l'analyse de trajectoires, le comptage de flux (entrées / sorties) et l'estimation de l'occupation dans des lieux publics (ex: séquences MOT17).

---

## 🏗️ Architecture du Projet

```text
computer_vision_project/
├── data/                  # Dossier pour placer vos vidéos de test (ex: test_video.mp4)
├── output/                # Vidéos générées avec incrustations graphiques
├── src/
│   ├── __init__.py
│   ├── geometry.py        # Calculs géométriques (centres de boîtes, intersection de segments, sens)
│   ├── interactive.py     # Sélection interactive de la ligne de comptage par clic souris
│   ├── launcher_ui.py     # Interface moderne de sélection des algorithmes IA (Détecteur + Tracker)
│   └── pipeline.py        # Pipeline séquentiel : YOLOv8 + Tracker + Historique + Rendu
├── main.py                # Script d'exécution en ligne de commande
├── yolov8n.pt             # Poids YOLOv8 Nano (6 Mo - rapide)
├── yolov8l.pt             # Poids YOLOv8 Large (84 Mo - haute précision)
├── requirements.txt       # Dépendances requises
└── .gitignore             # Exclusion des vidéos, poids et caches
```

---

## ⚙️ Installation des Dépendances

Activez votre environnement virtuel puis installez les dépendances :

```bash
# Activation de l'environnement virtuel
source .venv/bin/activate

# Installation des dépendances
pip install -r requirements.txt
```

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
   - **Détecteur** : `YOLOv8n` ou `YOLOv8l (Large)` *(sélectionné par défaut)*
   - **Tracker** : `ByteTrack` ou `BoT-SORT`
   - Cliquez sur **[LANCER LE PROGRAMME]** ou appuyez sur **[ENTRÉE]**.

2. **Écran 2 (Ligne de Comptage)** :
   - Cliquez 2 fois avec la souris pour positionner le segment virtuel de flux.
   - Appuyez sur **[ENTRÉE]** pour démarrer le traitement temps réel.

---

### 3. Options avancées en ligne de commande

Si vous préférez contourner l'interface graphique de sélection au démarrage :

| Option | Rôle | Exemple |
| :--- | :--- | :--- |
| `--model, -m` | Forcer le modèle YOLOv8 | `--model yolov8l.pt` ou `--model yolov8n.pt` |
| `--tracker, -t` | Forcer l'algorithme de suivi | `--tracker botsort.yaml` ou `--tracker bytetrack.yaml` |
| `--input, -i` | Chemin de la vidéo d'entrée | `--input data/test_video.mp4` |
| `--output, -o` | Chemin de la vidéo de sortie (défaut auto: `output/<modele>_<tracker>_result.mp4`) | `--output output/mon_test.mp4` |
| `--line` | Définir la ligne sans fenêtre graphique | `--line 100,500,800,500` *(x1,y1,x2,y2)* |
| `--no-ui` | Sauter l'interface de sélection au lancement | `--no-ui` |
| `--no-show` | Traitement rapide sans affichage direct | `--no-show` |
| `--history` | Longueur de la traînée de trajectoire | `--history 120` |

Exemple direct avec YOLOv8l et BoT-SORT :
```bash
python main.py --model yolov8l.pt --tracker botsort.yaml
```
