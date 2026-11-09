# Pipeline de Détection, Suivi et Analyse Vidéo (YOLOv8 + ByteTrack)

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
│   └── pipeline.py        # Pipeline séquentiel : YOLOv8 + ByteTrack + Historique + Rendu
├── main.py                # Script d'exécution en ligne de commande
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

> **Note :** Si votre machine ne dispose pas d'un GPU NVIDIA dédié, vous pouvez installer la version légère CPU de PyTorch avant d'installer les dépendances pour éviter de télécharger plusieurs gigaoctets de bibliothèques CUDA :
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

## 🚀 Utilisation

### 1. Déposer une vidéo
Placez votre vidéo brute dans le dossier [`data/`](file:///home/junaid_uth/projects/computer_vision_project/data) nommée `data/test_video.mp4`.

### 2. Lancer le pipeline avec sélection interactive
```bash
python main.py --input data/test_video.mp4 --output output/resultat.mp4
```

1. La première image de la vidéo s'affiche.
2. Cliquez deux fois avec la souris pour définir le segment virtuel de comptage.
3. Appuyez sur **[ENTRÉE]** pour valider et démarrer le traitement.
4. Appuyez sur **[R]** pour réinitialiser les points ou **[ÉCHAP]** pour utiliser la ligne médiane par défaut.

---

### 3. Options avancées

| Option | Rôle | Exemple |
| :--- | :--- | :--- |
| `--input, -i` | Chemin de la vidéo d'entrée | `--input data/test_video.mp4` |
| `--output, -o` | Chemin de la vidéo de sortie | `--output output/annotated.mp4` |
| `--model, -m` | Modèle YOLOv8 | `--model yolov8n.pt` (nano) ou `yolov8s.pt` (small) |
| `--line` | Définir la ligne sans fenêtre graphique | `--line 100,500,800,500` *(x1,y1,x2,y2)* |
| `--no-show` | Traitement rapide en tâche de fond (sans affichage direct) | `--no-show` |
| `--history` | Longueur de la traînée de trajectoire (défaut: 90 frames = ~3s) | `--history 120` |
| `--conf` | Seuil de confiance minimal | `--conf 0.35` |

Exemple en mode serveur (sans écran) :
```bash
python main.py --input data/test_video.mp4 --output output/resultat.mp4 --line 0,600,1920,600 --no-show
```
