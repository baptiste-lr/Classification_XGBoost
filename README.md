# Classification des images satellite avec XGBoost

## 📖 Description du Projet
Ce dépôt contient un pipeline complet de classification d'images satellite par apprentissage automatique, utilisant l'algorithme XGBoost (Extreme Gradient Boosting). Le projet traite toutes les bandes, de la préparation des données à la classification finale du raster.

Les principales fonctionnalités incluses :
* **Ingénierie de Caractéristiques** : Création d'indices spectraux (NDVI, MNDWI, etc.) et de couches de texture pour enrichir les données d'entrée.
* **Extraction de Données d'Entraînement** : Utilisation de polygones de référence (Shapefile) pour extraire des pixels d'entraînement.
* **Pipeline XGBoost** : Entraînement d'un classificateur, optimisation des hyperparamètres via `Recherche aléatoire CV`, et évaluation des performances.
* **Visualisation des Résultats** : Génération de matrices de confusion et de courbes ROC pour une analyse visuelle.
* **Classification finale** : Application du modèle entraîné sur l'image entière pour produire un raster classé.

## 🎯 Résultat de Classification

![Résultat de classification](./images/classif.png)
*Résultat de la classification d'image satellite avec XGBoost*

## ⚙️ Prérequis
Assurez-vous d'avoir Python 3.8+ installé.

1.  **Cloner le dépôt :**
```bash
    git clone https://github.com/baptiste-lr/Classification_XGBoost.git
    cd Classification_XGBoost
```

2.  **Installer les dépendances :**
```bash
    pip install -r requirements.txt
```

## 🚀 Utilisation

1.  **Préparation des données :**
    
    Placez votre image satellite (GeoTIFF) et votre fichier de polygones de référence (Shapefile) dans le dossier `données/`.

2.  **Configuration des paramètres :**
    
    Modifiez le fichier `config.ini` pour spécifier les chemins d'entrée et de sortie.
```ini
    [Chemins]
    input_raster = données/image_satellite.tif
    input_vector = données/polygones.shp
    output_dir = sorties/
```

3.  **Lancer le pipeline :**
    
    Exécutez le script principal depuis le terminal :
```bash
    python scripts/Main.py
```

Les résultats (raster classifié, graphiques, matrices de confusion) seront sauvegardés dans le dossier `sorties/`.

## 📂 Structure du Projet
```
📦 classification-images-satellites-xgboost/
├── README.md                    (Ce fichier)
├── requirements.txt             (Dépendances Python)
├── config.ini                   (Fichier de configuration)
├── images/                      (Images pour le README)
│   └── classif.png
├── scripts/
│   ├── Main.py                  (Script principal du pipeline)
│   ├── Classif_XGBoost.py       (Module pour le classificateur XGBoost)
│   ├── feature_engineering.py   (Génération des indices et textures)
│   └── extract_features.py      (Extraction des pixels d'entraînement)
└── sorties/                     (Dossier des résultats)
```
