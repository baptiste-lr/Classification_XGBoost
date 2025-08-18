# Classification des images satellite avec XGBoost

## 📖 Description du Projet
Ce dépôt contient un pipeline complet de classification d'images satellite par apprentissage automatique, utilisant l'algorithme XGBoost (Extreme Gradient Boosting). Le projet ici toutes les bandes, de la préparation des données à la classification finale du raster.

Les principales fonctionnalités incluses :
* **Ingénierie de Caractéristiques** : Création d'indices spectraux (NDVI, MNDWI, etc.) et de couches de texture pour enrichir les données d'entrée.
* **Extraction de Données d'Entrait** : Utilisation de polygones de référence (Shapefile) pour extraire des pixels d'entraînement.
* **Pipeline XGBoost** : Entrait d'un classificateur, optimisation des hyperparamètres via `Recherche aléatoireCV`, et évaluation des performances.
* **Visualisation des Résultats** : Génération de matrices de confusion et de cours ROC pour une analyse visuelle.
* **Finale de la classification** : Application du modèle intégré sur l'image entière pour produit un raster classé.

## ⚙️ Prérequis

Assurez-vous d'avoir Python 3.8+ installé.

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/baptiste-lr/Classification_XGBoost.git](https://github.com/baptiste-lr/Classification_XGBoost.git)
    cd Classification_XGBoost
    ```

2.  **Installer les dépendances :**
    ```bash
    pip install -r exigences.txt
    ```

## 🚀 Utilisation
1.  **Préparation des données :**
 Placez votre image satellite (GeoTIFF) et votre fichier de polygones de référence (Shapefile) dans le dossier `données/`.

2.  **Configuration des chimies :**
 Modifiez le fichier `config.ini` pour spécifier les chemins d'entrée et de sortie.
    ```ini
    [Chémins]
    input_raster = données/image_satellite.tif
    input_vector = données/polygones.shp
    output_dir = sorties/
    ```

3.  **Lancer le pipeline :**
 Exécutez le script principal depuis le terminal :
    ```bash
    scripts python/Main.py
    ```

Les résultats (raster classifié, graphiques, matrices de confusion) seront sauvegardés dans le dossier `sorties/`.

## 📂 Structure du Projet

📦 classification-d'images-satellites-xgboost/
├── README.md (Ce fichier)
├── requirements.txt (Dépendances Python)
├── config.ini (Fichier de configuration)
├── scripts/
│ ├── Main.py (Script principal du pipeline)
│ ├── Classif_XGBoost.py (Module pour le classificateur XGBoost)
│ ├── feature_engineering.py (Génération des indices et textures)
│ └── extract_features.py (Extraction des pixels d'entraînement)
└── sorties/ (Dossier des résultats)
