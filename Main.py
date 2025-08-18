# -*- coding: utf-8 -*-
# ==============================================================================
# Objet :
#   - Pipeline de classification de données raster par apprentissage automatique.
#   - Utilise l'algorithme XGBoost pour une classification supervisée.
# Entrées :
#   - Image raster multibande (format GeoTIFF).
#   - Fichier de polygones de référence pour l'entraînement (format Shapefile).
# Sorties :
#   - Raster classifié (format GeoTIFF).
#   - Graphiques de performance (matrice de confusion, courbes ROC).
#   - Matrices de confusion au format CSV.
# Auteur : MDPY (IRD ESPACE-Dev) et Baptiste dLR
# Date de création : juin 2024
# Dernière mise à jour : juillet 2025 (Optimisation pour XGBoost) par Baptiste dLR
# ==============================================================================

# --- Importation des bibliothèques fondamentales ---
import numpy as np
import os
import datetime
from collections import Counter
import configparser

# --- Importation des modules de traitement d'image et de données ---
import rasterio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Importation des modules de Machine Learning ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import xgboost as xgb
from xgboost import XGBClassifier

# --- Importation des modules personnalisés ---
from Classif_XGBoost import XGBoost_pipeline
from feature_engineering import creer_raster_multibandes
from extract_features import extract_pixels_from_polygons


def save_classified_raster(classified_array, output_path, meta):
    """
    Enregistre un tableau NumPy 2D de classes prédites en tant que fichier raster GeoTIFF.
    
    Args:
        classified_array (numpy.ndarray): Tableau 2D des classes prédites.
        output_path (str): Chemin de sauvegarde du fichier GeoTIFF.
        meta (dict): Dictionnaire des métadonnées Rasterio hérité du raster source.
    """
    print("     --| Sauvegarde des Résultats |--")
    # Utilisation du gestionnaire de contexte pour une écriture sécurisée.
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(classified_array.astype(rasterio.uint8), 1)


def Classify_raster(raster_source, classifier_model, selected_features=None):
    """
    Applique un modèle de classification entraîné sur une image raster complète en ignorant les pixels `nodata`.

    Args:
        raster_source (rasterio.DatasetReader): Objet Rasterio pour le raster source.
        classifier_model: Modèle de classification entraîné.
        selected_features (list, optional): Liste des noms de bandes à utiliser pour la classification.

    Returns:
        tuple: Tuple contenant les prédictions (excluant les pixels `nodata`) et un masque booléen des pixels `nodata`.
    """
    print(f"     --| Classification du Raster | XGBoost |--")
    
    img = raster_source.read()
    bands, height, width = img.shape
    
    nodata_value_src = raster_source.nodata
    
    if nodata_value_src is None:
        print("ATTENTION: Aucune valeur nodata n'est définie dans le raster source. Tous les pixels seront classifiés.")
        nodata_mask_flat = np.zeros(height * width, dtype=bool)
    else:
        nodata_mask_2d = (img == nodata_value_src).all(axis=0)
        nodata_mask_flat = nodata_2d.flatten()

    if selected_features is not None:
        src_band_names = [desc[0] if isinstance(desc, tuple) else desc for desc in raster_source.descriptions]
        band_indices = [src_band_names.index(b_name) for b_name in selected_features if b_name in src_band_names]
        
        if len(band_indices) != len(selected_features):
            print(f"ATTENTION: Certaines bandes demandées ne sont pas trouvées. Classification basée sur les {len(band_indices)} bandes disponibles.")
        
        img = img[band_indices, :, :]
        bands = img.shape[0]

    X_pixels = img.reshape(bands, height * width).T
    X_pixels_to_predict = X_pixels[~nodata_mask_flat]

    if selected_features is not None:
        X_pixels_for_prediction = pd.DataFrame(X_pixels_to_predict, columns=selected_features)
    else:
        X_pixels_for_prediction = pd.DataFrame(X_pixels_to_predict, columns=[f'band_{i+1}' for i in range(bands)])

    if isinstance(classifier_model, XGBClassifier):
        classified_data_pred = classifier_model.predict(X_pixels_for_prediction)
    else:
        dmatrix = xgb.DMatrix(X_pixels_for_prediction, feature_names=selected_features)
        classified_data_pred = classifier_model.predict(dmatrix)
        
    return classified_data_pred, nodata_mask_flat


def plot_results(y_true, y_pred, y_pred_proba, class_mapping, output_path, model_name):
    """
    Génère et sauvegarde la matrice de confusion normalisée et les courbes ROC par classe.
    
    Args:
        y_true (array): Vraies étiquettes du jeu de test.
        y_pred (array): Étiquettes prédites (labels discrets).
        y_pred_proba (array): Probabilités prédites pour chaque classe.
        class_mapping (dict): Dictionnaire mappant les ID numériques aux noms de classe.
        output_path (str): Chemin du dossier pour la sauvegarde des graphiques.
        model_name (str): Nom du modèle pour les titres de graphique et les noms de fichiers.
    """
    print(f"     --| Plot Results {model_name} |--")
    
    # --- 1. Matrice de Confusion Normalisée ---
    unique_labels = sorted(list(class_mapping.keys()))
    actual_labels = np.unique(np.concatenate((y_true, y_pred)))
    display_labels = [label for label in unique_labels if label in actual_labels]
    class_names = [class_mapping[label] for label in display_labels]

    cm = confusion_matrix(y_true, y_pred, labels=display_labels)
    
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    csv_path = os.path.join(output_path, f'confusion_matrix_{model_name}.csv')
    cm_df.to_csv(csv_path, sep=';')

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized[np.isnan(cm_normalized)] = 0

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='viridis',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matrice de Confusion Normalisée - {model_name}', fontsize=16)
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plot_path_cm = os.path.join(output_path, f'confusion_matrix_percent_{model_name}.png')
    plt.savefig(plot_path_cm, dpi=300)
    plt.close()

    # --- 2. Courbes ROC (par classe et micro-moyenne) ---
    print(f"     --| Plotting ROC Curves {model_name} |--")
    y_true_bin = label_binarize(y_true, classes=display_labels)
    n_classes = y_true_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(12, 10))

    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0 and y_pred_proba.shape[1] > i:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            if roc_auc[i] is not np.nan:
                plt.plot(fpr[i], tpr[i], lw=2,
                         label=f'ROC de la classe {class_names[i]} (AUC = {roc_auc[i]:.2f})')
        else:
            print(f"Avertissement: La courbe ROC de la classe {class_names[i]} ne peut pas être tracée.")

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Courbe ROC micro-moyenne (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title(f'Courbes ROC par Classe et Micro-Moyenne - {model_name}', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    
    plot_path_roc = os.path.join(output_path, f'roc_curves_{model_name}.png')
    plt.savefig(plot_path_roc, dpi=300)
    plt.close()
    
    print(f"--- Graphiques de résultats (Matrice & ROC) pour {model_name} sauvegardés. ---")


# ==============================================================================
# --- SCRIPT PRINCIPAL (MAIN) ---
# ==============================================================================

if __name__ == '__main__':
    start_time_total = datetime.datetime.now()

    # --- 1. Configuration des chemins d'entrée/sortie ---
    # Lecture des chemins depuis le fichier de configuration
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Récupération des chemins
    input_raster_path = config['Paths']['input_raster']
    input_vector_path = config['Paths']['input_vector']
    output_base_dir = config['Paths']['output_dir']

    # Définition des dossiers de sortie
    output_xgb_dir = os.path.join(output_base_dir, "XGBoost")
    
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(output_xgb_dir, exist_ok=True)

    C_OutFile = os.path.splitext(os.path.basename(input_raster_path))[0]
    FEng_path_raster = os.path.join(output_base_dir, f"{C_OutFile}_10bands.tif")
    output_name_xgb = f"{C_OutFile}_classification_XGB.tif"

    print(f"Raster d'entrée : {input_raster_path}")
    print("-" * 30)
    print(f"Sorties communes : {output_base_dir}")
    print(f"Sorties XGBoost : {output_xgb_dir}")
    print("-" * 30)
    print(f"Raster à 10 bandes : {FEng_path_raster}")

    # --- 2. Ingénierie des caractéristiques (Feature Engineering) ---
    print("\n--- Démarrage de l'ingénierie des caractéristiques (Feature Engineering) ---")
    features_stack, profile, band_names = creer_raster_multibandes(input_raster_path)
    profile['count'] = len(band_names)

    with rasterio.open(FEng_path_raster, 'w', **profile) as dst:
        dst.write(features_stack)
        for i, band_name in enumerate(band_names):
            dst.set_band_description(i + 1, band_name)
    print(f"--- Raster à {len(band_names)} bandes créé et sauvegardé. ---")

    # --- 3. Extraction et préparation des données d'apprentissage ---
    print("\n--- Extraction des données d'entraînement depuis le raster multi-bandes ---")
    
    X_model = extract_pixels_from_polygons(
        shapefile_path=input_vector_path,
        raster_path=FEng_path_raster,
        label_col="id",
        class_name_col="Classe"
    )

    class_map_df = X_model[['id', 'Classe']].drop_duplicates()
    class_mapping = pd.Series(class_map_df.Classe.values, index=class_map_df.id).to_dict()
    print("--- Dictionnaire de correspondance des classes créé ---")
    print(class_mapping)

    y_array = X_model["id"]
    X_model = X_model.drop(columns=["id", "Classe"])
    
    band_names_updated = list(X_model.columns)
    X_model.columns = band_names_updated

    counts = y_array.value_counts()
    classes_to_keep = counts[counts >= 2].index
    mask_data = y_array.isin(classes_to_keep)
    X_model = X_model.loc[mask_data]
    y_array = y_array.loc[mask_data]
    class_names = y_array.unique()
    print(f"--- Données prêtes pour {len(class_names)} classes. ---")

    # --- 4. Division des données en ensembles d'entraînement et de test ---
    print("Distribution des classes après filtrage (avant split) :")
    print(y_array.value_counts())
    print(f"Nombre total de classes pour l'entraînement/test : {len(class_names)}")
    print("\n--- Division des données (70% entraînement, 30% test) avec stratification ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y_array,
        test_size=0.3,
        random_state=42,
        stratify=y_array
    )
    print(f"Taille du jeu d'entraînement : {X_train.shape[0]} échantillons")
    print(f"Taille du jeu de test : {X_test.shape[0]} échantillons")
    
    # --- 5. Pipeline de classification : XGBoost ---
    print("\n====| DÉMARRAGE DU PIPELINE XGBOOST |====")
    start_time_xgb = datetime.datetime.now()
    
    metrics_xgb, _, le_encoder = XGBoost_pipeline(X_train, y_train, X_test, y_test, list(X_model.columns), class_mapping, output_xgb_dir)
    
    CLASSIFIED_NODATA_VALUE = 255

    with rasterio.open(FEng_path_raster) as src_to_classify:
        metaSRC_xgb = src_to_classify.meta.copy()

        classified_data_pred, nodata_mask_flat = Classify_raster(
            src_to_classify, metrics_xgb['Model'], selected_features=metrics_xgb.get('Selected Features')
        )

        decoded_valid_predictions = le_encoder.inverse_transform(classified_data_pred.flatten())
        
        final_classified_flat = np.full(src_to_classify.width * src_to_classify.height,
                                        CLASSIFIED_NODATA_VALUE,
                                        dtype=np.uint8)
        
        final_classified_flat[~nodata_mask_flat] = decoded_valid_predictions
        
        classified_raster = final_classified_flat.reshape(src_to_classify.height, src_to_classify.width)

        metaSRC_xgb.update(count=1, dtype=rasterio.uint8, nodata=CLASSIFIED_NODATA_VALUE)

        output_raster_path_xgb = os.path.join(output_xgb_dir, output_name_xgb)
        save_classified_raster(classified_raster, output_raster_path_xgb, metaSRC_xgb)
        print(f"--- Raster XGB classifié sauvegardé. ---")
        
    plot_results(metrics_xgb['True Labels'], metrics_xgb['Predicted'], metrics_xgb['Predicted Probabilities'], class_mapping, output_xgb_dir, "XGB")
    end_time_xgb = datetime.datetime.now()
    print(f"====| PIPELINE XGBOOST TERMINÉ ! Temps d'exécution : {end_time_xgb - start_time_xgb} |====")
    
    end_time_total = datetime.datetime.now()
    print(f"\n====| Classification Complète Terminée ! Temps d'exécution total : {end_time_total - start_time_total} |====")
