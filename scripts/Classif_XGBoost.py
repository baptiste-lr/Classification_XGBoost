# -*- coding: utf-8 -*-
# ==============================================================================
# Objet :
#   - Module spécialiste du classifieur XGBoost (Extreme Gradient Boosting).
#   - Implémente un pipeline complet pour l'entraînement et l'évaluation du modèle.
# Description :
#   Ce module gère le prétraitement des données (encodage, pondération), la
#   sélection/évaluation de l'importance des caractéristiques, et l'optimisation
#   des hyperparamètres via une recherche aléatoire (`RandomizedSearchCV`).
#   Il retourne un modèle final entraîné et les métriques de performance.
# Auteur : Baptiste LR (IRD ESPACE-Dev)
# Dernière mise à jour : Juillet 2025
# ==============================================================================

# ==============================================================================
#                 BIBLIOTHÈQUES FONDAMENTALES ET DE MANIPULATION DE DONNÉES
# ==============================================================================
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
#                 BIBLIOTHÈQUES SCIENTIFIQUES ET DE MACHINE LEARNING
# ==============================================================================
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


def xgb_feature_selection(X_train, y_train, band_names, weights, output_dir):
    """
    Entraîne un modèle XGBoost préliminaire pour évaluer l'importance de chaque caractéristique.
    Cette fonction est utilisée pour l'analyse de l'importance des variables par gain.
    
    Args:
        X_train (pd.DataFrame): Données d'entraînement.
        y_train (np.ndarray): Étiquettes d'entraînement encodées (0 à N-1).
        weights (np.ndarray): Poids des échantillons pour gérer le déséquilibre de classes.
        output_dir (str): Répertoire de sortie pour sauvegarder le graphique d'importance.

    Returns:
        list: Retourne l'ensemble des noms de bandes d'entrée. L'analyse d'importance est
              réalisée pour l'information, mais toutes les caractéristiques sont conservées
              pour le modèle final.
    """
    print("\n--- Évaluation de l'importance des caractéristiques par gain (XGBoost) ---")
    num_classes = len(np.unique(y_train))
    
    model_initial = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        eval_metric='mlogloss',
        max_depth=10,
        learning_rate=0.2,
        seed=42,
        n_jobs=-1
    )

    if not isinstance(X_train, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train, columns=band_names)
    else:
        X_train_df = X_train

    print("Entraînement du modèle préliminaire pour l'analyse d'importance...")
    model_initial.fit(X_train_df, y_train, sample_weight=weights)

    importance = model_initial.get_booster().get_score(importance_type='gain')

    if not importance:
        print("Aucune importance de caractéristique calculée. Retourne toutes les bandes d'origine.")
        return band_names

    feature_data = []
    for f_key, importance_val in importance.items():
        feature_name = None
        if f_key.startswith('f') and f_key[1:].isdigit():
            idx = int(f_key[1:])
            if 0 <= idx < len(band_names):
                feature_name = band_names[idx]
        elif f_key in band_names:
            feature_name = f_key
        
        if feature_name:
            feature_data.append({'feature_name': feature_name, 'importance': importance_val})

    if not feature_data:
        print("Aucune caractéristique valide n'a pu être extraite. Retourne toutes les bandes d'origine.")
        return band_names

    importance_df = pd.DataFrame(feature_data).sort_values(by='importance', ascending=False)
    total_gain = importance_df['importance'].sum()
    importance_df['importance'] = importance_df['importance'] / total_gain if total_gain > 0 else 0

    print("Importance des caractéristiques (normalisée par gain) :")
    print(importance_df.to_string())

    # Visualisation de l'importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature_name', data=importance_df, palette='viridis')
    plt.title('Importance des Caractéristiques par Gain (XGBoost)', fontsize=16)
    plt.xlabel('Importance Normalisée')
    plt.ylabel('Caractéristique')
    plt.tight_layout()
    plot_path_importance = os.path.join(output_dir, 'feature_importance_xgb.png')
    plt.savefig(plot_path_importance, dpi=300)
    plt.close()
    print(f"--- Graphique d'importance des caractéristiques sauvegardé : {plot_path_importance} ---")

    print(f"\n--- Le modèle final utilisera toutes les {len(band_names)} caractéristiques d'entrée. ---")
    return band_names


def XGBoost_pipeline(X_train, y_train, X_test, y_test, band_names, class_mapping, output_dir):
    """
    Exécute un pipeline complet de classification XGBoost, incluant l'encodage des
    étiquettes, la pondération des classes, l'optimisation des hyperparamètres,
    et l'évaluation finale.

    Args:
        X_train (pd.DataFrame): Jeu de données d'entraînement.
        y_train (pd.Series ou np.ndarray): Étiquettes d'entraînement.
        X_test (pd.DataFrame): Jeu de données de test.
        y_test (pd.Series ou np.ndarray): Étiquettes de test.
        band_names (list): Liste des noms de toutes les bandes (caractéristiques).
        class_mapping (dict): Dictionnaire mappant les ID de classe (numériques) à leurs noms.
        output_dir (str): Répertoire de sortie pour les résultats.

    Returns:
        tuple: Tuple contenant le dictionnaire de métriques, le DataFrame de la
               matrice de confusion et l'objet LabelEncoder.
    """
    print("\n====| DÉMARRAGE DU PIPELINE XGBOOST |====")

    # ========================================================================
    #               PRÉ-TRAITEMENT : ENCODAGE & PONDÉRATION
    # ========================================================================
    print("Encodage des étiquettes (de N à 0..N-1)...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_encoded)

    # --- PASSE 1 : Analyse d'importance des caractéristiques ---
    evaluated_features = xgb_feature_selection(
        X_train, y_train_encoded, band_names, sample_weights, output_dir
    )
    final_features_for_model = evaluated_features

    # --- PASSE 2 : Entraînement et Optimisation du Modèle Final ---
    print("\n--- Entraînement et Optimisation du Modèle XGBoost ---")

    param_dist_xgb = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 15),
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    xgb_model_base = XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        seed=42,
        n_jobs=-1
    )

    print("Recherche des meilleurs hyperparamètres (n_iter=20)...")
    rand_search_xgb = RandomizedSearchCV(
        estimator=xgb_model_base,
        param_distributions=param_dist_xgb,
        n_iter=20,
        cv=5,
        random_state=42,
        scoring='accuracy',
        verbose=1
    )

    rand_search_xgb.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    final_model = rand_search_xgb.best_estimator_
    print(f"Meilleurs hyperparamètres : {rand_search_xgb.best_params_}")

    # --- Évaluation du modèle final ---
    print("\nÉvaluation du modèle sur le jeu de test...")
    y_pred_encoded = final_model.predict(X_test).astype(int)
    y_pred_proba = final_model.predict_proba(X_test)

    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    kappa = cohen_kappa_score(y_test_encoded, y_pred_encoded)

    print("Décodage des étiquettes pour les résultats...")
    y_test_original = le.inverse_transform(y_test_encoded)
    y_pred_original = le.inverse_transform(y_pred_encoded)

    metrics = {
        "Model": final_model,
        "Accuracy": accuracy,
        "Kappa": kappa,
        "Selected Features": final_features_for_model,
        'True Labels': y_test_original,
        'Predicted': y_pred_original,
        'Predicted Probabilities': y_pred_proba
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Kappa: {kappa:.4f}")

    # Création du DataFrame de la matrice de confusion avec les noms de classes
    display_class_names_for_cm = [class_mapping[original_id] for original_id in le.classes_]
    cm = confusion_matrix(y_test_encoded, y_pred_encoded, labels=le.classes_)
    conf_matrix_df = pd.DataFrame(cm, index=display_class_names_for_cm, columns=display_class_names_for_cm)

    output_csv_path = os.path.join(output_dir, 'confusion_matrix_XGB.csv')
    conf_matrix_df.to_csv(output_csv_path)
    print(f"\n--- Matrice de confusion (CSV) sauvegardée : {output_csv_path} ---")

    print("\n--- Pipeline XGBoost Terminé ---")

    return metrics, conf_matrix_df, le