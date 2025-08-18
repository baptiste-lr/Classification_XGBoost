# -*- coding: utf-8 -*-
# ==============================================================================
# Objet :
#   - Ingénierie de caractéristiques pour l'imagerie satellite.
#   - Enrichit un raster source avec des indices spectraux et de texture.
# Description :
#   Ce module génère des couches d'information supplémentaires (indices) à partir
#   des bandes spectrales d'un raster. Ces nouvelles caractéristiques visent à
#   accentuer des propriétés biophysiques (végétation, eau, sol) et à capturer
#   la variabilité locale pour améliorer les performances des modèles de
#   classification par apprentissage automatique.
# ==============================================================================

import rasterio
import numpy as np
from scipy.ndimage import generic_filter


def creer_raster_multibandes(chemin_raster_4_bandes: str):
    """
    Calcule des indices spectraux et des indices de texture et les empile avec les bandes originales.

    Args:
        chemin_raster_4_bandes (str): Chemin d'accès au fichier GeoTIFF source à 4 bandes (Bleu, Vert, Rouge, Proche Infrarouge).

    Returns:
        tuple: Un tuple contenant :
            - stacked_data (numpy.ndarray): Un tableau NumPy multi-bandes des caractéristiques empilées.
            - profile (dict): Les métadonnées Rasterio mises à jour pour le nouveau raster.
            - band_names (list): La liste ordonnée des noms des bandes.
    """
    print("--- Démarrage de l'ingénierie des caractéristiques ---")

    with rasterio.open(chemin_raster_4_bandes) as src:
        if src.count != 4:
            raise ValueError(f"Le raster d'entrée doit avoir 4 bandes, mais il en a {src.count}.")

        nodata_value = src.nodata if src.nodata is not None else 255
         
        profile = src.profile
        print("Lecture des 4 bandes originales (B, G, R, PIR)...")
        B = src.read(1).astype('float32')
        B[B == nodata_value] = np.nan
        G = src.read(2).astype('float32')
        G[G == nodata_value] = np.nan
        R = src.read(3).astype('float32')
        R[R == nodata_value] = np.nan
        PIR = src.read(4).astype('float32')
        PIR[PIR == nodata_value] = np.nan
         
        # --- Calcul des indices spectraux ---
        with np.errstate(divide='ignore', invalid='ignore'):
            print("Calcul des indices spectraux...")
            NDVI = (PIR - R) / (PIR + R) # Normalized Difference Vegetation Index
            MNDWI = (G - PIR) / (G + PIR) # Modified Normalized Difference Water Index
            ExG = (2 * G - R - B) / (R + G + B) # Excess Green Index
             
            # Formule MSAVI (Modified Soil Adjusted Vegetation Index)
            term_sous_sqrt = (2 * PIR + 1)**2 - 8 * (PIR - R)
            term_sous_sqrt[term_sous_sqrt < 0] = 0
            MSAVI = (2 * PIR + 1 - np.sqrt(term_sous_sqrt)) / 2

            L = R + G + B # Luminance
            C3 = (G - R) / (G + R) + B # Band Ratio 
            BII = (R + G - B) / PIR # Brightness Index
            UAI = (R - PIR) / (R + PIR) # Urban Area Index

        # --- Calcul des indices de texture par variance (fenêtre 3x3) ---
        print("Calcul des indices de texture par variance (fenêtre 3x3)...")
        texture_red_3x3 = generic_filter(R, np.var, size=3)
        texture_red_3x3[np.isnan(R)] = nodata_value
        texture_pir_3x3 = generic_filter(PIR, np.var, size=3)
        texture_pir_3x3[np.isnan(PIR)] = nodata_value
         
        # --- Traitement des valeurs NaN/inf et empilement ---
        indices_calcules = [NDVI, MNDWI, ExG, MSAVI, L, C3, BII, UAI,
                            texture_red_3x3, texture_pir_3x3]
        for index in indices_calcules:
            index[np.isinf(index)] = nodata_value
            index[np.isnan(index)] = nodata_value
             
        B[np.isnan(B)] = nodata_value
        G[np.isnan(G)] = nodata_value
        R[np.isnan(R)] = nodata_value
        PIR[np.isnan(PIR)] = nodata_value

        print("Empilement des bandes et des nouvelles caractéristiques...")
        band_names = ["B", "G", "R", "PIR", "C3", "L", "MNDWI", "MSAVI", "NDVI", "ExG", "BII", "UAI",
                      "Texture_R_3x3", "Texture_PIR_3x3"]
        stacked_data = np.array([
            B, G, R, PIR,
            C3, L, MNDWI, MSAVI, NDVI, ExG,
            BII, UAI,
            texture_red_3x3, texture_pir_3x3,
        ])
         
        # --- Mise à jour des métadonnées pour le raster de sortie ---
        profile.update({
            'count': len(band_names),
            'dtype': 'float32',
            'descriptions': band_names,
            'nodata': nodata_value
        })

        print(f"--- Ingénierie des caractéristiques terminée. {len(band_names)} bandes prêtes. ---")
        return stacked_data, profile, band_names