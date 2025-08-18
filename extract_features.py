# -*- coding: utf-8 -*-
# ==============================================================================
# Objet :
#   - Extraction de valeurs de pixels pour l'entraînement d'un modèle ML.
# Description :
#   Ce module fournit une fonction robuste pour extraire les valeurs de pixels
#   d'un raster à partir de géométries de polygones stockées dans un fichier vecteur.
#   Les données extraites sont formatées pour la préparation d'ensembles de
#   données d'apprentissage supervisé en géomatique.
# ==============================================================================

import rasterio
import fiona
import pandas as pd
from rasterio.mask import mask
import numpy as np


def extract_pixels_from_polygons(shapefile_path, raster_path, label_col="id", class_name_col="Classe", return_coords=False):
    """
    Extrait les valeurs de pixels d'un raster qui sont couverts par des polygones
    d'un fichier vecteur.

    Args:
        shapefile_path (str): Chemin du fichier shapefile des polygones d'entraînement.
        raster_path (str): Chemin du fichier raster source.
        label_col (str, optional): Nom de la colonne dans le shapefile contenant l'identifiant numérique de la classe. Par défaut 'id'.
        class_name_col (str, optional): Nom de la colonne contenant le nom de la classe. Par défaut 'Classe'.
        return_coords (bool, optional): Si True, inclut les coordonnées (x,y) de chaque pixel dans le DataFrame. Par défaut False.

    Returns:
        pd.DataFrame: Un DataFrame contenant les valeurs des pixels, leur identifiant
                      de classe, et, si demandé, leurs coordonnées.
    """
    with rasterio.open(raster_path) as src:
        band_names = [desc[0] if isinstance(desc, tuple) else desc for desc in src.descriptions]
        print(f"Extraction des valeurs pixel par pixel pour les bandes : {band_names}")

    pixels_df = []

    with fiona.open(shapefile_path, "r") as shapefile:
        with rasterio.open(raster_path) as src:
            for feature in shapefile:
                geom = [feature['geometry']]
                label = feature['properties'][label_col]
                class_name = feature['properties'][class_name_col]

                out_image, out_transform = mask(src, geom, crop=True, nodata=src.nodata)
                
                # Masquer les pixels nodata pour éviter leur inclusion
                if src.nodata is not None:
                    valid_pixels_mask = (out_image != src.nodata).all(axis=0)
                    out_image = out_image[:, valid_pixels_mask]
                
                # Préparation des coordonnées si nécessaire
                if return_coords:
                    height, width = out_image.shape[1], out_image.shape[2]
                    rows, cols = np.where(valid_pixels_mask)
                    transform_inv = ~out_transform
                    xs, ys = transform_inv * (cols, rows)
                
                # Itération sur les pixels valides pour construire le DataFrame
                for i in range(out_image.shape[1]):
                    row_data = {
                        "id": label,
                        "Classe": class_name
                    }
                    for band_idx, value in enumerate(out_image[:, i]):
                        row_data[band_names[band_idx]] = value
                    if return_coords:
                        row_data['x'] = xs[i]
                        row_data['y'] = ys[i]
                    pixels_df.append(row_data)

    return pd.DataFrame(pixels_df)