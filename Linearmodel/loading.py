"""
Module: loading.py

Description:
Ce module fournit une fonction pour charger et préparer des données à partir d'un fichier CSV en utilisant Pandas. 
Il convertit les colonnes de date et d'heure en un format datetime, remplace les valeurs manquantes par 0, et supprime les colonnes originales de date et d'heure.

Utilisation:
Ce module peut être utilisé pour charger et préparer des données CSV en appelant la fonction `load_data` avec le chemin vers le fichier CSV.

Fonctions:
- load_data(file_path): Charge les données à partir d'un fichier CSV, convertit les colonnes de date et d'heure en datetime, remplace les valeurs manquantes par 0, et supprime les colonnes originales de date et d'heure.
"""

import pandas as pd

def load_data(file_path):
    """
    Charge les données à partir d'un fichier CSV en utilisant Pandas, convertit les colonnes de date et d'heure,
    et remplace les valeurs manquantes par 0.

    Paramètres :
    - file_path : str, chemin vers le fichier CSV

    Retourne :
    - DataFrame contenant les données chargées avec les colonnes de date et d'heure converties
      et les valeurs manquantes remplacées par 0
    """
    # Charger les données depuis le fichier CSV
    df = pd.read_csv(file_path, sep=';', na_values=['', ' '])

    # Convertir Date et Heures en datetime
    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Heures'], format='%Y-%m-%d %H:%M'
    )

    # Remplacer les valeurs manquantes par 0
    df.fillna(0, inplace=True)

    # Supprimer les anciennes colonnes Date et Heures
    df.drop(columns=['Date', 'Heures'], inplace=True)

    return df

