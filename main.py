"""
Module: main.py

Description:
Ce module exécute une série d'analyses statistiques et de visualisations sur un jeu de données en utilisant les fonctions 
définies dans les modules `loading`, `statistics`, `visualization` et `regression`. Le script charge les données à partir 
d'un fichier CSV, effectue des analyses statistiques, génère des visualisations, ajuste un modèle de régression linéaire, 
et affiche les résultats.

Utilisation:
Ce module peut être exécuté directement. Il charge un fichier de données spécifié, effectue des calculs statistiques, 
génère des graphiques, et ajuste un modèle de régression linéaire. Les résultats sont imprimés dans la console et les 
graphiques sont sauvegardés en tant que fichiers PNG.

Fonctions:
- main(): Fonction principale qui exécute toutes les étapes de l'analyse des données, des visualisations et de la régression linéaire.
"""

import numpy as np
import pandas as pd
from Linearmodel.loading import load_data
from Linearmodel.statistics import calculate_mean, calculate_std, calculate_correlation, calculate_median, calculate_variance, calculate_mode, calculate_weighted_mode, summary, find_highly_correlated_variables
from Linearmodel.visualization import plot_multiple_boxplots, plot_scatter, plot_heatmap, plot_predictions_vs_observations, plot_multiple_histograms
from Linearmodel.regression import OrdinaryLeastSquares

def main():
    file_path = 'eCO2mix_RTE_Annuel-Definitif_2020.csv'
    data = load_data(file_path)

    # Vérifier les premières lignes du DataFrame
    print(data.head())

    # Vérifier les types de données et les valeurs manquantes
    print(data.info())
    print(data.describe())

    # Afficher les histogrammes pour toutes les colonnes numériques
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    plot_multiple_histograms(data, numeric_columns, 'multiple_histograms.png')

    # Afficher les boxplots pour toutes les colonnes
    plot_multiple_boxplots(data, numeric_columns, 'multiple_boxplots.png')

    # Calculer la moyenne de la colonne 'Solaire'
    mean_solaire = calculate_mean(data, 'Solaire')
    print(f"Moyenne de Solaire: {mean_solaire}")

    # Calculer l'écart type de la colonne 'Gaz'
    std_gaz = calculate_std(data, 'Gaz')
    print(f"Ecart type de la colonne 'Gaz': {std_gaz}")

    # Calculer la corrélation entre les colonnes 'Fioul' et 'Charbon'
    correlation_fioul_charbon = calculate_correlation(data, 'Fioul', 'Charbon')
    print(f"Corrélation entre 'Fioul' et 'Charbon': {correlation_fioul_charbon['Correlation']}")

    # Calculer la médiane de la colonne 'Consommation'
    median_consommation = calculate_median(data, 'Consommation')
    print(f"Médiane de la Consommation: {median_consommation}")

    # Calculer la variance de la colonne 'Hydraulique'
    variance_hydraulique = calculate_variance(data, 'Hydraulique')
    print(f"Variance de l'Hydraulique: {variance_hydraulique}")

    # Calculer le mode de la colonne 'Prévision J'
    mode_prevision_j = calculate_mode(data, 'Prévision J')
    print(f"Mode de la Prévision J: {mode_prevision_j}")

    # Calculer le mode pondéré de la colonne 'Prévion J-1'
    weighted_mode_prevision_j_1 = calculate_weighted_mode(data, 'Prévision J-1')
    print(f"Mode pondéré de la Prévion J-1 : {weighted_mode_prevision_j_1}")

    # Réaliser une analyse descriptive du DataFrame
    results = summary(data)

    # Afficher les résultats de l'analyse descriptive
    for column, stats in results.items():
        print(f"--- {column} ---")
        for stat_name, value in stats.items():
            print(f"{stat_name}: {value}")
        print()

    # Afficher et sauvegarder le nuage de points entre 'Consommation' et 'Gaz'
    plot_scatter(data, 'Fioul', 'Gaz')

    # Afficher et sauvegarder la heatmap des corrélations
    plot_heatmap(data)

    # Trouver les variables hautement corrélées avec 'Taux de Co2'
    target = 'Taux de Co2'
    correlated_variables = find_highly_correlated_variables(data, target)
    print(f"Variables hautement corrélées avec {target} : {correlated_variables}")

    # Sélectionner les colonnes X et la colonne y pour la régression
    X = data[correlated_variables].values  # Variables explicatives pour l'entraînement
    y = data[target].values  # Variable cible pour l'entraînement

    # Initialisation du modèle de régression linéaire avec intercept
    model = OrdinaryLeastSquares(intercept=True)

    # Entraînement du modèle
    model.fit(X, y)

    # Affichage des coefficients estimés
    coeffs = model.get_coeffs()
    print("Coefficients estimés:", coeffs)

    # Calcul du coefficient de détermination R^2
    r_squared = model.determination_coefficient(X, y)
    print("Coefficient de détermination R^2:", r_squared)

    # Prédictions pour les données existantes
    y_pred = model.predict(X)

    # Affichage des prédictions et des observations réelles sur un graphique
    plot_predictions_vs_observations(y, y_pred, 'predictions_vs_observations.png')

if __name__ == "__main__":
    main()
