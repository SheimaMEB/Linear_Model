"""
Module: visualization.py

Description:
Ce module fournit des fonctions pour visualiser des données en utilisant des graphiques Matplotlib. 
Les fonctions incluent l'affichage de multiples histogrammes, boxplots, nuages de points, heatmaps des corrélations, 
et la comparaison des prédictions avec les observations réelles.

Utilisation:
Ce module peut être utilisé pour visualiser différentes représentations graphiques de données en appelant les fonctions 
avec un DataFrame Pandas et les noms des colonnes d'intérêt.

Fonctions:
- plot_multiple_histograms(data, columns, file_name='multiple_histograms.png'): Affiche plusieurs histogrammes dans une seule image pour les colonnes spécifiées.
- plot_multiple_boxplots(data, columns, file_name='multiple_boxplots.png'): Affiche plusieurs boxplots dans une seule image pour les colonnes spécifiées.
- plot_scatter(data, x_column, y_column): Affiche un nuage de points pour deux colonnes spécifiées dans les données.
- plot_heatmap(data): Affiche une heatmap des corrélations entre les colonnes numériques du DataFrame.
- plot_predictions_vs_observations(y_true, y_pred, file_name='predictions_vs_observations.png'): Affiche les prédictions et les observations réelles sur un graphique et sauvegarde l'image.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_multiple_histograms(data, columns, file_name='multiple_histograms.png'):
    """
    Affiche plusieurs histogrammes dans une seule image pour les colonnes spécifiées.

    Paramètres :
    - data : DataFrame, les données
    - columns : list of str, liste des noms des colonnes pour lesquelles afficher les histogrammes
    - file_name : str, nom du fichier pour sauvegarder l'image
    """
    num_columns = len(columns)
    num_rows = (num_columns + 2) // 3  # Ajuster le nombre de lignes pour s'adapter aux colonnes

    fig, axes = plt.subplots(num_rows, 3, figsize=(10, 3 * num_rows))  # Ajuster la largeur et la hauteur
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, num_columns))

    for i, (column, color) in enumerate(zip(columns, colors)):
        if column not in data.columns:
            raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")
        
        values = data[column].dropna()
        axes[i].hist(values, bins=20, edgecolor='black', color=color)
        axes[i].set_xlabel(column, fontsize=8)
        axes[i].set_ylabel('Fréquence', fontsize=8)
        axes[i].set_title(f'Histogramme de {column}', fontsize=10)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)  # Ajustement des espacements
    plt.savefig(file_name, dpi=300)
    plt.show()

def plot_multiple_boxplots(data, columns, file_name='multiple_boxplots.png'):
    """
    Affiche plusieurs boxplots dans une seule image pour les colonnes spécifiées.

    Paramètres :
    - data : DataFrame, les données
    - columns : list of str, liste des noms des colonnes pour lesquelles afficher les boxplots
    - file_name : str, nom du fichier pour sauvegarder l'image
    """
    num_columns = len(columns)
    num_rows = (num_columns + 2) // 3  # Ajuster le nombre de lignes pour s'adapter aux colonnes

    fig, axes = plt.subplots(num_rows, 3, figsize=(10, 3 * num_rows))  # Ajuster la largeur et la hauteur
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, num_columns))

    for i, (column, color) in enumerate(zip(columns, colors)):
        if column not in data.columns:
            raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")
        
        values = data[column].dropna()
        axes[i].boxplot(values, patch_artist=True,
                        boxprops=dict(facecolor=color, color=color),
                        whiskerprops=dict(color=color),
                        capprops=dict(color=color),
                        medianprops=dict(color='black'))
        axes[i].set_xlabel(column, fontsize=8)
        axes[i].set_ylabel('Valeurs', fontsize=8)
        axes[i].set_title(f'Boxplot de {column}', fontsize=10)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)  # Ajustement des espacements
    plt.savefig(file_name, dpi=300)
    plt.show()

def plot_scatter(data, x_column, y_column):
    """
    Affiche un nuage de points pour deux colonnes spécifiées dans les données.

    Paramètres :
    - data : DataFrame, les données
    - x_column : str, nom de la colonne pour l'axe des abscisses
    - y_column : str, nom de la colonne pour l'axe des ordonnées
    """
    if x_column not in data.columns or y_column not in data.columns:
        raise ValueError(f"Les colonnes spécifiées ({x_column}, {y_column}) n'existent pas dans le DataFrame.")
    
    x_values = data[x_column].dropna()
    y_values = data[y_column].dropna()
    
    plt.figure(figsize=(4, 3))  # Taille de l'image plus petite
    plt.scatter(x_values, y_values, facecolors='none', edgecolors='black', s=20)  # Cercles non pleins, couleur noire, plus petits
    plt.xlabel(x_column, fontsize=8)
    plt.ylabel(y_column, fontsize=8)
    plt.title(f'Nuage de points entre {x_column} et {y_column}', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{x_column}_vs_{y_column}_nuage_de_point.png', dpi=300)
    plt.show()

def plot_heatmap(data):
    """
    Affiche une heatmap des corrélations entre les colonnes numériques du DataFrame.

    Paramètres :
    - data : DataFrame, les données
    """
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
    plt.title('Heatmap des corrélations')
    plt.tight_layout()
    plt.savefig('heatmap.png')
    plt.show()

def plot_predictions_vs_observations(y_true, y_pred, file_name='predictions_vs_observations.png'):
    """
    Affiche les prédictions et les observations réelles sur un graphique et sauvegarde l'image.

    Paramètres :
    - y_true : array-like, les valeurs réelles
    - y_pred : array-like, les valeurs prédites
    - file_name : str, nom du fichier pour sauvegarder l'image
    """
    plt.figure(figsize=(4, 3))
    plt.plot(y_true, label='Observations réelles', linestyle='--', color='black', linewidth=1.5)
    plt.plot(y_pred, label='Prédictions', linestyle='-', color='lightpink', linewidth=1.5)
    plt.xlabel('Échantillons', fontsize=8)
    plt.ylabel('Taux de Co2', fontsize=8)
    plt.title('Prédictions vs Observations réelles', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.show()
