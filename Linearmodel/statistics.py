"""
Module: statistics.py

Description:
Ce module fournit des fonctions pour effectuer des calculs statistiques sur des données contenues dans un DataFrame Pandas. 
Les fonctions incluent le calcul de la moyenne, de l'écart type, de la corrélation, de la médiane, de la variance, du mode, 
du mode pondéré, et une analyse descriptive des données.

Utilisation:
Ce module peut être utilisé pour analyser des jeux de données en appelant les fonctions avec un DataFrame Pandas et le nom de la colonne d'intérêt.

Fonctions:
- calculate_mean(data, column): Calcule la moyenne d'une colonne spécifiée dans les données.
- calculate_std(data, column): Calcule l'écart type d'une colonne spécifiée dans les données.
- calculate_correlation(data, column1, column2): Calcule la corrélation entre deux colonnes spécifiées dans les données.
- calculate_median(data, column): Calcule la médiane d'une colonne spécifiée dans les données.
- calculate_variance(data, column): Calcule la variance d'une colonne spécifiée dans les données.
- calculate_mode(data, column): Calcule le mode d'une colonne spécifiée dans les données.
- calculate_weighted_mode(data, column): Calcule le mode pondéré d'une colonne spécifiée dans les données.
- summary(data): Réalise une analyse descriptive du DataFrame.
- find_highly_correlated_variables(data, target, threshold=0.55): Trouve toutes les variables dans le DataFrame qui ont une corrélation supérieure au seuil donné avec la variable cible.
"""

import numpy as np

def calculate_mean(data, column):
    """
    Calcule la moyenne d'une colonne spécifiée dans les données.

    Parameters:
    - data: DataFrame, les données
    - column: str, le nom de la colonne pour calculer la moyenne

    Returns:
    - float, la moyenne de la colonne
    """
    values = data[column].dropna()
    return sum(values) / len(values) if len(values) > 0 else 0.0

def calculate_std(data, column):
    """
    Calcule l'écart type d'une colonne spécifiée dans les données.

    Parameters:
    - data: DataFrame, les données
    - column: str, le nom de la colonne pour calculer l'écart type

    Returns:
    - float, l'écart type de la colonne
    """
    values = data[column].dropna()
    mean = calculate_mean(data, column)
    variance = sum((x - mean) ** 2 for x in values) / len(values) if len(values) > 0 else 0.0
    return variance ** 0.5

def calculate_correlation(data, column1, column2):
    """
    Calcule la corrélation entre deux colonnes spécifiées dans les données.

    Parameters:
    - data: DataFrame, les données
    - column1: str, le nom de la première colonne
    - column2: str, le nom de la deuxième colonne

    Returns:
    - dict, un dictionnaire contenant la corrélation entre les deux colonnes
    """
    values1 = data[column1].dropna()
    values2 = data[column2].dropna()

    mean1 = calculate_mean(data, column1)
    mean2 = calculate_mean(data, column2)

    numerator = sum((x1 - mean1) * (x2 - mean2) for x1, x2 in zip(values1, values2))
    denominator = (sum((x1 - mean1) ** 2 for x1 in values1) * sum((x2 - mean2) ** 2 for x2 in values2)) ** 0.5

    correlation = numerator / denominator if denominator != 0 else 0.0

    return {'Correlation': correlation}


def calculate_median(data, column):
    """
    Calcule la médiane d'une colonne spécifiée dans les données.

    Parameters:
    - data: DataFrame, les données
    - column: str, le nom de la colonne pour calculer la médiane

    Returns:
    - float, la médiane de la colonne
    """
    values = [row[column] for row in data.to_dict('records') if row[column] is not None]
    values.sort()
    n = len(values)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return values[n // 2]
    else:
        mid1 = values[n // 2 - 1]
        mid2 = values[n // 2]
        return (mid1 + mid2) / 2.0
    
    

def calculate_variance(data, column):
    """
    Calcule la variance d'une colonne spécifiée dans les données.

    Parameters:
    - data: DataFrame, les données
    - column: str, le nom de la colonne pour calculer la variance

    Returns:
    - float, la variance de la colonne
    """
    values = [row[column] for row in data.to_dict('records') if row[column] is not None]
    mean = calculate_mean(data, column)
    variance = sum((x - mean) ** 2 for x in values) / len(values) if values else 0.0
    return variance



def calculate_mode(data, column):
    """
    Calcule le mode d'une colonne spécifiée dans les données.

    Parameters:
    - data: DataFrame, les données
    - column: str, le nom de la colonne pour calculer le mode

    Returns:
    - float or int or str, le mode de la colonne
    """
    values = data[column].dropna()  # Exclure les valeurs NaN
    value_counts = {}
    
    # Compter les occurrences de chaque valeur dans la colonne
    for value in values:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1
    
    # Trouver la valeur avec le plus grand nombre d'occurrences (le mode)
    mode_value = None
    max_count = 0
    
    for value, count in value_counts.items():
        if count > max_count:
            max_count = count
            mode_value = value
    
    return mode_value


def calculate_weighted_mode(data, column):
    """
    Calcule le mode pondéré d'une colonne spécifiée dans les données.

    Parameters:
    - data: DataFrame, les données
    - column: str, le nom de la colonne pour calculer le mode pondéré

    Returns:
    - float or str, le mode pondéré de la colonne
    """
    if column not in data.columns:
        raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")

    # Compter les occurrences de chaque valeur dans la colonne
    value_counts = data[column].value_counts()

    # Obtenir la valeur la plus fréquente (mode)
    mode_value = value_counts.idxmax()

    # Vérifier si le mode est unique ou s'il y a une égalité de fréquence
    if len(value_counts[value_counts == value_counts.max()]) > 1:
        # S'il y a égalité, calculer le mode pondéré
        weights = data[column].map(value_counts)
        weighted_mode = sum(data[column] * weights) / sum(weights)
        return weighted_mode
    else:
        return mode_value

    
def summary(data):
    """
    Réalise une analyse descriptive du DataFrame.

    Parameters:
    - data: DataFrame, les données à analyser

    Returns:
    - dict, un dictionnaire contenant les résultats des calculs statistiques
    """
    summary_dict = {}
    
    # Calcul des statistiques pour chaque colonne numérique
    for column in data.columns:
        if np.issubdtype(data[column].dtype, np.number):
            summary_dict[column] = {
                'Mean': calculate_mean(data, column),
                'Std': calculate_std(data, column),
                'Median': calculate_median(data, column),
                'Variance': calculate_variance(data, column),
                'Mode': calculate_mode(data, column),
                'Weighted Mode': calculate_weighted_mode(data, column)
            }
    
    # Calcul de la corrélation entre les paires de colonnes numériques
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            col1 = numeric_columns[i]
            col2 = numeric_columns[j]
            correlation = calculate_correlation(data, col1, col2)
            summary_dict[f'Correlation {col1} vs {col2}'] = correlation
    
    return summary_dict


def find_highly_correlated_variables(data, target, threshold=0.55):
    """
    Trouve toutes les variables dans le DataFrame qui ont une corrélation supérieure au seuil donné avec la variable cible.

    Paramètres :
    - data : DataFrame, le jeu de données complet
    - target : str, la variable cible
    - threshold : float, le seuil de corrélation

    Retourne :
    - List[str], les noms des variables hautement corrélées
    """
    correlations = data.corr()[target]
    highly_correlated = correlations[correlations.abs() > threshold].index.tolist()
    highly_correlated.remove(target)  # Supprimer la variable cible de la liste
    return highly_correlated

