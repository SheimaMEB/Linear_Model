# Linearmodel

## Description

`Linearmodel` est un package Python conçu pour effectuer des analyses statistiques et des visualisations, ainsi que pour appliquer des modèles de régression linéaire. Ce package est structuré pour faciliter l'analyse de données et la modélisation statistique.

## Installation

Pour installer ce package, utilisez `pip` :

```bash
pip install Linearmodel/dist/Linearmodel-0.1.tar.gz

## Exemple d'utilisation

from Linearmodel.loading import load_data
from Linearmodel.statistics import calculate_mean, calculate_std, calculate_correlation, calculate_median, calculate_variance, calculate_mode, calculate_weighted_mode, summary, find_highly_correlated_variables
from Linearmodel.visualization import plot_multiple_boxplots, plot_scatter, plot_heatmap, plot_predictions_vs_observations, plot_multiple_histograms
from Linearmodel.regression import OrdinaryLeastSquares

### Charger les données
data = load_data('path/to/your/data.csv')

### Calculer la moyenne d'une colonne
mean_value = calculate_mean(data, 'column_name')

### Visualiser les données
plot_multiple_histograms(data, ['column1', 'column2'])

### Effectuer une régression linéaire
model = OrdinaryLeastSquares()
X = data[['feature1', 'feature2']].values
y = data['target'].values
model.fit(X, y)
predictions = model.predict(X)


## Fonctionnalités

###loading.py

    -load_data(file_path): Charge les données à partir dun fichier CSV en utilisant Pandas.

###statistics.py

    -calculate_mean(data, column): Calcule la moyenne dune colonne spécifiée.
    -calculate_std(data, column): Calcule lécart type dune colonne spécifiée.
    -calculate_correlation(data, column1, column2): Calcule la corrélation entre deux colonnes.
    -calculate_median(data, column): Calcule la médiane dune colonne spécifiée.
    -calculate_variance(data, column): Calcule la variance dune colonne spécifiée.
    -calculate_mode(data, column): Calcule le mode dune colonne spécifiée.
    -calculate_weighted_mode(data, column): Calcule le mode pondéré dune colonne spécifiée.
    -summary(data): Réalise une analyse descriptive du DataFrame.
    -find_highly_correlated_variables(data, target, threshold): Trouve les variables hautement corrélées avec la variable cible.

###visualization.py

    -plot_multiple_histograms(data, columns, file_name): Affiche plusieurs histogrammes dans une seule image pour les colonnes spécifiées.
    -plot_multiple_boxplots(data, columns, file_name): Affiche plusieurs boxplots dans une seule image pour les colonnes spécifiées.
    -plot_scatter(data, x_column, y_column): Affiche un nuage de points pour deux colonnes spécifiées dans les données.
    -plot_heatmap(data): Affiche une heatmap des corrélations entre les colonnes numériques du DataFrame.
    -plot_predictions_vs_observations(y_true, y_pred, file_name): Affiche les prédictions et les observations réelles sur un graphique et sauvegarde limage.

###regression.py

    -OrdinaryLeastSquares: Classe pour effectuer une régression linéaire par la méthode des moindres carrés ordinaires.
        -__init__(self, intercept=True): Initialise le modèle des moindres carrés ordinaires.
        -fit(self, X, y): Calcule les coefficients des moindres carrés ordinaires.
        -predict(self, X): Prédit les valeurs de y pour une nouvelle matrice de données X.
        -get_coeffs(self): Retourne les coefficients estimés du modèle.
        -determination_coefficient(self, X, y): Calcule le coefficient de détermination R^2.
    
    
###__init__.py

    Fichier dinitialisation du package permettant dimporter les modules disponibles dans Linearmodel.

###setup.py

    Fichier de configuration pour installer le package avec pip.
    
    
## Tests 

Pour exécuter les tests unitaires, utilisez pytest : pytest Tests/ #Commande à éxecuter sur le terminal


Sheïma MEBARKA.
