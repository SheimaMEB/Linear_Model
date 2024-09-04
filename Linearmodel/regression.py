"""
Module: regression.py

Description:
Ce module implémente une classe pour effectuer la régression linéaire en utilisant la méthode des moindres carrés ordinaires (Ordinary Least Squares, OLS). 
Il permet d'ajuster un modèle linéaire aux données, de faire des prédictions, et de calculer le coefficient de détermination (R^2).

Utilisation:
Ce module peut être utilisé pour ajuster un modèle de régression linéaire aux données en utilisant la classe `OrdinaryLeastSquares`. 
Vous pouvez créer une instance de cette classe, ajuster le modèle avec les données d'entraînement, prédire les valeurs pour de nouvelles données, et obtenir les coefficients du modèle.

Classe:
- OrdinaryLeastSquares: Classe pour effectuer la régression linéaire en utilisant les moindres carrés ordinaires.

Méthodes:
- __init__(self, intercept=True): Initialise le modèle des moindres carrés ordinaires.
- fit(self, X, y): Calcule les coefficients des moindres carrés ordinaires.
- predict(self, X): Prédit les valeurs de y pour une nouvelle matrice de données X.
- get_coeffs(self): Retourne les coefficients estimés du modèle.
- determination_coefficient(self, X, y): Calcule le coefficient de détermination R^2.
"""

import numpy as np

class OrdinaryLeastSquares:
    def __init__(self, intercept=True):
        """
        Initialise le modèle des moindres carrés ordinaires.

        Parameters:
        - intercept: bool, indique s'il faut ajouter une constante au modèle.
        """
        self.intercept = intercept
        self.coeffs = None

    def fit(self, X, y):
        """
        Calcule les coefficients des moindres carrés ordinaires.

        Parameters:
        - X: ndarray, matrice des variables explicatives (n_samples, n_features)
        - y: ndarray, vecteur des réponses (n_samples,)
        """
        if self.intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Estimation des coefficients β̂ = (X^T X)^-1 X^T y
        X_transpose = X.T
        self.coeffs = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

    def predict(self, X):
        """
        Prédit les valeurs de y pour une nouvelle matrice de données X.

        Parameters:
        - X: ndarray, matrice des variables explicatives (n_samples, n_features)

        Returns:
        - ndarray, vecteur des prédictions (n_samples,)
        """
        if self.intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        return X @ self.coeffs

    def get_coeffs(self):
        """
        Retourne les coefficients estimés du modèle.

        Returns:
        - ndarray, les coefficients estimés (n_features,)
        """
        return self.coeffs

    def determination_coefficient(self, X, y):
        """
        Calcule le coefficient de détermination R^2.

        Parameters:
        - X: ndarray, matrice des variables explicatives (n_samples, n_features)
        - y: ndarray, vecteur des réponses (n_samples,)

        Returns:
        - float, le coefficient de détermination R^2
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        return r_squared
