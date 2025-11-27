"""
Random Forest Classifier Implementation

A custom Random Forest implementation using sklearn's ExtraTreeClassifier
as base estimators. Includes bootstrapping, out-of-bag scoring, 
hyperparameter grid search, and AdaBoost ensemble methods.

Author: Aarush Chhiber
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import ExtraTreeClassifier


class RandomForest:
    """
    Random Forest classifier using bootstrap aggregating (bagging).
    
    Combines multiple ExtraTreeClassifier estimators trained on 
    bootstrapped samples to reduce overfitting and improve predictions.
    
    Features:
        - Bootstrap sampling with feature subsampling
        - Out-of-bag (OOB) score evaluation
        - Hyperparameter grid search
        - AdaBoost ensemble support
        - Feature importance visualization
    
    Attributes:
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of each tree
        max_features (float): Fraction of features to use per tree
        decision_trees (list): List of fitted ExtraTreeClassifier instances
    """
    
    def __init__(self, n_estimators, max_depth, max_features, random_seed=None):
        """
        Initialize the Random Forest.
        
        Args:
            n_estimators (int): Number of trees in the ensemble
            max_depth (int): Maximum depth for each tree
            max_features (float): Fraction of features to sample (0-1)
            random_seed (int, optional): Random seed for reproducibility
        """
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.max_features = max_features
        self.random_seed = random_seed
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []
        self.decision_trees = [
            ExtraTreeClassifier(max_depth=self.max_depth, criterion="entropy")
            for _ in range(self.n_estimators)
        ]
        self.alphas = []  # Weights for AdaBoost

    def _bootstrapping(self, num_training, num_features, random_seed=None):
        """
        Create bootstrap sample indices.
        
        Generates row indices with replacement and column indices
        without replacement for training each tree.
        
        Args:
            num_training (int): Total number of training samples
            num_features (int): Total number of features
            random_seed (int, optional): Random seed for this bootstrap
            
        Returns:
            tuple: (row_indices, col_indices) for bootstrap sample
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        row_idx = np.random.choice(num_training, size=num_training, replace=True)
        n_cols = int(self.max_features * num_features) if self.max_features <= 1 else int(self.max_features)
        n_cols = max(1, n_cols)  # Ensure at least one feature
        col_idx = np.random.choice(num_features, size=n_cols, replace=False)
        
        return row_idx, col_idx

    def bootstrapping(self, num_training, num_features):
        """
        Initialize bootstrap datasets for all trees.
        
        Creates bootstrap samples and tracks out-of-bag samples
        for each tree in the ensemble.
        
        Args:
            num_training (int): Number of training samples
            num_features (int): Number of features
        """
        np.random.seed(self.random_seed)
        
        for i in range(self.n_estimators):
            total = set(list(range(num_training)))
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            total = total - set(row_idx)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)

    def fit(self, X, y):
        """
        Fit the Random Forest to training data.
        
        Creates bootstrap samples and trains each tree on its
        respective bootstrap dataset.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Training labels of shape (n_samples,)
        """
        num_training, num_features = X.shape
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []
        
        np.random.seed(self.random_seed)
        
        for i in range(self.n_estimators):
            total = set(range(num_training))
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            total = total - set(row_idx)
            
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)
            
            # Train on bootstrap sample
            X_boot = X[row_idx][:, col_idx]
            y_boot = y[row_idx]
            self.decision_trees[i].fit(X_boot, y_boot)

    def adaboost(self, X, y):
        """
        Train using AdaBoost (Adaptive Boosting).
        
        Sequentially trains weak learners, adjusting sample weights
        based on previous errors to focus on difficult examples.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
        """
        N = X.shape[0]
        self.alphas = []
        weights = np.ones(N) / N
        
        for i, tree in enumerate(self.decision_trees):
            # Train tree with sample weights
            tree.fit(X, y, sample_weight=weights)
            pred = tree.predict(X)
            
            # Calculate weighted error
            incorrect = (pred != y)
            error = np.sum(weights * incorrect) / np.sum(weights)
            error = np.clip(error, 1e-10, 1 - 1e-10)  # Avoid log(0)
            
            # Calculate classifier weight
            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)
            
            # Update sample weights
            weights *= np.exp(-alpha * (pred == y) + alpha * (pred != y))
            weights /= np.sum(weights)

    def OOB_score(self, X, y):
        """
        Calculate out-of-bag accuracy score.
        
        Uses samples not included in each tree's bootstrap sample
        to estimate generalization accuracy.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            
        Returns:
            float: Mean OOB accuracy
        """
        accuracy = []
        
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    predictions.append(
                        self.decision_trees[t].predict(
                            np.reshape(X[i][self.feature_indices[t]], (1, -1))
                        )[0]
                    )
            if len(predictions) > 0:
                accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
        
        return np.mean(accuracy)

    def predict(self, X):
        """
        Make predictions using the ensemble.
        
        Aggregates probability predictions from all trees
        and returns the class with highest total probability.
        
        Args:
            X (np.ndarray): Features of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted labels of shape (n_samples,)
        """
        N = X.shape[0]
        y = np.zeros((N, 7))
        
        for t in range(self.n_estimators):
            X_curr = X[:, self.feature_indices[t]]
            y += self.decision_trees[t].predict_proba(X_curr)
        
        pred = np.argmax(y, axis=1)
        return pred

    def predict_adaboost(self, X):
        """
        Make predictions using AdaBoost weighted voting.
        
        Aggregates weighted votes from all classifiers based
        on their alpha (importance) values.
        
        Args:
            X (np.ndarray): Features
            
        Returns:
            np.ndarray: Predicted labels
        """
        N = X.shape[0]
        weighted_votes = np.zeros((N, 7))

        for alpha, tree in zip(self.alphas, self.decision_trees[:len(self.alphas)]):
            pred = tree.predict(X)
            for i in range(N):
                class_index = int(pred[i])
                weighted_votes[i, class_index] += alpha

        return np.argmax(weighted_votes, axis=1)

    def plot_feature_importance(self, data_train):
        """
        Visualize feature importances from the first tree.
        
        Creates a bar chart showing the relative importance of
        each feature based on Gini impurity reduction.
        
        Args:
            data_train: Training data (DataFrame or array) for feature names
        """
        tree = self.decision_trees[0]
        importances = tree.feature_importances_
        
        # Get feature names
        if hasattr(data_train, 'columns'):
            feature_names = data_train.columns[:-1] if data_train.shape[1] == len(importances) + 1 else data_train.columns
        else:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        # Filter non-zero importances
        nonzero_mask = importances > 0
        importances = importances[nonzero_mask]
        feature_names = np.array(feature_names)[nonzero_mask]
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        importances = importances[sorted_idx]
        feature_names = feature_names[sorted_idx]
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.bar(feature_names, importances)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.title("Feature Importances (Gini)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def hyperparameter_grid_search(
        self, n_estimators_range, max_depth_range, max_features_range
    ):
        """
        Generate hyperparameter combinations for grid search.
        
        Creates all combinations of hyperparameters within
        the specified ranges for exhaustive search.
        
        Args:
            n_estimators_range (tuple): (start, stop, step) for n_estimators
            max_depth_range (tuple): (start, stop, step) for max_depth
            max_features_range (tuple): (start, stop, step) for max_features
            
        Returns:
            list: List of (n_estimators, max_depth, max_features) tuples
        """
        n_start, n_stop, n_step = n_estimators_range
        d_start, d_stop, d_step = max_depth_range
        f_start, f_stop, f_step = max_features_range
        
        n_vals = np.arange(n_start, n_stop + n_step, n_step)
        d_vals = np.arange(d_start, d_stop + d_step, d_step)
        
        # Handle float stepping for max_features
        f_vals = []
        val = f_start
        while val <= f_stop + 1e-8:
            f_vals.append(round(val, 10))
            val += f_step
        
        # Generate all combinations
        grid = []
        for n in n_vals:
            for d in d_vals:
                for f in f_vals:
                    grid.append((int(n), int(d), f))
        
        return grid
