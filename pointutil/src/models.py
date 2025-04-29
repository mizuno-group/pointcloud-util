# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

latent variable model module

@author: tadahaya
"""
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

class DPGMMFeatureExtractor:
    """
    Extract features from point-cloud data using Dirichlet Process Gaussian Mixture Models (DPGMM).

    Automatically determines the optimal number of clusters from data and
    provides meaningful features for downstream machine learning.

    Parameters
    ----------
    max_components : int, default=10
        The maximum number of mixture components (clusters) considered.

    weight_concentration_prior : float, default=0.1
        Controls the tendency to create new clusters. Lower values prefer fewer clusters.

    effective_threshold : float, default=0.01
        Threshold to determine significant clusters based on their weights.

    random_state : int, default=42
        Random seed for reproducibility.
    """
    def __init__(
            self, max_components=10, weight_concentration_prior=0.1, effective_threshold=0.01,
            max_iter=1000, random_state=42
            ):
        self.max_components = max_components
        self.weight_concentration_prior = weight_concentration_prior
        self.effective_threshold = effective_threshold
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def fit(self, X):
        """
        Fit the DPGMM model to the point-cloud data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data points.

        Returns
        -------
        self : object
            Fitted extractor instance.
        """
        self.model = BayesianGaussianMixture(
            n_components=self.max_components,
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=self.weight_concentration_prior,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model.fit(X)
        return self

    def transform(self, X, aggregation=["mean"], only_significant=True):
        """
        Compute cluster responsibility vectors as features for input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data points.

        aggregation : list of str, default=["mean"]
            aggregation methods to apply to the cluster responsibilities.
            mean, std, entropy, or combination of these.
            if multiple methods are provided, the concatenated features are returned.
            the order is mean, std, then entropy

        only_significant : bool, default=False
            If True, returns responsibility only for clusters whose weight exceeds effective_threshold.

        Returns
        -------
        features : np.ndarray of shape (n_samples, n_clusters or n_significant_clusters)
            Responsibility vectors for each data point. If only_significant is True,
            vectors are re-normalized to sum to 1.
        """
        if self.model is None:
            raise RuntimeError("The model must be fitted before calling transform().")
        responsibilities = self.model.predict_proba(X)
        if only_significant:
            significant_clusters = self.model.weights_ > self.effective_threshold
            responsibilities = responsibilities[:, significant_clusters]
            # Renormalize responsibilities to sum to 1 for each sample
            sum_responsibilities = responsibilities.sum(axis=1, keepdims=True)
            # Avoid division by zero
            sum_responsibilities[sum_responsibilities == 0] = 1.0
            responsibilities = responsibilities / sum_responsibilities
        features = []
        if "mean" in aggregation:
            mean_features = responsibilities.mean(axis=0)
            features.append(mean_features)
        if "std" in aggregation:
            std_features = responsibilities.std(axis=0)
            features.append(std_features)
        if "entropy" in aggregation:
            entropy_features = -np.sum(responsibilities * np.log(responsibilities + 1e-8), axis=1)
            features.append([np.mean(entropy_features)])
        return np.concatenate(features)


    def extract_global_features(self):
        """
        Extract global summary features (cluster centers, covariances, weights) from the fitted model.

        Returns
        -------
        global_features : dict
            Dictionary containing:
                - 'centers': cluster means (np.ndarray of shape (n_clusters, n_features))
                - 'covariances': covariance matrices (np.ndarray of shape (n_clusters, n_features, n_features))
                - 'weights': cluster weights (np.ndarray of shape (n_clusters,))
                - 'n_effective_clusters': number of clusters with significant weight (>0.01)
        """
        if self.model is None:
            raise RuntimeError("The model must be fitted before extracting features.")

        significant = self.model.weights_ > self.effective_threshold
        global_features = {
            'centers': self.model.means_[significant],
            'covariances': self.model.covariances_[significant],
            'weights': self.model.weights_[significant],
            'n_effective_clusters': significant.sum()
        }
        return global_features