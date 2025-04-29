# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

denoising module

@author: tadahaya
"""

import numpy as np
import math
from scipy.spatial import KDTree
from scipy.stats import poisson

def poisson_denoise(
    data, 
    percent_radius=0.1, 
    p_threshold=0.05, 
    min_neighbors=2,
    expected_lambda=None
):
    """
    Denoise points based on Poisson density thresholding.

    Each point is examined to see if its neighborhood density is significantly lower
    than the expected density assuming a Poisson distribution. Points with too few neighbors
    are classified as noise.

    Parameters
    ----------
    data : np.ndarray
        Data points, shape (N, D) where N is the number of points and D is dimension.
    
    percent_radius : float or list of float, default=0.1
        Radius defined as a percentage of data range per dimension. 
        If a single float, the same percentage is applied across all dimensions.
    
    p_threshold : float, default=0.05
        Threshold for Poisson cumulative probability below which points are marked as noise.

    min_neighbors : int, default=2
        Minimum required neighbors. Points with fewer neighbors are automatically considered noise.

    expected_lambda : float or None, default=None
        Externally provided expected lambda (mean neighbor count).
        If None, lambda is estimated from the data.

    Returns
    -------
    mask : np.ndarray (bool)
        Boolean mask, True for valid points, False for noise.
    """

    N, D = data.shape

    # Convert percent_radius to list for each dimension if it's a float
    if isinstance(percent_radius, float):
        percent_radius = [percent_radius] * D
    assert len(percent_radius) == D, "percent_radius length must match data dimensionality."

    # Calculate radius per dimension based on data range and percentage
    data_range = np.ptp(data, axis=0)
    radii = data_range * np.array(percent_radius)

    # Scale data points by radii to transform ellipsoid neighborhoods to spheres
    data_scaled = data / radii

    # KDTree for fast neighbor search in scaled space (radius=1 sphere)
    tree = KDTree(data_scaled)

    # Count neighbors within radius=1 (excluding the point itself)
    neighbor_counts = tree.query_ball_point(data_scaled, r=1.0, return_length=True) - 1

    # Estimate expected lambda if not provided externally
    if expected_lambda is None:
        # Volume of scaled data space (bounding box)
        volume_space = np.prod(np.ptp(data_scaled, axis=0))
        density = N / volume_space

        # Volume of a unit sphere in D dimensions
        unit_sphere_volume = (np.pi ** (D / 2)) / math.gamma(D / 2 + 1)

        # Expected number of neighbors in the unit sphere
        expected_lambda = density * unit_sphere_volume

    # Compute Poisson cumulative probability (P-value) for observed neighbor counts
    p_values = poisson.cdf(neighbor_counts, expected_lambda)

    # Points with fewer neighbors than min_neighbors or significantly low density are noise
    mask = (p_values >= p_threshold) & (neighbor_counts >= min_neighbors)

    return mask
