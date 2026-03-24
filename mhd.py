"""
Modified Hausdorff Distance (MHD) and Hausdorff Distance.

MHD is less sensitive to outliers than the standard Hausdorff distance
by using mean nearest-neighbor distances instead of the maximum.

References:
  Dubuisson, M.-P., Jain, A.K. (1994). A modified Hausdorff distance
  for object matching. ICPR.
"""

import numpy as np
from typing import Union

try:
    from scipy.spatial.distance import cdist
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise distances between rows of A and B. Returns shape (len(A), len(B))."""
    if _HAS_SCIPY:
        return cdist(A, B, metric="euclidean")
    # Fallback: manual Euclidean distances
    return np.sqrt(((A[:, np.newaxis, :] - B[np.newaxis, :, :]) ** 2).sum(axis=2))


def hausdorff_distance(
    A: np.ndarray,
    B: np.ndarray,
) -> float:
    """
    Symmetric Hausdorff distance between two point sets.

    H(A,B) = max( h(A,B), h(B,A) )
    where h(A,B) = max_{a in A} min_{b in B} d(a,b).

    Parameters
    ----------
    A : array-like, shape (n_A, d)
        First set of points (e.g. boundary or segmentation).
    B : array-like, shape (n_B, d)
        Second set of points.

    Returns
    -------
    float
        Hausdorff distance.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.size == 0 or B.size == 0:
        return np.nan
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    D = _pairwise_distances(A, B)
    h_AB = np.max(np.min(D, axis=1))
    h_BA = np.max(np.min(D, axis=0))
    return float(max(h_AB, h_BA))


def modified_hausdorff_distance(
    A: np.ndarray,
    B: np.ndarray,
    symmetric: bool = True,
) -> float:
    """
    Modified Hausdorff Distance (MHD).

    Uses mean of nearest-neighbor distances instead of max, so it is
    more robust to a few outlier points.

    MHD(A,B) = max( mean_a min_b d(a,b), mean_b min_a d(a,b) )
    when symmetric=True (default).

    Parameters
    ----------
    A : array-like, shape (n_A, d)
        First set of points.
    B : array-like, shape (n_B, d)
        Second set of points.
    symmetric : bool, default True
        If True, return max(d_mean(A→B), d_mean(B→A)).
        If False, return only d_mean(A→B).

    Returns
    -------
    float
        Modified Hausdorff distance.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.size == 0 or B.size == 0:
        return np.nan
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    D = _pairwise_distances(A, B)
    # For each point in A, min distance to B
    d_A_to_B = np.min(D, axis=1)
    mean_A_to_B = np.mean(d_A_to_B)
    if not symmetric:
        return float(mean_A_to_B)
    # For each point in B, min distance to A
    d_B_to_A = np.min(D, axis=0)
    mean_B_to_A = np.mean(d_B_to_A)
    return float(max(mean_A_to_B, mean_B_to_A))


def mhd(
    A: Union[np.ndarray, list],
    B: Union[np.ndarray, list],
    symmetric: bool = True,
) -> float:
    """
    Alias for modified_hausdorff_distance(A, B, symmetric=symmetric).
    """
    return modified_hausdorff_distance(A, B, symmetric=symmetric)


