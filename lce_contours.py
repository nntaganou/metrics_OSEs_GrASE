"""
LCE (Loop Current Eddy) contour detection: find contours in the LCE region and compute mean.

Used by MHD timeseries (all forecasts), animations, and any script that needs mean LCE
contours (16-18 cm in the LCE box). Works on any (lon, lat, ssh) grid; not tied to a specific date.
"""

import numpy as np
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_lce_region_contours(
    lon: np.ndarray,
    lat: np.ndarray,
    ssh: np.ndarray,
    lc_contour: Optional[np.ndarray] = None,
    level_min: float = 0.16,  # 16 cm
    level_max: float = 0.18,  # 18 cm
    num_levels: int = 21,  # Search at 21 levels (0.1 cm increments)
    min_lat: float = 23.0,
    max_lat: float = 28.5,
    min_lon: float = -90.0,  # Minimum longitude (west boundary)
    max_lon: float = -83.5,   # Maximum longitude (east boundary)
    min_lat_span: float = 1.0,  # Contour lat extent must be > this (degrees) to count as LCE
) -> List[np.ndarray]:
    """
    Find all contours in the LCE region (16-18 cm range) where 100% of points are in the region
    and the contour spans more than min_lat_span degrees in latitude (filters out small eddies).

    Works on whatever (lon, lat, ssh) you pass; typically ssh is demeaned by the caller.
    lc_contour is unused (kept for API compatibility).

    Returns
    -------
    List of contour arrays (N, 2) lon/lat that lie entirely in [min_lat, max_lat], [min_lon, max_lon]
    and have (max_lat - min_lat) > min_lat_span.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    ssh = np.asarray(ssh, dtype=float)
    if ssh.ndim != 2:
        raise ValueError("ssh must be 2D (ny, nx)")
    ny, nx = ssh.shape
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    if lon.shape != (ny, nx) or lat.shape != (ny, nx):
        raise ValueError("lon, lat must be (ny, nx) or 1D and consistent with ssh shape")

    ssh_work = np.where(np.isfinite(ssh), ssh, np.nan)
    if np.all(np.isnan(ssh_work)):
        return []

    levels_to_search = np.linspace(level_min, level_max, num_levels)
    lce_region_contours = []

    for search_level in levels_to_search:
        cs = plt.contour(lon, lat, ssh_work, levels=[search_level])
        plt.close("all")

        segs = getattr(cs, "allsegs", [])
        if not segs:
            continue
        segs = segs[0]
        if isinstance(segs, np.ndarray):
            segs = [segs] if segs.ndim == 2 and segs.shape[0] >= 2 else []

        if not segs:
            continue

        for seg in segs:
            if len(seg) < 3:
                continue
            lons = seg[:, 0]
            lats = seg[:, 1]
            if np.any(lons < min_lon) or np.any(lons > max_lon) or np.any(lats < min_lat) or np.any(lats > max_lat):
                continue
            lat_span = np.nanmax(lats) - np.nanmin(lats)
            if lat_span < min_lat_span:
                continue
            lce_region_contours.append(seg)

    return lce_region_contours


def _contour_area_deg2(contour: np.ndarray) -> float:
    """Shoelace formula for polygon area (lon/lat in degrees). Used to pick largest LCE."""
    if contour is None or len(contour) < 3:
        return 0.0
    x = contour[:, 0]
    y = contour[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, 1) - np.roll(x, 1) * y))


def largest_lce_contour(contour_list: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Return the single contour with largest area (by shoelace in lon/lat).
    Use this for model LCE when multiple contours would make the mean look wrong.
    Returns None if contour_list is empty.
    """
    if not contour_list:
        return None
    best = None
    best_area = -1.0
    for c in contour_list:
        if c is None or len(c) < 3:
            continue
        a = _contour_area_deg2(c)
        if a > best_area:
            best_area = a
            best = c
    return best


def compute_mean_contour(contour_list: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Compute a single mean contour from a list of contours (e.g. LCE segments).
    Resamples each to the same number of points, then averages (lon, lat).
    Returns (N, 2) array or None if contour_list is empty.
    """
    if not contour_list:
        return None
    n_points = 200  # target points per contour
    resampled = []
    for c in contour_list:
        if c is None or len(c) < 2:
            continue
        n = len(c)
        idx = np.linspace(0, n - 1, n_points).astype(int)
        idx = np.clip(idx, 0, n - 1)
        resampled.append(c[idx])
    if not resampled:
        return None
    stacked = np.stack(resampled, axis=0)
    mean_c = np.nanmean(stacked, axis=0)
    return mean_c
