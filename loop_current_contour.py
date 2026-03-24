"""
Compute the 17 cm contour of the Loop Current (largest contour at that level) in the Gulf of Mexico.
Uses sea surface height from AVISO NetCDF, HYCOM, or any (lon, lat, ssh) grid.
"""

from __future__ import annotations

import os
import glob
import re
import numpy as np
from typing import Tuple, Optional, List

try:
    from io_py import read_hycom_grid, sub_var2
    _HAS_IO = True
except ImportError:
    _HAS_IO = False

try:
    import netCDF4 as nc
    _HAS_NETCDF = True
except ImportError:
    nc = None
    _HAS_NETCDF = False

try:
    from scipy.interpolate import RegularGridInterpolator
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Domain: GOM -99 to -81 lon, 18 to 30N. LCE contours only east of 94W.
GOM_BBOX = (-99.0, 18.0, -81.0, 30.0)
GOM_BBOX_FOR_CONTOURS = (-99.0, 18.0, -81.0, 30.0)
DEMEAN_BBOX = (-99.0, 18.0, -81.0, 30.0)
LCE_EAST_OF_LON = -94.0  # 94W: LCE contours computed only east of this (LC has no restriction)

AVISO_DATE_START = "20250415"
AVISO_DATE_END = "20251231"
_AVISO_DATE_RE = re.compile(r"(\d{8})")
LAND_FILL_THRESHOLD = 1e9

YUCATAN_CHANNEL_BOX = (-90.0, 21.0, -86.0, 23.5)
FLORIDA_STRAITS_BOX = (-82.0, 24.0, -80.0, 26.0)
DEFAULT_SSH_LEVEL_M = 0.17


def _in_box(lon: float, lat: float, box: Tuple[float, float, float, float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = box
    return (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max)


def demean_region(ssh: np.ndarray, lon: np.ndarray, lat: np.ndarray, bbox: Tuple[float, float, float, float] = DEMEAN_BBOX) -> np.ndarray:
    """Subtract spatial mean over bbox (valid ocean points)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    mask = np.isfinite(ssh) & (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    if not np.any(mask):
        return ssh
    mean_val = np.nanmean(ssh[mask])
    return np.where(np.isfinite(ssh), ssh - mean_val, np.nan)


def demean_region_hycom(ssh: np.ndarray, lon: np.ndarray, lat: np.ndarray, grid_path: Optional[str], bbox: Tuple[float, float, float, float] = DEMEAN_BBOX) -> np.ndarray:
    """Area-weighted demean for HYCOM. Requires grid file with pscx, pscy; raises if missing or read fails."""
    if not _HAS_IO:
        raise RuntimeError("HYCOM area-weighted demean requires io_py (read_hycom_grid).")
    if not grid_path or not os.path.isfile(grid_path):
        raise FileNotFoundError(f"HYCOM grid file required for area-weighted demean: {grid_path!r} missing or not a file.")
    grid_data = read_hycom_grid(grid_path, ["pscx", "pscy"])
    pscx_arr = np.asarray(grid_data["pscx"])
    pscy_arr = np.asarray(grid_data["pscy"])
    area = pscx_arr * pscy_arr
    lon_min, lat_min, lon_max, lat_max = bbox
    mask = np.isfinite(ssh) & (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    if not np.any(mask):
        return ssh
    total_area = np.nansum(area[mask])
    if total_area <= 0:
        raise ValueError("Area-weighted demean: no valid area in bbox (total_area <= 0).")
    mean_val = np.nansum(ssh[mask] * area[mask]) / total_area
    return np.where(np.isfinite(ssh), ssh - mean_val, np.nan)


def demean_region_netcdf_mercator(ssh: np.ndarray, lon: np.ndarray, lat: np.ndarray, grid_nc_path: Optional[str], bbox: Tuple[float, float, float, float] = DEMEAN_BBOX, dx_var: str = "pscx", dy_var: str = "pscy") -> np.ndarray:
    """Area-weighted demean using grid NetCDF; fallback to unweighted."""
    if not grid_nc_path or not _HAS_NETCDF or not os.path.isfile(grid_nc_path):
        return demean_region(ssh, lon, lat, bbox)
    try:
        with nc.Dataset(grid_nc_path, "r") as ds:
            if dx_var not in ds.variables or dy_var not in ds.variables:
                return demean_region(ssh, lon, lat, bbox)
            pscx = np.asarray(ds.variables[dx_var][:])
            pscy = np.asarray(ds.variables[dy_var][:])
        area = pscx * pscy
        lon_min, lat_min, lon_max, lat_max = bbox
        mask = np.isfinite(ssh) & (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        if not np.any(mask):
            return ssh
        total_area = np.nansum(area[mask])
        if total_area <= 0:
            return demean_region(ssh, lon, lat, bbox)
        mean_val = np.nansum(ssh[mask] * area[mask]) / total_area
        return np.where(np.isfinite(ssh), ssh - mean_val, np.nan)
    except Exception:
        return demean_region(ssh, lon, lat, bbox)


def largest_contour_17cm(lon: np.ndarray, lat: np.ndarray, ssh: np.ndarray, level_m: float = 0.17) -> Optional[np.ndarray]:
    """Find largest 17 cm (or level_m) contour by length. Returns (N,2) lon,lat or None."""
    import matplotlib.pyplot as plt
    ssh_work = np.where(np.isfinite(ssh) & (np.abs(ssh) < LAND_FILL_THRESHOLD), ssh, np.nan)
    if np.all(np.isnan(ssh_work)):
        return None
    cs = plt.contour(lon, lat, ssh_work, levels=[level_m])
    plt.close("all")
    segs = getattr(cs, "allsegs", [[]])
    if not segs:
        return None
    segs = segs[0] if isinstance(segs[0], list) else [segs[0]]
    best = None
    best_len = 0
    for seg in segs:
        if seg is None or len(seg) < 2:
            continue
        seg = np.asarray(seg)
        if len(seg) > best_len:
            best = seg
            best_len = len(seg)
    return best


def all_contours_17cm(lon: np.ndarray, lat: np.ndarray, ssh: np.ndarray, level_m: float = 0.17) -> List[np.ndarray]:
    """Return all 17 cm (or level_m) contour segments. Each element is (N,2) lon,lat."""
    import matplotlib.pyplot as plt
    ssh_work = np.where(np.isfinite(ssh) & (np.abs(ssh) < LAND_FILL_THRESHOLD), ssh, np.nan)
    if np.all(np.isnan(ssh_work)):
        return []
    cs = plt.contour(lon, lat, ssh_work, levels=[level_m])
    plt.close("all")
    segs = getattr(cs, "allsegs", [[]])
    if not segs:
        return []
    segs = segs[0] if isinstance(segs[0], list) else [segs[0]]
    out = []
    for seg in segs:
        if seg is None or len(seg) < 2:
            continue
        out.append(np.asarray(seg))
    return out


def filter_contour_from_latitude(contour: Optional[np.ndarray], start_lat: float = 21.0) -> Optional[np.ndarray]:
    """Return contour with points at or north of start_lat, or None."""
    if contour is None or len(contour) < 2:
        return None
    c = np.asarray(contour)
    mask = c[:, 1] >= start_lat
    if not np.any(mask):
        return None
    return c[mask]


def clip_contours_to_longitude_cutoff(c1: Optional[np.ndarray], c2: Optional[np.ndarray], lon_cutoff: float = -81.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Clip contours to lon <= lon_cutoff."""
    def clip(c):
        if c is None or len(c) < 2:
            return None
        c = np.asarray(c)
        m = c[:, 0] <= lon_cutoff
        if not np.any(m):
            return None
        return c[m]
    return clip(c1), clip(c2)


def clip_contours_to_overlap(c1: np.ndarray, c2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clip both contours to the longitude overlap range."""
    if len(c1) < 2 or len(c2) < 2:
        return c1, c2
    lon_min = max(np.nanmin(c1[:, 0]), np.nanmin(c2[:, 0]))
    lon_max = min(np.nanmax(c1[:, 0]), np.nanmax(c2[:, 0]))
    if lon_min >= lon_max:
        return c1, c2
    m1 = (c1[:, 0] >= lon_min) & (c1[:, 0] <= lon_max)
    m2 = (c2[:, 0] >= lon_min) & (c2[:, 0] <= lon_max)
    return c1[m1], c2[m2]


def mhd(c1: np.ndarray, c2: np.ndarray, symmetric: bool = True) -> float:
    """Modified Hausdorff distance (mean of directed distances). Returns distance in degrees."""
    def directed_hausdorff(a, b):
        from scipy.spatial.distance import cdist
        d = cdist(a, b, metric="euclidean")
        return np.mean(np.min(d, axis=1))
    d12 = directed_hausdorff(c1, c2)
    d21 = directed_hausdorff(c2, c1)
    if symmetric:
        return max(d12, d21)
    return d12


def interpolate_ssh_to_grid(lon_src: np.ndarray, lat_src: np.ndarray, ssh_src: np.ndarray, lon_tgt: np.ndarray, lat_tgt: np.ndarray) -> np.ndarray:
    """Interpolate SSH from source to target grid."""
    if not _HAS_SCIPY:
        raise ImportError("scipy required for interpolate_ssh_to_grid")
    from scipy.interpolate import RegularGridInterpolator
    lon_1d = np.unique(lon_src) if lon_src.ndim >= 2 else np.asarray(lon_src).ravel()
    lat_1d = np.unique(lat_src) if lat_src.ndim >= 2 else np.asarray(lat_src).ravel()
    if ssh_src.ndim == 2:
        interp = RegularGridInterpolator((lat_1d, lon_1d), ssh_src, method="linear", bounds_error=False, fill_value=np.nan)
    else:
        interp = RegularGridInterpolator((lat_1d, lon_1d), ssh_src.squeeze(), method="linear", bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([lat_tgt.ravel(), lon_tgt.ravel()])
    out = interp(pts).reshape(lon_tgt.shape)
    return np.where(np.isfinite(out), out, np.nan)


def load_ssh_and_grid_hycom(archv_file: str, grid_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load lon, lat, ssh from HYCOM .a files. SSH in dm."""
    try:
        from hycom_io import load_hycom_ssh_and_grid
    except ImportError as e:
        raise ImportError("HYCOM I/O (hycom_io.py) not found. Ensure hycom_io.py and io_py are available.") from e
    return load_hycom_ssh_and_grid(archv_file, grid_file)


def load_ssh_from_netcdf(nc_path: str, lon_var: str = "longitude", lat_var: str = "latitude", ssh_var: str = "ssh", time_index: int = 0, bbox: Optional[Tuple[float, float, float, float]] = None, ssh_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load lon, lat, ssh from NetCDF. Returns 2D arrays; ssh in meters."""
    if not _HAS_NETCDF:
        raise ImportError("netCDF4 required for load_ssh_from_netcdf")
    with nc.Dataset(nc_path, "r") as ds:
        lon = np.asarray(ds.variables[lon_var][:])
        lat = np.asarray(ds.variables[lat_var][:])
        ssh = np.asarray(ds.variables[ssh_var][:])
    if ssh.ndim == 3:
        ssh = ssh[time_index, :, :]
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    ssh = np.asarray(ssh, dtype=float) * ssh_scale
    ssh = np.where(np.abs(ssh) < LAND_FILL_THRESHOLD, ssh, np.nan)
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        # If bbox uses -180..180 and data are 0..360, convert so the mask matches
        if lon_min < 0 or lon_max < 0:
            if np.nanmin(lon) >= 0 and np.nanmax(lon) > 180:
                lon = np.where(lon > 180, lon - 360.0, lon)
        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        lon = np.where(mask, lon, np.nan)
        lat = np.where(mask, lat, np.nan)
        ssh = np.where(mask, ssh, np.nan)
    return lon, lat, ssh


def _lon_to_180(lon: np.ndarray) -> np.ndarray:
    """Convert longitude 0..360 to -180..180 in place."""
    lon = np.asarray(lon, dtype=float)
    if np.nanmax(lon) > 180:
        lon = np.where(lon > 180, lon - 360.0, lon)
    return lon


def load_aviso_sla_plus_mdt_on_aviso_grid(aviso_nc_path: str, mdt_path: str, time_index: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load AVISO SLA and MDT; add them with no interpolation. SLA and MDT must be on the same grid
    (same shape). Full grids, 1e9->nan, lon -180..180. Returns lon, lat, ssh (m) on that grid.
    """
    if not _HAS_NETCDF:
        raise ImportError("netCDF4 required")
    with nc.Dataset(aviso_nc_path, "r") as ds:
        vlon = "longitude" if "longitude" in ds.variables else "lon"
        vlat = "latitude" if "latitude" in ds.variables else "lat"
        lon_av = np.asarray(ds.variables[vlon][:])
        lat_av = np.asarray(ds.variables[vlat][:])
        sla = np.asarray(ds.variables["sla"][:])
    if sla.ndim == 3:
        sla = sla[time_index, :, :]
    if lon_av.ndim == 1:
        lon_av, lat_av = np.meshgrid(lon_av, lat_av)
    lon_av = _lon_to_180(lon_av)
    sla = np.where(np.isfinite(sla) & (np.abs(sla) < LAND_FILL_THRESHOLD), sla, np.nan)

    with nc.Dataset(mdt_path, "r") as ds:
        vlon = "longitude" if "longitude" in ds.variables else "lon"
        vlat = "latitude" if "latitude" in ds.variables else "lat"
        lon_m = np.asarray(ds.variables[vlon][:])
        lat_m = np.asarray(ds.variables[vlat][:])
        mdt = np.asarray(ds.variables["mdt"][:])
    if mdt.ndim == 3:
        mdt = mdt[0, :, :]
    if lon_m.ndim == 1:
        lon_m, lat_m = np.meshgrid(lon_m, lat_m)
    lon_m = _lon_to_180(lon_m)
    mdt = np.where(np.isfinite(mdt) & (np.abs(mdt) < LAND_FILL_THRESHOLD), mdt, np.nan)

    if sla.shape != mdt.shape:
        raise ValueError(
            f"AVISO SLA and MDT must be on the same grid (no interpolation). "
            f"Got SLA shape {sla.shape}, MDT shape {mdt.shape}. Use an MDT file on the AVISO grid."
        )
    ssh = sla + mdt
    ssh = np.where(np.isfinite(ssh) & (np.abs(ssh) < LAND_FILL_THRESHOLD), ssh, np.nan)
    return lon_av, lat_av, ssh


def load_ssh_aviso_plus_mdt(aviso_nc_path: str, mdt_path: str, time_index: int = 0, bbox: Optional[Tuple[float, float, float, float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load AVISO SLA + MDT on AVISO grid (same as plot). bbox ignored; use load_aviso_sla_plus_mdt_on_aviso_grid for full control."""
    return load_aviso_sla_plus_mdt_on_aviso_grid(aviso_nc_path, mdt_path, time_index=time_index)


def get_aviso_contour_from_ssh(lon: np.ndarray, lat: np.ndarray, ssh: np.ndarray, level_cm: float = 17.0) -> Optional[np.ndarray]:
    """Compute 17 cm LC contour from already-loaded AVISO (lon, lat, ssh). Demean over GOM (-99 to -81, 18-30N). No file I/O."""
    ssh = demean_region(ssh, lon, lat, DEMEAN_BBOX)
    return largest_contour_17cm(lon, lat, ssh, level_m=level_cm / 100.0)


def get_aviso_contours_only(date: str, aviso_dir: str, mdt_path: str, level_cm: float = 17.0, bbox: Tuple[float, float, float, float] = GOM_BBOX_FOR_CONTOURS) -> Optional[np.ndarray]:
    """Load AVISO SLA + MDT, restrict to Gulf of Mexico bbox, detect 17 cm LC contour ONLY in Gulf."""
    pattern = os.path.join(aviso_dir, f"*{date}*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    lon, lat, ssh = load_aviso_sla_plus_mdt_on_aviso_grid(files[0], mdt_path)
    # Restrict to Gulf of Mexico bbox in lon/lat before finding contour
    lon_min, lat_min, lon_max, lat_max = bbox
    mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    # Set ssh outside bbox to NaN so contours are only detected in Gulf
    ssh_gom = np.where(mask, ssh, np.nan)
    return get_aviso_contour_from_ssh(lon, lat, ssh_gom, level_cm=level_cm)


def get_hycom_aviso_contours(hycom_archv_file: str, hycom_grid_file: str, date: str, aviso_dir: str, mdt_path: str, level_cm: float = 17.0, bbox: Tuple[float, float, float, float] = GOM_BBOX_FOR_CONTOURS, aviso_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Get HYCOM and AVISO contours for date. GOM -99 to -81 demean; LC contour has no lon restriction. Returns (contour_hycom, contour_aviso).
    If aviso_data is (lon_av, lat_av, ssh_av), use it to compute contour_aviso (no AVISO file load)."""
    lon_hycom, lat_hycom, ssh_dm = load_ssh_and_grid_hycom(hycom_archv_file, hycom_grid_file)
    ssh_hycom = ssh_dm / 10.0  # dm -> m
    ssh_hycom = demean_region_hycom(ssh_hycom, lon_hycom, lat_hycom, hycom_grid_file, DEMEAN_BBOX)
    contour_hycom = largest_contour_17cm(lon_hycom, lat_hycom, ssh_hycom, level_m=level_cm / 100.0)
    if not (AVISO_DATE_START <= date <= AVISO_DATE_END):
        return contour_hycom, None
    if aviso_data is not None:
        lon_av, lat_av, ssh_av = aviso_data
        # Restrict AVISO SSH to Gulf bbox before detecting LC so contour is ONLY in Gulf
        lon_min, lat_min, lon_max, lat_max = bbox
        mask = (lon_av >= lon_min) & (lon_av <= lon_max) & (lat_av >= lat_min) & (lat_av <= lat_max)
        ssh_gom = np.where(mask, ssh_av, np.nan)
        contour_aviso = get_aviso_contour_from_ssh(lon_av, lat_av, ssh_gom, level_cm=level_cm)
        return contour_hycom, contour_aviso
    pattern = os.path.join(aviso_dir, f"*{date}*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        return contour_hycom, None
    contour_aviso = get_aviso_contours_only(date, aviso_dir, mdt_path, level_cm=level_cm, bbox=bbox)
    return contour_hycom, contour_aviso


def get_model_contour_from_ssh(lon: np.ndarray, lat: np.ndarray, ssh: np.ndarray, level_m: float = 0.17, demean: bool = True) -> Optional[np.ndarray]:
    """Compute 17 cm LC contour from model (lon, lat, ssh). Optionally demean first."""
    if demean:
        ssh = demean_region(ssh, lon, lat, DEMEAN_BBOX)
    return largest_contour_17cm(lon, lat, ssh, level_m=level_m)
