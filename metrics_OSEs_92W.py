"""
OSEs metrics: animation, MHD time series, mean±std vs lead time, and first-LCE timing distribution.

  python metrics_OSEs.py --hycom [--animate] [--timeseries] [--mean-std] [--timing-distribution]
      Use HYCOM data (paths in script). Toggle outputs with bool flags.

  python metrics_OSEs.py --no-hycom --netcdf-dir DIR --aviso-dir DIR --mdt PATH [--forecast-start YYYY-MM-DD] ...
      Use NetCDF SSH files. --forecast-start required for --mean-std and --timing-distribution.

  Outputs (when enabled): animation (-o), MHD time series plot, mean±std vs lead time, timing histogram.
"""
import os
import sys
# Ensure loop_current_contour is taken from this directory (metrics_for_group_test), and
# io_py/info from pycodes. Script dir is already sys.path[0]; add pycodes so io_py can be found.
_script_dir = os.path.dirname(os.path.abspath(__file__))
#_pycodes_dir = os.path.dirname(_script_dir)
_pycodes_dir = "/gpfs/home/nntaganou/pycodes/"
if _pycodes_dir not in sys.path:
    sys.path.append(_pycodes_dir)

from loop_current_contour import (
    get_hycom_aviso_contours,
    GOM_BBOX,
    GOM_BBOX_FOR_CONTOURS,
    DEMEAN_BBOX,
    LCE_EAST_OF_LON,
    LAND_FILL_THRESHOLD,
    AVISO_DATE_START,
    AVISO_DATE_END,
    load_ssh_and_grid_hycom,
    demean_region,
    demean_region_hycom,
    demean_region_netcdf_mercator,
    largest_contour_17cm,
    all_contours_17cm,
    has_lce_17cm,
    get_aviso_contours_only,
    get_model_contour_from_ssh,
    load_aviso_sla_plus_mdt_on_aviso_grid,
)
from mhd import mhd
from lce_contours import find_lce_region_contours
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, Optional, List, Dict
import argparse
import glob
import re
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from collections import defaultdict

try:
    import netCDF4 as nc
except ImportError:
    nc = None

OUTPUT_DIR = os.environ.get("MHD_OUTPUT_DIR", ".")

# -----------------------------------------------------------------------------
# HYCOM / AVISO paths (edit these for your site)
# -----------------------------------------------------------------------------
# Base directory containing forecast subdirs named like *2025_REF and *2025_GLIDERS.
HYCOM_BASE_FORECAST_DIR = "/gpfs/research/coaps/nntaganou/HYCOM2.3-TSIS/GOMb0.04/FORECASTS"
# AVISO gridded SLA+MDT NetCDF directory (files matched by date, e.g. *YYYYMMDD*.nc).
AVISO_DIR = "/gpfs/research/coaps/abozec/HYCOM2.3-TSIS/AVISO/GRIDDED"
# Path to MDT NetCDF (e.g. CNES CLS global MDT).
MDT_PATH = "/gpfs/research/coaps/abozec/HYCOM2.3-TSIS/AVISO/clim/mdt_cnes_cls22_global.nc"
# For single-forecast animation (--animate without --animate-all): one REF and one GLIDERS run.
# Subdirs under HYCOM_BASE_FORECAST_DIR, e.g. "Jul15-2025_REF" and "Jul15-2025_GLIDERS".
HYCOM_SINGLE_REF_SUBDIR = "Jul15-2025_REF"
HYCOM_SINGLE_GLIDERS_SUBDIR = "Jul15-2025_GLIDERS"
# HYCOM grid file (optional). If None, grid is taken as {forecast_data_dir}/grad/regional.grid.a
# for each run. Set to a full path if you use one shared grid file for all forecasts.
HYCOM_GRID_FILE = None  # e.g. "/path/to/regional.grid.a"
# -----------------------------------------------------------------------------
# Drop in LC northernmost latitude (degrees) for first-LCE detachment detection.
LC_NORTH_DROP_DEGREES = 2.0
MAX_LEAD_DAYS = 90
# MHD: only contour points east of this longitude. LCEs included only if all their points are east of it.
# Swap the next two lines to switch between 90W (standard) and 92W (test: extend domain west).
MHD_LON_MIN = -92.0   # 92W for test
# MHD_LON_MIN = -90.0   # 90W standard


def filter_contour_from_latitude(contour: np.ndarray, start_lat: float = 21.0) -> np.ndarray:
    """Filter contour to keep only points at or above the specified latitude."""
    if contour is None or len(contour) == 0:
        return contour
    
    lats = contour[:, 1]
    mask = lats >= start_lat
    
    if not np.any(mask):
        lat_diffs = np.abs(lats - start_lat)
        closest_idx = np.argmin(lat_diffs)
        return contour[closest_idx:closest_idx+1] if len(contour) > closest_idx else contour[closest_idx:]
    
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return contour[:0]
    
    first_idx = indices[0]
    last_idx = indices[-1]
    return contour[first_idx:last_idx+1]


def clip_contours_to_longitude_cutoff(
    contour_hycom: np.ndarray,
    contour_aviso: np.ndarray,
    lon_cutoff: float = -81.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clip both contours to points west of the longitude cutoff."""
    if contour_hycom is None or contour_aviso is None:
        return contour_hycom, contour_aviso
    
    hycom_mask = contour_hycom[:, 0] < lon_cutoff
    aviso_mask = contour_aviso[:, 0] < lon_cutoff
    
    contour_hycom_clipped = contour_hycom[hycom_mask]
    contour_aviso_clipped = contour_aviso[aviso_mask]
    
    return contour_hycom_clipped, contour_aviso_clipped


def clip_contours_to_lon_min(contour: np.ndarray, lon_min: float) -> np.ndarray:
    """Keep only contour points with longitude >= lon_min (e.g. east of 90W: lon_min=-90)."""
    if contour is None or len(contour) == 0:
        return contour
    mask = contour[:, 0] >= lon_min
    return contour[mask]


def clip_contours_to_overlap(
    contour_hycom: np.ndarray,
    contour_aviso: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clip both contours to their overlapping longitude range."""
    if contour_hycom is None or contour_aviso is None:
        return contour_hycom, contour_aviso
    
    hycom_lon_min = np.min(contour_hycom[:, 0])
    hycom_lon_max = np.max(contour_hycom[:, 0])
    aviso_lon_min = np.min(contour_aviso[:, 0])
    aviso_lon_max = np.max(contour_aviso[:, 0])
    
    overlap_lon_min = max(hycom_lon_min, aviso_lon_min)
    overlap_lon_max = min(hycom_lon_max, aviso_lon_max)
    
    hycom_mask = (contour_hycom[:, 0] >= overlap_lon_min) & (contour_hycom[:, 0] <= overlap_lon_max)
    aviso_mask = (contour_aviso[:, 0] >= overlap_lon_min) & (contour_aviso[:, 0] <= overlap_lon_max)
    
    contour_hycom_clipped = contour_hycom[hycom_mask]
    contour_aviso_clipped = contour_aviso[aviso_mask]
    
    return contour_hycom_clipped, contour_aviso_clipped


    


def extract_date_from_filename(filename: str) -> Optional[str]:
    """Extract date from HYCOM filename. Returns YYYYMMDD format."""
    match = re.search(r'(\d{4})_(\d{3})', filename)
    if match:
        year = match.group(1)
        day_of_year = int(match.group(2))
        date_obj = datetime(int(year), 1, 1) + timedelta(days=day_of_year - 1)
        return date_obj.strftime("%Y%m%d")
    return None


def extract_date_from_netcdf_path(filepath: str) -> Optional[str]:
    """Extract date from NetCDF filename and return YYYYMMDD.
    Accepts YYYYMMDD, YYYY-MM-DD, and YYYY_MM_DD patterns in the basename."""
    name = os.path.basename(filepath)
    # Prefer separated form first, then compact form.
    match = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', name)
    if not match:
        match = re.search(r'(\d{4})(\d{2})(\d{2})', name)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"
    return None


def infer_forecast_start_from_path(filepath: str) -> Optional[datetime]:
    """Infer forecast start from parent folder name (YYYYMMDD, YYYY-MM-DD, Apr15-2025, april15_2025, etc.)."""
    folder = os.path.basename(os.path.dirname(filepath))
    if not folder:
        return None
    m = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', folder)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = re.search(
        r'(?i)(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
        r'[-_ ]*(\d{1,2})[-_ ]*(\d{4})',
        folder,
    )
    if m:
        month_token = m.group(1).lower()[:3]
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        try:
            return datetime(int(m.group(3)), month_map[month_token], int(m.group(2)))
        except (ValueError, KeyError):
            pass
    return None


# -----------------------------------------------------------------------------
# Generic processing core: no HYCOM-specific I/O. Other groups can call this
# with (date, contour_model, contour_aviso, lon_model, lat_model, ssh_model)
# from their own NetCDF loader.
# -----------------------------------------------------------------------------
def process_model_file_for_animation_core(
    date: str,
    contour_model: np.ndarray,
    contour_aviso: np.ndarray,
    lon_model: np.ndarray,
    lat_model: np.ndarray,
    ssh_model: np.ndarray,
    aviso_dir: str,
    mdt_path: str,
    lon_cutoff: float = -81.0,
    use_cutoff: bool = True,
    model_ssh_already_demeaned: bool = False,
    aviso_ssh_tuple: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    lce_min_lon_span: float = 1.3,
) -> Dict:
    """
    Compute LCE, MHD, and return result dict from already-loaded contours and SSH.
    AVISO LCE is computed on the AVISO grid (no interpolation to model grid).
    If model_ssh_already_demeaned is True, model SSH is not demeaned again (used when caller
    already applied regular or area-weighted demean).
    If aviso_ssh_tuple is (lon_av, lat_av, ssh_av), use it for AVISO LCE (no AVISO file load).

    No file I/O (unless aviso_ssh_tuple is None and AVISO LCE is computed). For NetCDF groups: load your (lon, lat, ssh), demean (regular or Mercator),
    get contour_model via get_model_contour_from_ssh(..., demean=False), contour_aviso via
    get_aviso_contours_only(), then call this with (..., ssh_model=ssh_demeaned, model_ssh_already_demeaned=True).
    """
    from lce_contours import find_lce_region_contours, compute_mean_contour, largest_lce_contour
    result = {
        'date': date,
        'mhd': np.nan,
        'contour_hycom_full': None,
        'contour_aviso_full': None,
        'contour_lce_hycom': None,
        'contour_lce_aviso': None,
        'contours_lc_hycom_all': [],
        'contours_lc_aviso_all': [],
        'success': False
    }
    try:
        contour_hycom = filter_contour_from_latitude(contour_model, start_lat=21.0)
        contour_aviso_f = filter_contour_from_latitude(contour_aviso, start_lat=21.0)
        if use_cutoff:
            contour_hycom, contour_aviso_f = clip_contours_to_longitude_cutoff(
                contour_hycom, contour_aviso_f, lon_cutoff=lon_cutoff
            )
        contour_hycom_full = contour_hycom.copy() if contour_hycom is not None else None
        contour_aviso_full = contour_aviso_f.copy() if contour_aviso_f is not None else None

        contour_lce_hycom = None
        if model_ssh_already_demeaned:
            ssh_model_demeaned = ssh_model
        else:
            ssh_model_demeaned = demean_region(ssh_model, lon_model, lat_model, DEMEAN_BBOX)
        # All LC contours at 17 cm for plotting (filtered and clipped like single LC)
        def _filter_clip_lc_list(segs, lon_cutoff_val):
            out = []
            for seg in segs:
                c = filter_contour_from_latitude(seg, start_lat=21.0)
                if c is None or len(c) < 2:
                    continue
                if use_cutoff:
                    c = c[c[:, 0] <= lon_cutoff_val]
                if len(c) >= 2:
                    out.append(c)
            return out
        result['contours_lc_hycom_all'] = _filter_clip_lc_list(
            all_contours_17cm(lon_model, lat_model, ssh_model_demeaned, level_m=0.17), lon_cutoff
        )
        if not result['contours_lc_hycom_all'] and contour_hycom_full is not None and len(contour_hycom_full) >= 2:
            result['contours_lc_hycom_all'] = [contour_hycom_full]
        ssh_for_lce = np.where(lon_model >= LCE_EAST_OF_LON, ssh_model_demeaned, np.nan)
        lce_list = find_lce_region_contours(
            lon_model, lat_model, ssh_for_lce,
            lc_contour=contour_hycom_full,
            level_min=0.17, level_max=0.17, num_levels=1,
            min_lat=22.5, min_lon=-94.0, max_lon=-83.5,
            min_lon_span=lce_min_lon_span,
        )
        if len(lce_list) > 0:
            # Use largest LCE by area so REF/GLIDERS show one dominant eddy (avoids wrong shape from averaging multiple)
            contour_lce_hycom = largest_lce_contour(lce_list)

        contour_lce_aviso = None
        try:
            if aviso_ssh_tuple is not None:
                lon_av, lat_av, ssh_aviso = aviso_ssh_tuple
            else:
                pattern = os.path.join(aviso_dir, f"*{date}*.nc")
                files = sorted(glob.glob(pattern))
                if not files:
                    raise FileNotFoundError("no AVISO file")
                lon_av, lat_av, ssh_aviso = load_aviso_sla_plus_mdt_on_aviso_grid(files[0], mdt_path)
            ssh_aviso = demean_region(ssh_aviso, lon_av, lat_av, DEMEAN_BBOX)
            result['contours_lc_aviso_all'] = _filter_clip_lc_list(
                all_contours_17cm(lon_av, lat_av, ssh_aviso, level_m=0.17), lon_cutoff
            )
            if not result['contours_lc_aviso_all'] and contour_aviso_full is not None and len(contour_aviso_full) >= 2:
                result['contours_lc_aviso_all'] = [contour_aviso_full]
            ssh_aviso_for_lce = np.where(lon_av >= LCE_EAST_OF_LON, ssh_aviso, np.nan)
            lce_aviso_list = find_lce_region_contours(
                lon_av, lat_av, ssh_aviso_for_lce,
                lc_contour=contour_aviso_full,
                level_min=0.17, level_max=0.17, num_levels=1,
                min_lat=22.5, min_lon=-94.0, max_lon=-83.5,
                min_lon_span=lce_min_lon_span,
            )
            if len(lce_aviso_list) > 0:
                contour_lce_aviso = largest_lce_contour(lce_aviso_list)
        except Exception:
            pass

        # Enforce LCE box: only keep LCEs whose entire contour lies within the LCE domain.
        # Longitude box: MHD_LON_MIN <= lon <= -83.5 (92W or 90W western edge), Latitude box: 22.5 <= lat <= 28.5
        def _filter_lce_box(c: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if c is None or len(c) == 0:
                return None
            lons = c[:, 0]
            lats = c[:, 1]
            if (
                np.min(lons) >= MHD_LON_MIN
                and np.max(lons) <= -83.5
                and np.min(lats) >= 22.5
                and np.max(lats) <= 28.5
            ):
                return c
            return None

        contour_lce_hycom = _filter_lce_box(contour_lce_hycom)
        contour_lce_aviso = _filter_lce_box(contour_lce_aviso)

        # MHD: LC uses only min lat 21 and max lon -81 (already applied above).
        # For the MHD calculation itself, we further restrict to east of 90W.
        # LCE is included in MHD only if the ENTIRE LCE contour is east of 90W (no partial inclusion).
        contour_hycom_for_mhd = contour_hycom.copy() if contour_hycom is not None else None
        contour_aviso_for_mhd = contour_aviso_f.copy() if contour_aviso_f is not None else None
        # Include LCE only if ALL its points are east of 90W; if any point is west of 90W, exclude that LCE entirely (no clipping).
        if contour_lce_hycom is not None and len(contour_lce_hycom) > 0 and np.min(contour_lce_hycom[:, 0]) >= MHD_LON_MIN:
            contour_hycom_for_mhd = (
                np.vstack([contour_hycom_for_mhd, contour_lce_hycom])
                if contour_hycom_for_mhd is not None
                else contour_lce_hycom
            )
        if contour_lce_aviso is not None and len(contour_lce_aviso) > 0 and np.min(contour_lce_aviso[:, 0]) >= MHD_LON_MIN:
            contour_aviso_for_mhd = (
                np.vstack([contour_aviso_for_mhd, contour_lce_aviso])
                if contour_aviso_for_mhd is not None
                else contour_lce_aviso
            )
        # After adding any qualifying LCEs, clip both sets to east of 90W for the MHD metric
        if contour_hycom_for_mhd is not None:
            contour_hycom_for_mhd = clip_contours_to_lon_min(contour_hycom_for_mhd, MHD_LON_MIN)
        if contour_aviso_for_mhd is not None:
            contour_aviso_for_mhd = clip_contours_to_lon_min(contour_aviso_for_mhd, MHD_LON_MIN)

        # Optional debug dump for a specific date (commented out): save HYCOM and AVISO LC / LC+LCE coordinates
        # if date == "20250715":
        #     try:
        #         import os as _os
        #         out_dir = OUTPUT_DIR or "."
        #         _os.makedirs(out_dir, exist_ok=True)
        #         if contour_aviso is not None and len(contour_aviso) > 0:
        #             np.savetxt(_os.path.join(out_dir, "lc_raw_aviso_20250715.txt"), contour_aviso, fmt="%.6f", header="lon lat")
        #         if contour_aviso_f is not None and len(contour_aviso_f) > 0:
        #             np.savetxt(_os.path.join(out_dir, "lc_only_aviso_20250715.txt"), contour_aviso_f, fmt="%.6f", header="lon lat")
        #         if contour_hycom is not None and len(contour_hycom) > 0:
        #             np.savetxt(_os.path.join(out_dir, "lc_only_hycom_20250715.txt"), contour_hycom, fmt="%.6f", header="lon lat")
        #         if contour_hycom_for_mhd is not None and len(contour_hycom_for_mhd) > 0:
        #             np.savetxt(_os.path.join(out_dir, "lc_for_mhd_hycom_20250715.txt"), contour_hycom_for_mhd, fmt="%.6f", header="lon lat")
        #         if contour_aviso_for_mhd is not None and len(contour_aviso_for_mhd) > 0:
        #             np.savetxt(_os.path.join(out_dir, "lc_for_mhd_aviso_20250715.txt"), contour_aviso_for_mhd, fmt="%.6f", header="lon lat")
        #     except Exception:
        #         pass

        # Store contours whenever we have them (so test can plot REF/AVISO/GLIDERS per day even if one is missing)
        if contour_hycom_full is not None and len(contour_hycom_full) > 0:
            result['contour_hycom_full'] = contour_hycom_full
            result['contour_lce_hycom'] = contour_lce_hycom
        if contour_aviso_full is not None and len(contour_aviso_full) > 0:
            result['contour_aviso_full'] = contour_aviso_full
            result['contour_lce_aviso'] = contour_lce_aviso
        if result.get('contour_hycom_full') is not None or result.get('contour_aviso_full') is not None:
            result['success'] = True
        # LC-only MHD (no LCE) — always computed for --no-lce output
        result['mhd_lc_only'] = np.nan
        if contour_hycom is not None and contour_aviso_f is not None and len(contour_hycom) > 0 and len(contour_aviso_f) > 0:
            h_lc = clip_contours_to_lon_min(contour_hycom, MHD_LON_MIN)
            a_lc = clip_contours_to_lon_min(contour_aviso_f, MHD_LON_MIN)
            if len(h_lc) > 0 and len(a_lc) > 0:
                result['mhd_lc_only'] = mhd(h_lc, a_lc, symmetric=True)
        # MHD when we have both sides and can compute; else NaN (contours still stored)
        if contour_hycom_full is not None and len(contour_hycom_full) > 0 and contour_aviso_full is not None and len(contour_aviso_full) > 0:
            if len(contour_hycom_for_mhd) > 0 and len(contour_aviso_for_mhd) > 0:
                result['mhd'] = mhd(contour_hycom_for_mhd, contour_aviso_for_mhd, symmetric=True)
            else:
                result['mhd'] = result['mhd_lc_only']
        else:
            result['mhd'] = np.nan
    except Exception:
        result['success'] = False
    return result


# -----------------------------------------------------------------------------
# HYCOM-specific wrapper: gets date from HYCOM filename, loads via get_hycom_aviso_contours
# and load_ssh_and_grid_hycom, then calls the generic core. Replace with your own
# wrapper for NetCDF (e.g. process_netcdf_file_for_animation) that calls the core.
# -----------------------------------------------------------------------------
def process_hycom_file_for_animation(
    hycom_file: str,
    grid_file: str,
    aviso_dir: str,
    mdt_path: str,
    lon_cutoff: float = -81.0,
    use_cutoff: bool = True,
    lce_min_lon_span: float = 1.3,
) -> Dict:
    """
    Process a single HYCOM file: load contours and SSH via HYCOM I/O, then run generic core.

    Returns dict with: date, mhd, contour_hycom_full, contour_aviso_full,
                       contour_lce_hycom, contour_lce_aviso, success
    """
    result = {
        'date': None,
        'mhd': np.nan,
        'contour_hycom_full': None,
        'contour_aviso_full': None,
        'contour_lce_hycom': None,
        'contour_lce_aviso': None,
        'success': False
    }
    try:
        date = extract_date_from_filename(hycom_file)
        if date is None:
            return result
        # Load AVISO once for this date (LC contour + LCE in core)
        aviso_ssh_tuple = None
        if AVISO_DATE_START <= date <= AVISO_DATE_END:
            pattern = os.path.join(aviso_dir, f"*{date}*.nc")
            files = sorted(glob.glob(pattern))
            if files:
                aviso_ssh_tuple = load_aviso_sla_plus_mdt_on_aviso_grid(files[0], mdt_path)
        contour_hycom, contour_aviso = get_hycom_aviso_contours(
            hycom_archv_file=hycom_file,
            hycom_grid_file=grid_file,
            date=date,
            aviso_dir=aviso_dir,
            mdt_path=mdt_path,
            aviso_data=aviso_ssh_tuple,
        )
        if contour_hycom is None or contour_aviso is None:
            result['date'] = date
            return result
        lon_hycom, lat_hycom, ssh_hycom = load_ssh_and_grid_hycom(hycom_file, grid_file)
        ssh_hycom = ssh_hycom / 10.0
        ssh_hycom = demean_region_hycom(ssh_hycom, lon_hycom, lat_hycom, grid_file, DEMEAN_BBOX)
        return process_model_file_for_animation_core(
            date=date,
            contour_model=contour_hycom,
            contour_aviso=contour_aviso,
            lon_model=lon_hycom,
            lat_model=lat_hycom,
            ssh_model=ssh_hycom,
            aviso_dir=aviso_dir,
            mdt_path=mdt_path,
            lon_cutoff=lon_cutoff,
            use_cutoff=use_cutoff,
            model_ssh_already_demeaned=True,
            aviso_ssh_tuple=aviso_ssh_tuple,
            lce_min_lon_span=lce_min_lon_span,
        )
    except Exception as e:
        print(f"  Error processing {os.path.basename(hycom_file)}: {e}")
        result['success'] = False
    return result


# -----------------------------------------------------------------------------
# Generic NetCDF processor: load (lon, lat, ssh) from NetCDF, then run core.
# Other groups can use this as-is by passing their file paths and variable names.
# -----------------------------------------------------------------------------
def load_ssh_from_netcdf(
    nc_path: str,
    lon_var: str = "longitude",
    lat_var: str = "latitude",
    ssh_var: str = "ssh",
    time_idx: int = 0,
    ssh_scale: float = 1.0,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load lon, lat, and SSH from a NetCDF file as 2D arrays (meters).

    Handles 1D or 2D lon/lat; if SSH has a time dimension, time_idx is used.
    ssh_scale converts stored SSH to meters (e.g. 0.01 if stored in cm).

    Returns
    -------
    lon, lat, ssh : 2D arrays, same shape. SSH in meters.
    """
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 is required for load_ssh_from_netcdf")
    with nc.Dataset(nc_path, "r") as ds:
        lon = np.asarray(ds.variables[lon_var][:])
        lat = np.asarray(ds.variables[lat_var][:])
        ssh = np.asarray(ds.variables[ssh_var][:])
        fill = getattr(ds.variables[ssh_var], "fill_value", np.nan)
    # Flatten to 1D if masked
    if hasattr(lon, "filled"):
        lon = lon.filled(np.nan)
    if hasattr(lat, "filled"):
        lat = lat.filled(np.nan)
    if hasattr(ssh, "filled"):
        ssh = ssh.filled(np.nan)
    # Time slice
    if ssh.ndim == 3:
        ssh = ssh[time_idx, :, :]
    elif ssh.ndim != 2:
        raise ValueError(f"SSH must be 2D or 3D (time, lat, lon), got ndim={ssh.ndim}")
    # 1D lon/lat -> 2D
    if lon.ndim == 1 and lat.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon, lat)
    elif lon.ndim == 2 and lat.ndim == 2:
        lon_2d, lat_2d = lon, lat
    else:
        raise ValueError("lon/lat must be both 1D or both 2D")
    # Align shapes: SSH might be (lat, lon) or (lon, lat)
    if ssh.shape != lon_2d.shape:
        if ssh.shape == (lon_2d.shape[1], lon_2d.shape[0]):
            ssh = ssh.T
        else:
            raise ValueError(f"SSH shape {ssh.shape} does not match lon/lat shape {lon_2d.shape}")
    ssh_m = np.asarray(ssh, dtype=float) * ssh_scale
    # Mask fill value (same as AVISO): use variable's fill_value then land/fill threshold
    try:
        fill = float(fill)
    except (TypeError, ValueError):
        fill = np.nan
    if np.isnan(fill):
        ssh_m = np.where(np.isfinite(ssh_m), ssh_m, np.nan)
    else:
        ssh_m = np.where(ssh_m != fill, ssh_m, np.nan)
    ssh_m = np.where(np.abs(ssh_m) < LAND_FILL_THRESHOLD, ssh_m, np.nan)
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        mask = (lon_2d >= lon_min) & (lon_2d <= lon_max) & (lat_2d >= lat_min) & (lat_2d <= lat_max)
        # Keep full grid but mask outside bbox so contour code ignores
        ssh_m = np.where(mask, ssh_m, np.nan)
    return lon_2d, lat_2d, ssh_m


def process_netcdf_file_for_animation(
    nc_path: str,
    date: str,
    aviso_dir: str,
    mdt_path: str,
    lon_var: str = "longitude",
    lat_var: str = "latitude",
    ssh_var: str = "ssh",
    time_idx: int = 0,
    ssh_scale: float = 1.0,
    lon_cutoff: float = -81.0,
    use_cutoff: bool = True,
    bbox: Optional[Tuple[float, float, float, float]] = GOM_BBOX,
    grid_path_netcdf: Optional[str] = None,
    mercator: bool = False,
    grid_dx_var: str = "pscx",
    grid_dy_var: str = "pscy",
    lce_min_lon_span: float = 1.3,
) -> Dict:
    """
    Process a single NetCDF file: load (lon, lat, ssh), demean (regular or area-weighted
    if Mercator), get model and AVISO contours, then run the generic core.

    Parameters
    ----------
    nc_path : str
        Path to the NetCDF file.
    date : str
        Date as YYYYMMDD (for AVISO lookup and result).
    aviso_dir, mdt_path : str
        AVISO gridded directory and MDT NetCDF path.
    lon_var, lat_var, ssh_var : str
        NetCDF variable names for longitude, latitude, and sea surface height.
    time_idx : int
        Time index if SSH has a time dimension (default 0).
    ssh_scale : float
        Multiply SSH by this to get meters (e.g. 0.01 if stored in cm; 1.0 if already meters).
    lon_cutoff, use_cutoff, bbox
        Passed to core / contour logic.
    grid_path_netcdf : str, optional
        Path to NetCDF grid file with cell sizes (dx, dy). Required when mercator=True.
    mercator : bool
        If True and grid_path_netcdf is set, use area-weighted demean (for Mercator grids).
    grid_dx_var, grid_dy_var : str
        Variable names in grid NetCDF for cell sizes (default pscx, pscy).

    Returns
    -------
    dict with: date, mhd, contour_hycom_full, contour_aviso_full,
               contour_lce_hycom, contour_lce_aviso, success
    """
    result = {
        "date": date,
        "mhd": np.nan,
        "contour_hycom_full": None,
        "contour_aviso_full": None,
        "contour_lce_hycom": None,
        "contour_lce_aviso": None,
        "success": False,
    }
    try:
        lon_model, lat_model, ssh_model = load_ssh_from_netcdf(
            nc_path,
            lon_var=lon_var,
            lat_var=lat_var,
            ssh_var=ssh_var,
            time_idx=time_idx,
            ssh_scale=ssh_scale,
            bbox=bbox,
        )
        if mercator and grid_path_netcdf:
            ssh_demeaned = demean_region_netcdf_mercator(
                ssh_model, lon_model, lat_model,
                grid_path_netcdf, bbox=DEMEAN_BBOX,
                dx_var=grid_dx_var, dy_var=grid_dy_var,
            )
        else:
            ssh_demeaned = demean_region(ssh_model, lon_model, lat_model, DEMEAN_BBOX)
        contour_model = get_model_contour_from_ssh(lon_model, lat_model, ssh_demeaned, demean=False)
        contour_aviso = get_aviso_contours_only(date, aviso_dir=aviso_dir, mdt_path=mdt_path, bbox=bbox or GOM_BBOX_FOR_CONTOURS)
        if contour_model is None or contour_aviso is None:
            return result
        return process_model_file_for_animation_core(
            date=date,
            contour_model=contour_model,
            contour_aviso=contour_aviso,
            lon_model=lon_model,
            lat_model=lat_model,
            ssh_model=ssh_demeaned,
            aviso_dir=aviso_dir,
            mdt_path=mdt_path,
            lon_cutoff=lon_cutoff,
            use_cutoff=use_cutoff,
            model_ssh_already_demeaned=True,
            lce_min_lon_span=lce_min_lon_span,
        )
    except Exception as e:
        print(f"  Error processing {os.path.basename(nc_path)}: {e}")
        result["success"] = False
    return result


def create_animation_frame(
    ax1, ax2,
    frame_idx: int,
    all_results_ref: List[Dict],
    all_results_gliders: List[Dict],
    dates_ref: List[datetime],
    dates_gliders: List[datetime],
    xlim_timeseries: Tuple[datetime, datetime] = None,
    ylim_timeseries: Tuple[float, float] = None,
    ref_label: str = "HYCOM_REF",
    gliders_label: str = "HYCOM_GLIDERS",
):
    """Create a single animation frame."""
    # Clear axes
    ax1.clear()
    ax2.clear()
    
    # IMMEDIATELY set fixed limits and disable autoscaling BEFORE any plotting
    if xlim_timeseries is not None:
        ax2.set_xlim(xlim_timeseries[0], xlim_timeseries[1], auto=False, emit=False)
    if ylim_timeseries is not None:
        ax2.set_ylim(ylim_timeseries[0], ylim_timeseries[1], auto=False, emit=False)
    
    # Disable autoscaling completely
    ax2.set_autoscalex_on(False)
    ax2.set_autoscaley_on(False)
    ax2.autoscale(enable=False)
    
    # Get current date
    if frame_idx < len(dates_ref):
        current_date = dates_ref[frame_idx]
        current_date_str = current_date.strftime("%Y%m%d")
    else:
        return
    
    # Find current results
    result_ref = None
    result_gliders = None
    
    for r in all_results_ref:
        if r['date'] == current_date_str:
            result_ref = r
            break

    for r in all_results_gliders:
        if r['date'] == current_date_str:
            result_gliders = r
            break
    
    # Plot contours on top subplot (using cartopy); use .get() so days with only REF or only AVISO don't KeyError
    if result_ref is not None:
        contour_hycom_ref = result_ref.get('contour_hycom_full')
        contour_aviso = result_ref.get('contour_aviso_full')
        contour_lce_hycom = result_ref.get('contour_lce_hycom', None)
        contour_lce_aviso = result_ref.get('contour_lce_aviso', None)
        
        # Plot LC and LCE mean contours separately but with same style for each model
        hycom_ref_plotted = False
        if contour_hycom_ref is not None and len(contour_hycom_ref) > 0:
            ax1.plot(contour_hycom_ref[:, 0], contour_hycom_ref[:, 1],
                    color='steelblue', linewidth=2.5, label=ref_label, alpha=0.8,
                    transform=ccrs.PlateCarree())
            hycom_ref_plotted = True
        
        if contour_lce_hycom is not None and len(contour_lce_hycom) > 0:
            ax1.plot(contour_lce_hycom[:, 0], contour_lce_hycom[:, 1],
                    color='steelblue', linewidth=2.5, alpha=0.8,
                    transform=ccrs.PlateCarree())
            if not hycom_ref_plotted:
                ax1.plot([], [], color='steelblue', linewidth=2.5, label=ref_label, alpha=0.8)
        
        # AVISO
        aviso_plotted = False
        if contour_aviso is not None and len(contour_aviso) > 0:
            ax1.plot(contour_aviso[:, 0], contour_aviso[:, 1],
                    "k-", linewidth=2.5, label="AVISO", alpha=0.8,
                    transform=ccrs.PlateCarree())
            aviso_plotted = True
        
        if contour_lce_aviso is not None and len(contour_lce_aviso) > 0:
            ax1.plot(contour_lce_aviso[:, 0], contour_lce_aviso[:, 1],
                    "k-", linewidth=2.5, alpha=0.8,
                    transform=ccrs.PlateCarree())
            if not aviso_plotted:
                ax1.plot([], [], "k-", linewidth=2.5, label="AVISO", alpha=0.8)
    
    if result_gliders is not None:
        contour_hycom_gliders = result_gliders.get('contour_hycom_full')
        contour_lce_gliders = result_gliders.get('contour_lce_hycom', None)
        
        # Plot LC and LCE mean separately but with same style for GLIDERS
        gliders_plotted = False
        if contour_hycom_gliders is not None and len(contour_hycom_gliders) > 0:
            ax1.plot(contour_hycom_gliders[:, 0], contour_hycom_gliders[:, 1],
                    "r-", linewidth=2.5, label=gliders_label, alpha=0.8,
                    transform=ccrs.PlateCarree())
            gliders_plotted = True
        
        if contour_lce_gliders is not None and len(contour_lce_gliders) > 0:
            ax1.plot(contour_lce_gliders[:, 0], contour_lce_gliders[:, 1],
                    "r-", linewidth=2.5, alpha=0.8,
                    transform=ccrs.PlateCarree())
            if not gliders_plotted:
                ax1.plot([], [], "r-", linewidth=2.5, label=gliders_label, alpha=0.8)
    
    # Format date for title
    date_formatted = current_date.strftime("%m/%d/%Y")
    
    # Set map extent and add cartopy features (western edge follows MHD_LON_MIN: 92W or 90W)
    ax1.set_extent([MHD_LON_MIN, -80, 18, 30], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
    # Remove ocean color - use white/transparent background
    ax1.add_feature(cfeature.OCEAN, color='white', zorder=0)
    
    # Add gridlines
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    ax1.set_title(f"Loop Current & Mean LCE Contours - {date_formatted}", fontsize=12, fontweight='bold')
    ax1.legend(loc="best", fontsize=9)
    
    # Plot time series on bottom subplot (up to current date)
    dates_ref_plot = []
    mhd_ref_km = []
    dates_gliders_plot = []
    mhd_gliders_km = []
    
    for i, r in enumerate(all_results_ref):
        if dates_ref[i] <= current_date:
            m = r.get('mhd')
            if m is not None and np.isfinite(m):
                dates_ref_plot.append(dates_ref[i])
                mhd_ref_km.append(m * 111.0)

    for i, r in enumerate(all_results_gliders):
        if dates_gliders[i] <= current_date:
            m = r.get('mhd')
            if m is not None and np.isfinite(m):
                dates_gliders_plot.append(dates_gliders[i])
                mhd_gliders_km.append(m * 111.0)
    
    if len(dates_ref_plot) > 0:
        ax2.plot(dates_ref_plot, mhd_ref_km, 'o-', linewidth=2, markersize=5,
                color='steelblue', label=f'MHD: AVISO vs {ref_label}')
    
    if len(dates_gliders_plot) > 0:
        ax2.plot(dates_gliders_plot, mhd_gliders_km, 'o-', linewidth=2, markersize=5,
                color='red', label=f'MHD: AVISO vs {gliders_label}')
    
    # Add vertical line at current date
    ax2.axvline(x=current_date, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Modified Hausdorff Distance (km)', fontsize=11)
    ax2.set_title('MHD Time Series', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Force fixed axis limits - set multiple times to ensure they stick
    if xlim_timeseries is not None:
        ax2.set_xlim(xlim_timeseries[0], xlim_timeseries[1], auto=False, emit=False)
    if ylim_timeseries is not None:
        ax2.set_ylim(ylim_timeseries[0], ylim_timeseries[1], auto=False, emit=False)
    
    # Disable autoscaling completely
    ax2.set_autoscalex_on(False)
    ax2.set_autoscaley_on(False)
    ax2.autoscale(enable=False)
    
    # Don't use tight_layout - it can interfere with fixed limits
    # Instead, manually adjust subplot spacing if needed
    plt.subplots_adjust(hspace=0.3)
    
    # CRITICAL: Re-enforce limits one final time
    if xlim_timeseries is not None:
        ax2.set_xlim(xlim_timeseries[0], xlim_timeseries[1], auto=False, emit=False)
    if ylim_timeseries is not None:
        ax2.set_ylim(ylim_timeseries[0], ylim_timeseries[1], auto=False, emit=False)
    
    # Final enforcement - disable autoscaling again
    ax2.set_autoscalex_on(False)
    ax2.set_autoscaley_on(False)
    ax2.autoscale(enable=False)


def create_animation(
    results_ref: List[Dict],
    results_gliders: List[Dict],
    output_file: str = "animation_mhd_jun03_lc+lce.mp4",
    fps: int = 2,
    ref_label: str = "HYCOM_REF",
    gliders_label: str = "HYCOM_GLIDERS",
):
    """Create animation from processed results. Uses results_ref / results_gliders as-is (no date filter)."""
    results_ref = sorted(results_ref, key=lambda x: x.get('date') or '')
    results_gliders = sorted(results_gliders, key=lambda x: x.get('date') or '')

    if len(results_ref) == 0:
        print("Error: No REF results to animate")
        return
    
    # One datetime per result (fallback if date missing)
    def _to_dt(r):
        d = r.get('date')
        return datetime.strptime(d, "%Y%m%d") if d else datetime(1900, 1, 1)
    dates_ref = [_to_dt(r) for r in results_ref]
    dates_gliders = [_to_dt(r) for r in results_gliders]
    
    # Calculate fixed axis limits for time series
    # X-axis: from first date to last date (with some padding)
    all_dates = dates_ref + dates_gliders
    if len(all_dates) > 0:
        date_min = min(all_dates)
        date_max = max(all_dates)
        # Add padding (5% of total range)
        date_range = (date_max - date_min).total_seconds()
        date_padding = timedelta(seconds=date_range * 0.05)
        xlim_timeseries = (date_min - date_padding, date_max + date_padding)
    else:
        xlim_timeseries = None
    
    # Y-axis: from 0 to max MHD value (with some padding)
    all_mhd_values = []
    for r in results_ref:
        m = r.get('mhd')
        if m is not None and np.isfinite(m):
            all_mhd_values.append(m * 111.0)
    for r in results_gliders:
        m = r.get('mhd')
        if m is not None and np.isfinite(m):
            all_mhd_values.append(m * 111.0)
    
    if len(all_mhd_values) > 0:
        y_min = max(0, np.min(all_mhd_values) - 5)
        y_max = np.max(all_mhd_values) + 10
        ylim_timeseries = (y_min, y_max)
    else:
        ylim_timeseries = None
    
    # Create figure with two subplots
    # Top subplot uses cartopy projection, bottom uses regular matplotlib
    fig = plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax2 = plt.subplot(2, 1, 2)
    
    # Set fixed limits on ax2 BEFORE animation starts
    if xlim_timeseries is not None:
        ax2.set_xlim(xlim_timeseries[0], xlim_timeseries[1])
        ax2.set_autoscalex_on(False)
    if ylim_timeseries is not None:
        ax2.set_ylim(ylim_timeseries[0], ylim_timeseries[1])
        ax2.set_autoscaley_on(False)
    ax2.autoscale(enable=False)
    
    # Animation function
    def animate(frame):
        create_animation_frame(
            ax1, ax2, frame,
            results_ref, results_gliders,
            dates_ref, dates_gliders,
            xlim_timeseries=xlim_timeseries,
            ylim_timeseries=ylim_timeseries,
            ref_label=ref_label,
            gliders_label=gliders_label,
        )
    
    # Create animation
    num_frames = len(results_ref)
    print(f"Creating animation with {num_frames} frames...")
    
    anim = FuncAnimation(
        fig, animate, frames=num_frames,
        interval=1000/fps, repeat=True, blit=False
    )
    
    # Save animation
    print(f"Saving animation to {output_file}...")
    try:
        # Try ffmpeg first (for MP4)
        anim.save(output_file, writer='ffmpeg', fps=fps, bitrate=1800)
    except Exception as e:
        print(f"Warning: ffmpeg not available, trying alternative writer: {e}")
        # Fallback to imagemagick if ffmpeg not available
        try:
            anim.save(output_file, writer='imagemagick', fps=fps)
        except Exception as e2:
            print(f"Error: Could not save animation: {e2}")
            return
    print(f"Animation saved to {output_file}")
    
    plt.close()


def first_detachment_day_from_max_lat_series(
    lead_max_lat: List[Tuple[int, float]],
    drop_degrees: float = LC_NORTH_DROP_DEGREES,
    lead_has_lce: Optional[Dict[int, bool]] = None,
) -> Optional[int]:
    """
    Given a sorted list of (lead_time, max_latitude), return the first lead day where
    max_lat drops by more than drop_degrees from the running maximum so far (detachment).
    When lead_has_lce is provided, uses 0.5° threshold and also requires a new LCE to appear.
    When both lce_now and lce_prev are False (no LCE activity), running_max is lowered to
    max_lat so that a stale peak does not trigger a false detachment later.
    """
    if len(lead_max_lat) < 2:
        return None
    use_lce = lead_has_lce is not None
    threshold = 0.5 if use_lce else drop_degrees
    running_max = lead_max_lat[0][1]
    prev_lead = lead_max_lat[0][0]
    for lead, max_lat in lead_max_lat[1:]:
        if max_lat < running_max - threshold:
            if use_lce:
                lce_now = lead_has_lce.get(lead, False)
                lce_prev = lead_has_lce.get(prev_lead, False)
                if not (lce_now and not lce_prev):
                    # No new LCE: lower running_max only when there is no LCE activity at
                    # all (both False), so a stale peak cannot trigger a false detachment.
                    # If lce_prev=True (LCE just disappeared) keep running_max high so a
                    # reappearing LCE the next day still fires correctly.
                    if not lce_now and not lce_prev:
                        running_max = max_lat
                    else:
                        running_max = max(running_max, max_lat)
                    prev_lead = lead
                    continue
            return lead
        running_max = max(running_max, max_lat)
        prev_lead = lead
    return None


def count_detachments_from_max_lat_series(
    lead_max_lat: List[Tuple[int, float]],
    drop_degrees: float = LC_NORTH_DROP_DEGREES,
    lead_has_lce: Optional[Dict[int, bool]] = None,
) -> int:
    """
    Count all detachment events in a (lead_time, max_latitude) series.
    A detachment is when max_lat drops more than drop_degrees from the running max.
    After each detachment the running max resets to the current max_lat.
    When lead_has_lce is provided, uses 1° threshold and also requires a new LCE to appear.
    """
    if len(lead_max_lat) < 2:
        return 0
    use_lce = lead_has_lce is not None
    threshold = 0.5 if use_lce else drop_degrees
    count = 0
    running_max = lead_max_lat[0][1]
    prev_lead = lead_max_lat[0][0]
    for lead, max_lat in lead_max_lat[1:]:
        if max_lat < running_max - threshold:
            if use_lce:
                lce_now = lead_has_lce.get(lead, False)
                lce_prev = lead_has_lce.get(prev_lead, False)
                if not (lce_now and not lce_prev):
                    if not lce_now and not lce_prev:
                        running_max = max_lat
                    else:
                        running_max = max(running_max, max_lat)
                    prev_lead = lead
                    continue
            count += 1
            running_max = max_lat
        else:
            running_max = max(running_max, max_lat)
        prev_lead = lead
    return count


def detect_final_separation(
    date_max_lat: List[Tuple[datetime, float]],
    drop_degrees: float = 2.0,
    recovery_days: int = 30,
    date_has_lce: Optional[Dict[datetime, bool]] = None,
) -> Tuple[Optional[datetime], bool]:
    """
    Detect the final LC separation from a (date, max_lat) time series.

    Classic mode (date_has_lce is None):
      drop >= drop_degrees (2°) with no latitude recovery within recovery_days.

    LCE-assisted mode (date_has_lce provided):
      Trigger: drop >= 0.5° + new LCE appeared (present now, absent previous step).
      Confirmation over next recovery_days: latitude does not rise more than drop_degrees (2°)
      above the trigger latitude AND LCE is continuously detected.

    Returns (separation_date, is_confirmed):
      is_confirmed=True  if >= recovery_days of data remain after the drop
      is_confirmed=False if < recovery_days remain (uncertain)
    Returns (None, False) if no qualifying drop is found.
    """
    if len(date_max_lat) < 2:
        return None, False
    series = sorted(date_max_lat, key=lambda x: x[0])
    use_lce = date_has_lce is not None
    drop_trigger = 0.5 if use_lce else drop_degrees
    running_max = series[0][1]
    i = 1
    while i < len(series):
        date, max_lat = series[i]
        if max_lat < running_max - drop_trigger:
            if use_lce:
                # Require new LCE appeared at this step
                prev_date = series[i - 1][0]
                lce_now = date_has_lce.get(date, False)
                lce_prev = date_has_lce.get(prev_date, False)
                if not (lce_now and not lce_prev):
                    running_max = max(running_max, max_lat)
                    i += 1
                    continue
                # Confirm: within recovery_days, LC stays > 0.5° below pre-drop peak AND LCE persists
                ahead = [(d, lat) for d, lat in series[i + 1:] if (d - date).days <= recovery_days]
                lat_recovered = any(lat >= max_lat + drop_degrees for _, lat in ahead)
                lce_continuous = all(date_has_lce.get(d, False) for d, _ in ahead)
                if lat_recovered or not lce_continuous:
                    i += 1
                    continue
            else:
                # Classic: latitude must not recover within recovery_days
                ahead = [(d, lat) for d, lat in series[i + 1:] if (d - date).days <= recovery_days]
                if any(lat >= running_max - drop_degrees for _, lat in ahead):
                    running_max = max_lat
                    i += 1
                    continue
            last_date = series[-1][0]
            days_remaining = (last_date - date).days
            return date, days_remaining >= recovery_days
        running_max = max(running_max, max_lat)
        i += 1
    return None, False


def process_hycom_file_for_lc_only_mhd(
    hycom_file: str,
    grid_file: str,
    aviso_dir: str,
    mdt_path: str,
    forecast_start: datetime,
    lon_cutoff: float = -81.0,
) -> Optional[Dict]:
    """
    Lightweight LC-only MHD: loads SSH and extracts LC contour at 17 cm, then computes MHD.
    Skips LCE detection entirely. Returns result dict with mhd == mhd_lc_only.
    """
    date = extract_date_from_filename(hycom_file)
    if date is None:
        return None
    date_obj = datetime.strptime(date, "%Y%m%d")
    lead = (date_obj - forecast_start).days
    if lead < 0 or lead > MAX_LEAD_DAYS:
        return None
    try:
        contour_hycom, contour_aviso = get_hycom_aviso_contours(
            hycom_archv_file=hycom_file,
            hycom_grid_file=grid_file,
            date=date,
            aviso_dir=aviso_dir,
            mdt_path=mdt_path,
        )
        contour_hycom = filter_contour_from_latitude(contour_hycom, 21.0) if contour_hycom is not None else None
        contour_aviso = filter_contour_from_latitude(contour_aviso, 21.0) if contour_aviso is not None else None
        contour_hycom, contour_aviso = clip_contours_to_longitude_cutoff(
            contour_hycom, contour_aviso, lon_cutoff=lon_cutoff
        )
        if contour_hycom is None or contour_aviso is None:
            return None
        h_lc = clip_contours_to_lon_min(contour_hycom, MHD_LON_MIN)
        a_lc = clip_contours_to_lon_min(contour_aviso, MHD_LON_MIN)
        if len(h_lc) == 0 or len(a_lc) == 0:
            return None
        mhd_val = mhd(h_lc, a_lc, symmetric=True)
        return {
            'date': date,
            'mhd': mhd_val,
            'mhd_lc_only': mhd_val,
            'forecast_start': forecast_start,
            'lead_time_days': lead,
            'success': True,
        }
    except Exception:
        return None


def process_hycom_file_for_timing_only(
    hycom_file: str,
    grid_file: str,
    aviso_dir: str,
    mdt_path: str,
    forecast_start: datetime,
    lon_cutoff: float = -81.0,
    load_aviso: bool = True,
    detect_lce: bool = False,
    lce_min_lon_span: float = 1.3,
    sep_min_lon: Optional[float] = None,
    det_min_lon: Optional[float] = None,
) -> Optional[Tuple]:
    """
    Lightweight: load LC contours only, filter/clip, return (lead, max_lat_model, max_lat_aviso[, has_lce_timing, has_lce_sep, has_lce_det]).
    No MHD. Pass detect_lce=True to get LCE flags as elements 4-6:
      has_lce_timing (4th): primary LCE, default box — for --timing-distribution
      has_lce_sep    (5th): LCE with sep_min_lon west boundary — for --separation-timing
      has_lce_det    (6th): LCE with det_min_lon west boundary — for --detachment-count
    Returns None if date/contours invalid.
    """
    date = extract_date_from_filename(hycom_file)
    if date is None:
        return None
    date_obj = datetime.strptime(date, "%Y%m%d")
    lead = (date_obj - forecast_start).days
    if lead < 0 or lead > MAX_LEAD_DAYS:
        return None
    try:
        contour_hycom, contour_aviso = get_hycom_aviso_contours(
            hycom_archv_file=hycom_file,
            hycom_grid_file=grid_file,
            date=date,
            aviso_dir=aviso_dir,
            mdt_path=mdt_path,
            load_aviso=load_aviso,
        )
        contour_hycom = filter_contour_from_latitude(contour_hycom, 21.0) if contour_hycom is not None else None
        contour_aviso = filter_contour_from_latitude(contour_aviso, 21.0) if contour_aviso is not None else None
        contour_hycom, contour_aviso = clip_contours_to_longitude_cutoff(
            contour_hycom, contour_aviso, lon_cutoff=lon_cutoff
        )
        max_lat_model = float(np.max(contour_hycom[:, 1])) if contour_hycom is not None and len(contour_hycom) > 0 else None
        max_lat_aviso = float(np.max(contour_aviso[:, 1])) if contour_aviso is not None and len(contour_aviso) > 0 else None
        if max_lat_model is None and (not load_aviso or max_lat_aviso is None):
            return None
        if detect_lce:
            try:
                lon_h, lat_h, ssh_dm = load_ssh_and_grid_hycom(hycom_file, grid_file)
                ssh_h = demean_region_hycom(ssh_dm / 10.0, lon_h, lat_h, grid_file, DEMEAN_BBOX)
                _base = dict(level_min=0.17, level_max=0.17, num_levels=1, min_lon_span=lce_min_lon_span)
                lce_timing_flag = len(find_lce_region_contours(lon_h, lat_h, ssh_h, **_base)) > 0
                _sep_contours = find_lce_region_contours(lon_h, lat_h, ssh_h, **_base, min_lon=sep_min_lon) if sep_min_lon is not None else []
                lce_sep_flag = len(_sep_contours) > 0
                lce_det_flag = (
                    len(find_lce_region_contours(lon_h, lat_h, ssh_h, **_base, min_lon=det_min_lon)) > 0
                    if det_min_lon is not None else lce_timing_flag
                )
            except Exception:
                lce_timing_flag = lce_sep_flag = lce_det_flag = False
            return (lead, max_lat_model, max_lat_aviso, lce_timing_flag, lce_sep_flag, lce_det_flag)
        return (lead, max_lat_model, max_lat_aviso)
    except Exception:
        return None


def compute_divergence_from_series(
    series_model: List[Tuple[int, float]],
    series_aviso: List[Tuple[int, float]],
    lead_has_lce_model: Optional[Dict[int, bool]] = None,
) -> Optional[int]:
    """From (lead, max_lat) series for model and AVISO, return model_first - aviso_first or None."""
    first_model = first_detachment_day_from_max_lat_series(series_model, lead_has_lce=lead_has_lce_model)
    first_aviso = first_detachment_day_from_max_lat_series(series_aviso)
    if first_model is not None and first_aviso is not None:
        return first_model - first_aviso
    return None


def compute_divergence_from_results(
    results: List[Dict],
    forecast_start: datetime,
    lon_cutoff: float = -81.0,
    max_lead_days: Optional[int] = MAX_LEAD_DAYS,
) -> Optional[int]:
    """
    From a list of results (each with date, contour_hycom_full, contour_aviso_full),
    build (lead, max_lat) series for model and AVISO, find first detachment day each,
    return model_first - aviso_first (days), or None if either has no LCE.
    """
    series_model, series_aviso = _build_max_lat_series_from_results(
        results=results,
        forecast_start=forecast_start,
        lon_cutoff=lon_cutoff,
        max_lead_days=max_lead_days,
    )
    series_model.sort(key=lambda x: x[0])
    series_aviso.sort(key=lambda x: x[0])
    first_model = first_detachment_day_from_max_lat_series(series_model)
    first_aviso = first_detachment_day_from_max_lat_series(series_aviso)
    if first_model is not None and first_aviso is not None:
        return first_model - first_aviso
    return None


def _build_max_lat_series_from_results(
    results: List[Dict],
    forecast_start: datetime,
    lon_cutoff: float = -81.0,
    max_lead_days: Optional[int] = MAX_LEAD_DAYS,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Build (lead, max_lat) series for model and AVISO from processed results."""
    series_model: List[Tuple[int, float]] = []
    series_aviso: List[Tuple[int, float]] = []
    for r in results:
        if not r.get("success") or r.get("date") is None:
            continue
        date_obj = datetime.strptime(r["date"], "%Y%m%d")
        lead = (date_obj - forecast_start).days
        if lead < 0:
            continue
        if max_lead_days is not None and lead > max_lead_days:
            continue
        ch = r.get("contour_hycom_full")
        ca = r.get("contour_aviso_full")
        if ch is not None and len(ch) > 0:
            ch_f = filter_contour_from_latitude(ch, 21.0)
            if ch_f is not None and len(ch_f) > 0:
                ch_c = ch_f[ch_f[:, 0] < lon_cutoff]
                if len(ch_c) > 0:
                    series_model.append((lead, float(np.max(ch_c[:, 1]))))
        if ca is not None and len(ca) > 0:
            ca_f = filter_contour_from_latitude(ca, 21.0)
            if ca_f is not None and len(ca_f) > 0:
                ca_c = ca_f[ca_f[:, 0] < lon_cutoff]
                if len(ca_c) > 0:
                    series_aviso.append((lead, float(np.max(ca_c[:, 1]))))
    series_model.sort(key=lambda x: x[0])
    series_aviso.sort(key=lambda x: x[0])
    return series_model, series_aviso


def _max_drop_from_running_max(lead_max_lat: List[Tuple[int, float]]) -> float:
    """Largest drop from running maximum in a (lead, max_lat) series."""
    if len(lead_max_lat) < 2:
        return 0.0
    running_max = lead_max_lat[0][1]
    max_drop = 0.0
    for _, max_lat in lead_max_lat[1:]:
        max_drop = max(max_drop, running_max - max_lat)
        running_max = max(running_max, max_lat)
    return float(max_drop)


def _lc_only_results(results: List[Dict]) -> List[Dict]:
    """Return a copy of results with mhd replaced by mhd_lc_only, for LC-only plots."""
    out = []
    for r in results:
        rc = dict(r)
        rc['mhd'] = rc.get('mhd_lc_only', np.nan)
        out.append(rc)
    return out


def plot_timeseries_all_forecasts(
    results_ref: List[Dict],
    results_gliders: List[Dict],
    output_dir: str,
    ref_label: str = "REF",
    gliders_label: str = "GLIDERS",
    suffix: str = "",
) -> None:
    """Static plot: if results have 'forecast_start', plot lead time vs MHD (one line per forecast + mean). Else date vs MHD."""
    fig, ax = plt.subplots(figsize=(14, 6))
    valid_ref = [r for r in results_ref if r.get("success") and r.get("date")]
    valid_gliders = [r for r in results_gliders if r.get("success") and r.get("date")]
    has_forecast_start = valid_ref and valid_ref[0].get("forecast_start") is not None
    if has_forecast_start:
        # Group by forecast_start, plot lead time vs MHD per forecast + mean ± std
        soft_blue = "#6BAED6"
        ref_by_fs = defaultdict(list)
        for r in valid_ref:
            fs = r.get("forecast_start")
            if fs is not None:
                ref_by_fs[fs].append(r)
        all_lead_ref = defaultdict(list)
        for fs, group in sorted(ref_by_fs.items()):
            leads, mhds = [], []
            for r in group:
                lead = r.get("lead_time_days")
                if lead is None:
                    date_obj = datetime.strptime(r["date"], "%Y%m%d")
                    lead = (date_obj - fs).days
                if 0 <= lead <= MAX_LEAD_DAYS:
                    leads.append(lead)
                    mhds.append(r["mhd"] * 111.0)
                    all_lead_ref[lead].append(r["mhd"] * 111.0)
            if leads:
                s = sorted(zip(leads, mhds))
                ax.plot([x[0] for x in s], [x[1] for x in s], "-", linewidth=1, color=soft_blue, alpha=0.5)
        if all_lead_ref:
            lead_sorted = sorted(all_lead_ref.keys())
            mean_ref = [np.mean(all_lead_ref[lt]) for lt in lead_sorted]
            std_ref  = [np.std(all_lead_ref[lt])  for lt in lead_sorted]
            ax.plot(lead_sorted, mean_ref, "-", linewidth=2.5, color="steelblue", label=f"Mean ({ref_label})", zorder=5)
            ax.fill_between(lead_sorted,
                            [m - s for m, s in zip(mean_ref, std_ref)],
                            [m + s for m, s in zip(mean_ref, std_ref)],
                            color="steelblue", alpha=0.2)
        gliders_by_fs = defaultdict(list)
        for r in valid_gliders:
            fs = r.get("forecast_start")
            if fs is not None:
                gliders_by_fs[fs].append(r)
        all_lead_gl = defaultdict(list)
        for fs, group in sorted(gliders_by_fs.items()):
            leads, mhds = [], []
            for r in group:
                lead = r.get("lead_time_days")
                if lead is None:
                    date_obj = datetime.strptime(r["date"], "%Y%m%d")
                    lead = (date_obj - fs).days
                if 0 <= lead <= MAX_LEAD_DAYS:
                    leads.append(lead)
                    mhds.append(r["mhd"] * 111.0)
                    all_lead_gl[lead].append(r["mhd"] * 111.0)
            if leads:
                s = sorted(zip(leads, mhds))
                ax.plot([x[0] for x in s], [x[1] for x in s], "-", linewidth=1, color="salmon", alpha=0.5)
        if all_lead_gl:
            lead_sorted = sorted(all_lead_gl.keys())
            mean_gl = [np.mean(all_lead_gl[lt]) for lt in lead_sorted]
            std_gl  = [np.std(all_lead_gl[lt])  for lt in lead_sorted]
            ax.plot(lead_sorted, mean_gl, "-", linewidth=2.5, color="darkred", label=f"Mean ({gliders_label})", zorder=5)
            ax.fill_between(lead_sorted,
                            [m - s for m, s in zip(mean_gl, std_gl)],
                            [m + s for m, s in zip(mean_gl, std_gl)],
                            color="salmon", alpha=0.2)
        ax.set_xlabel("Lead Time (days)", fontsize=11)
        ax.set_xlim(0, MAX_LEAD_DAYS)
    else:
        if valid_ref:
            dates_ref = [datetime.strptime(r["date"], "%Y%m%d") for r in valid_ref]
            mhd_ref = [r["mhd"] * 111.0 for r in valid_ref]
            ax.plot(dates_ref, mhd_ref, "o-", linewidth=2, markersize=5, color="steelblue", label=f"MHD: AVISO vs {ref_label}")
        if valid_gliders:
            dates_gl = [datetime.strptime(r["date"], "%Y%m%d") for r in valid_gliders]
            mhd_gl = [r["mhd"] * 111.0 for r in valid_gliders]
            ax.plot(dates_gl, mhd_gl, "o-", linewidth=2, markersize=5, color="red", label=f"MHD: AVISO vs {gliders_label}")
        ax.set_xlabel("Date", fontsize=11)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Modified Hausdorff Distance (km)", fontsize=11)
    ax.set_ylim(0, 350)
    ax.set_title("MHD time series (all forecasts)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, f"mhd_timeseries_all_forecasts{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def plot_timeseries_by_date(
    results_ref: List[Dict],
    results_gliders: List[Dict],
    output_dir: str,
    ref_label: str = "REF",
    gliders_label: str = "GLIDERS",
    suffix: str = "",
) -> None:
    """MHD vs actual date: one thin line per forecast, plus mean ± std shading across forecasts on each date."""

    def _collect(results):
        by_fs = defaultdict(list)    # forecast_start -> [(date, mhd_km)]
        by_date = defaultdict(list)  # actual date    -> [mhd_km, ...]
        for r in results:
            if not r.get("success") or not r.get("date") or not np.isfinite(r.get("mhd", np.nan)):
                continue
            fs = r.get("forecast_start")
            if fs is None:
                continue
            date_obj = datetime.strptime(r["date"], "%Y%m%d")
            mhd_km = r["mhd"] * 111.0
            by_fs[fs].append((date_obj, mhd_km))
            by_date[date_obj].append(mhd_km)
        return by_fs, by_date

    ref_by_fs, ref_by_date = _collect(results_ref)
    gl_by_fs, gl_by_date = _collect(results_gliders)

    if not ref_by_date and not gl_by_date:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    soft_blue = "#6BAED6"

    for fs, pts in sorted(ref_by_fs.items()):
        pts_s = sorted(pts)
        ax.plot([p[0] for p in pts_s], [p[1] for p in pts_s],
                "-", linewidth=1, color=soft_blue, alpha=0.5)

    if ref_by_date:
        dates_s = sorted(ref_by_date.keys())
        mean_ref = [np.mean(ref_by_date[d]) for d in dates_s]
        std_ref  = [np.std(ref_by_date[d])  for d in dates_s]
        ax.plot(dates_s, mean_ref, "-", linewidth=2.5, color="steelblue",
                label=f"Mean ({ref_label})", zorder=5)
        ax.fill_between(dates_s,
                        [m - s for m, s in zip(mean_ref, std_ref)],
                        [m + s for m, s in zip(mean_ref, std_ref)],
                        color="steelblue", alpha=0.2)

    for fs, pts in sorted(gl_by_fs.items()):
        pts_s = sorted(pts)
        ax.plot([p[0] for p in pts_s], [p[1] for p in pts_s],
                "-", linewidth=1, color="salmon", alpha=0.5)

    if gl_by_date:
        dates_s = sorted(gl_by_date.keys())
        mean_gl = [np.mean(gl_by_date[d]) for d in dates_s]
        std_gl  = [np.std(gl_by_date[d])  for d in dates_s]
        ax.plot(dates_s, mean_gl, "-", linewidth=2.5, color="darkred",
                label=f"Mean ({gliders_label})", zorder=5)
        ax.fill_between(dates_s,
                        [m - s for m, s in zip(mean_gl, std_gl)],
                        [m + s for m, s in zip(mean_gl, std_gl)],
                        color="salmon", alpha=0.2)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Modified Hausdorff Distance (km)", fontsize=11)
    ax.set_ylim(0, 350)
    ax.set_title("MHD vs actual date (mean ± std across forecasts)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, f"mhd_timeseries_by_date{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def plot_mean_std_by_date(
    results_ref: List[Dict],
    results_gliders: List[Dict],
    output_dir: str,
    ref_label: str = "REF",
    gliders_label: str = "GLIDERS",
    suffix: str = "",
) -> None:
    """MHD mean ± std vs actual date (no individual forecast lines)."""

    def _collect(results):
        by_date = defaultdict(list)
        for r in results:
            if not r.get("success") or not r.get("date") or not np.isfinite(r.get("mhd", np.nan)):
                continue
            if r.get("forecast_start") is None:
                continue
            date_obj = datetime.strptime(r["date"], "%Y%m%d")
            by_date[date_obj].append(r["mhd"] * 111.0)
        return by_date

    ref_by_date = _collect(results_ref)
    gl_by_date  = _collect(results_gliders)

    if not ref_by_date and not gl_by_date:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    if ref_by_date:
        dates_s  = sorted(ref_by_date.keys())
        mean_ref = [np.mean(ref_by_date[d]) for d in dates_s]
        std_ref  = [np.std(ref_by_date[d])  for d in dates_s]
        ax.plot(dates_s, mean_ref, "-", linewidth=2.5, color="steelblue",
                label=f"Mean ({ref_label})", zorder=5)
        ax.fill_between(dates_s,
                        [m - s for m, s in zip(mean_ref, std_ref)],
                        [m + s for m, s in zip(mean_ref, std_ref)],
                        color="steelblue", alpha=0.2)

    if gl_by_date:
        dates_s = sorted(gl_by_date.keys())
        mean_gl = [np.mean(gl_by_date[d]) for d in dates_s]
        std_gl  = [np.std(gl_by_date[d])  for d in dates_s]
        ax.plot(dates_s, mean_gl, "-", linewidth=2.5, color="darkred",
                label=f"Mean ({gliders_label})", zorder=5)
        ax.fill_between(dates_s,
                        [m - s for m, s in zip(mean_gl, std_gl)],
                        [m + s for m, s in zip(mean_gl, std_gl)],
                        color="salmon", alpha=0.2)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Modified Hausdorff Distance (km)", fontsize=11)
    ax.set_ylim(0, 350)
    ax.set_title("MHD mean ± std vs actual date", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, f"mhd_mean_std_by_date{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def plot_mean_std_from_results(
    results_ref: List[Dict],
    results_gliders: List[Dict],
    forecast_start: Optional[datetime],
    output_dir: str,
    ref_label: str = "REF",
    gliders_label: str = "GLIDERS",
    suffix: str = "",
) -> None:
    """Mean ± std MHD (km) vs lead time (days). Same logic as pycodes/plot_mhd_timeseries_mean_std: REF-only, GLIDERS-only, and both plots."""
    ref_data = defaultdict(list)
    for r in results_ref:
        if not r.get("success") or r.get("date") is None:
            continue
        fs = r.get("forecast_start", forecast_start)
        if fs is None:
            continue
        lead = r.get("lead_time_days")
        if lead is None:
            date_obj = datetime.strptime(r["date"], "%Y%m%d")
            lead = (date_obj - fs).days
        if 0 <= lead <= MAX_LEAD_DAYS:
            ref_data[lead].append(r["mhd"] * 111.0)
    gliders_data = defaultdict(list)
    for r in results_gliders:
        if not r.get("success") or r.get("date") is None:
            continue
        fs = r.get("forecast_start", forecast_start)
        if fs is None:
            continue
        lead = r.get("lead_time_days")
        if lead is None:
            date_obj = datetime.strptime(r["date"], "%Y%m%d")
            lead = (date_obj - fs).days
        if 0 <= lead <= MAX_LEAD_DAYS:
            gliders_data[lead].append(r["mhd"] * 111.0)
    has_ref = bool(ref_data)
    has_gliders = bool(gliders_data)
    # REF only
    if ref_data:
        lead_sorted = sorted(ref_data.keys())
        mean_ref = [np.mean(ref_data[lt]) for lt in lead_sorted]
        std_ref = [np.std(ref_data[lt]) for lt in lead_sorted]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lead_sorted, mean_ref, "-", color="steelblue", linewidth=2.5)
        ax.fill_between(lead_sorted, np.array(mean_ref) - np.array(std_ref), np.array(mean_ref) + np.array(std_ref), color="steelblue", alpha=0.2)
        ax.set_xlabel("Lead Time (days)", fontsize=12)
        ax.set_xlim(0, MAX_LEAD_DAYS)
        ax.set_ylabel("Modified Hausdorff Distance (km)", fontsize=12)
        ax.set_title(f"MHD: AVISO vs {ref_label} (mean ± std)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 200)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f"plot_mhd_timeseries_mean_std_ref{suffix}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote {path}")
    # GLIDERS only
    if gliders_data:
        lead_sorted = sorted(gliders_data.keys())
        mean_gl = [np.mean(gliders_data[lt]) for lt in lead_sorted]
        std_gl = [np.std(gliders_data[lt]) for lt in lead_sorted]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lead_sorted, mean_gl, "-", color="darkred", linewidth=2.5)
        ax.fill_between(lead_sorted, np.array(mean_gl) - np.array(std_gl), np.array(mean_gl) + np.array(std_gl), color="salmon", alpha=0.2)
        ax.set_xlabel("Lead Time (days)", fontsize=12)
        ax.set_xlim(0, MAX_LEAD_DAYS)
        ax.set_ylabel("Modified Hausdorff Distance (km)", fontsize=12)
        ax.set_title(f"MHD: AVISO vs {gliders_label} (mean ± std)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 200)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f"plot_mhd_timeseries_mean_std_gliders{suffix}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote {path}")
    # Both on one figure
    if has_ref and has_gliders:
        lead_sorted_ref = sorted(ref_data.keys())
        mean_ref = [np.mean(ref_data[lt]) for lt in lead_sorted_ref]
        std_ref = [np.std(ref_data[lt]) for lt in lead_sorted_ref]
        lead_sorted_gl = sorted(gliders_data.keys())
        mean_gl = [np.mean(gliders_data[lt]) for lt in lead_sorted_gl]
        std_gl = [np.std(gliders_data[lt]) for lt in lead_sorted_gl]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(lead_sorted_ref, np.array(mean_ref) - np.array(std_ref), np.array(mean_ref) + np.array(std_ref), color="steelblue", alpha=0.2)
        ax.plot(lead_sorted_ref, mean_ref, "-", color="steelblue", linewidth=2.5, label=ref_label)
        ax.fill_between(lead_sorted_gl, np.array(mean_gl) - np.array(std_gl), np.array(mean_gl) + np.array(std_gl), color="salmon", alpha=0.2)
        ax.plot(lead_sorted_gl, mean_gl, "-", color="darkred", linewidth=2.5, label=gliders_label)
        ax.legend()
        ax.set_xlabel("Lead Time (days)", fontsize=12)
        ax.set_xlim(0, MAX_LEAD_DAYS)
        ax.set_ylabel("Modified Hausdorff Distance (km)", fontsize=12)
        ax.set_title(f"MHD: AVISO vs {ref_label} and {gliders_label} (mean ± std)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 200)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f"plot_mhd_timeseries_mean_std_both{suffix}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote {path}")


def _plot_one_timing_histogram(
    items: List[Tuple[int, datetime]],
    output_dir: str,
    title: str,
    xlabel: str,
    filename: str,
    color: str = "#1f77b4",
) -> None:
    """One histogram (REF or GLIDERS). items is list of (divergence_days, forecast_start_date).
    Forecast dates are annotated inside bars."""
    if not items:
        return
    divergences = [d for d, _ in items]
    bin_edges = [-float("inf"), -25, -15, -5, 5, 15, 25, float("inf")]
    bin_centers = [-30, -20, -10, 0, 10, 20, 30]
    bin_labels = ["-25+", "-25 to -15", "-15 to -5", "-5 to 5", "5 to 15", "15 to 25", "25+"]
    widths = [10, 9, 9, 10, 9, 9, 10]
    n = len(items)

    bins: List[List[datetime]] = [[] for _ in bin_centers]
    for div, fs_dt in items:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= div < bin_edges[i + 1]:
                bins[i].append(fs_dt)
                break
    pcts = [100.0 * len(b) / n for b in bins]

    counts = [len(b) for b in bins]
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (cen, w, cnt) in enumerate(zip(bin_centers, widths, counts)):
        ax.bar(cen, cnt, width=w, color=color, alpha=0.8, edgecolor="black")
        if bins[i] and cnt > 0:
            for j, fs_dt in enumerate(bins[i]):
                ax.text(cen, j + 0.5, fs_dt.strftime("%Y-%m-%d"),
                        ha="center", va="center", fontsize=7, fontweight="bold", color="black", clip_on=True)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Number of forecasts", fontsize=12)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2 = ax.twinx()
    ax2.set_ylim(0, ax.get_ylim()[1] / n * 100)
    ax2.set_ylabel("% of occurrence", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")
    print(f"  Total: {n}, mean divergence: {np.mean(divergences):.2f} days, std: {np.std(divergences):.2f} days")


def plot_timing_distribution(
    timing_ref: List,
    timing_gliders: List,
    output_dir: str,
    ref_label: str = "HYCOM_REF",
    gliders_label: str = "HYCOM_GLIDERS",
) -> None:
    """Grouped-bar histogram: REF and GLIDERS side by side per bin. Each entry is (forecast_start, divergence_days)."""
    if not timing_ref and not timing_gliders:
        print("No divergence values for timing distribution.")
        return
    bin_edges = [-float("inf"), -25, -15, -5, 5, 15, 25, float("inf")]
    bin_centers = [-30, -20, -10, 0, 10, 20, 30]
    bin_labels = ["-25+", "-25 to -15", "-15 to -5", "-5 to 5", "5 to 15", "15 to 25", "25+"]
    widths = [10, 9, 9, 10, 9, 9, 10]

    def _bin(timing):
        b = [[] for _ in bin_centers]
        for entry in timing:
            fs, div = entry[0], entry[1]
            aviso_offset = entry[2] if len(entry) > 2 else None
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= div < bin_edges[i + 1]:
                    b[i].append((fs, div, aviso_offset))
                    break
        return b

    bins_ref = _bin(timing_ref)
    bins_gl  = _bin(timing_gliders)
    n_ref = len(timing_ref)
    n_gl  = len(timing_gliders)
    n_total = max(n_ref, n_gl, 1)

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (cen, w) in enumerate(zip(bin_centers, widths)):
        hw = w / 2 * 0.7
        # REF bar (left), GLIDERS bar (right) — touching at bin center
        cnt_r = len(bins_ref[i])
        ax.bar(cen - hw / 2, cnt_r, width=hw, color="steelblue", alpha=0.8, edgecolor="black",
               label=ref_label if i == 0 else None)
        for j, (fs_dt, div, aviso_offset) in enumerate(bins_ref[i]):
            if aviso_offset is not None:
                label = f"{aviso_offset}d"  # line 2: f"\n{div:+d}d"
                print(f"[TimingDist REF] forecast={fs_dt.strftime('%Y-%m-%d')} aviso_lead={aviso_offset}d div={div:+d}d")
            else:
                label = fs_dt.strftime('%-m/%-d/%y')  # line 2: f"\n{div:+d}d"
            ax.text(cen - hw / 2, j + 0.5, label,
                    ha="center", va="center", fontsize=6, fontweight="bold", color="white",
                    rotation=0, clip_on=True, linespacing=1.2)
        cnt_g = len(bins_gl[i])
        ax.bar(cen + hw / 2, cnt_g, width=hw, color="darkred", alpha=0.8, edgecolor="black",
               label=gliders_label if i == 0 else None)
        for j, (fs_dt, div, aviso_offset) in enumerate(bins_gl[i]):
            if aviso_offset is not None:
                label = f"{aviso_offset}d"  # line 2: f"\n{div:+d}d"
                print(f"[TimingDist GLIDERS] forecast={fs_dt.strftime('%Y-%m-%d')} aviso_lead={aviso_offset}d div={div:+d}d")
            else:
                label = fs_dt.strftime('%-m/%-d/%y')  # line 2: f"\n{div:+d}d"
            ax.text(cen + hw / 2, j + 0.5, label,
                    ha="center", va="center", fontsize=6, fontweight="bold", color="white",
                    rotation=0, clip_on=True, linespacing=1.2)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Divergence (days): model first LCE day − AVISO first LCE day\n← earlier than AVISO                                                    later than AVISO →", fontsize=12)
    ax.set_ylabel("Number of forecasts", fontsize=12)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2 = ax.twinx()
    ax2.set_ylim(0, ax.get_ylim()[1] / n_total * 100)
    ax2.set_ylabel("% of occurrence", fontsize=12)
    ax.set_title("Distribution of first LCE detachment timing vs AVISO", fontsize=14, fontweight="bold", pad=28)
    ax.text(0.5, 1.02, "Bar label: days from forecast start to AVISO first detachment",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontstyle="italic")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.3)
    mean_lines = []
    if timing_ref:
        divs = [e[1] for e in timing_ref]
        mean_lines.append(f"{ref_label}: mean = {np.mean(divs):.1f} d, std = {np.std(divs):.1f} d (n={n_ref})")
    if timing_gliders:
        divs = [e[1] for e in timing_gliders]
        mean_lines.append(f"{gliders_label}: mean = {np.mean(divs):.1f} d, std = {np.std(divs):.1f} d (n={n_gl})")
    if mean_lines:
        ax.text(0.02, 0.97, "\n".join(mean_lines), transform=ax.transAxes,
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "histogram_first_lce_divergence_aviso.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")
    if timing_ref:
        divs = [e[1] for e in timing_ref]
        print(f"  {ref_label}: n={n_ref}, mean={np.mean(divs):.2f} days, std={np.std(divs):.2f} days")
    if timing_gliders:
        divs = [e[1] for e in timing_gliders]
        print(f"  {gliders_label}: n={n_gl}, mean={np.mean(divs):.2f} days, std={np.std(divs):.2f} days")


def _plot_separation_histogram(
    confirmed: List[Tuple[int, datetime]],
    uncertain: List[Tuple[int, datetime]],
    output_dir: str,
    title: str,
    xlabel: str,
    filename: str,
    color: str,
) -> None:
    """Histogram of separation timing with confirmed (solid) and uncertain (hatched) bars stacked.
    Each entry is (divergence_days, forecast_start_date). Forecast dates are annotated inside bars."""
    all_items = confirmed + uncertain
    if not all_items:
        return
    bin_edges = [-float("inf"), -25, -15, -5, 5, 15, 25, float("inf")]
    bin_centers = [-30, -20, -10, 0, 10, 20, 30]
    bin_labels = ["-25+", "-25 to -15", "-15 to -5", "-5 to 5", "5 to 15", "15 to 25", "25+"]
    widths = [10, 9, 9, 10, 9, 9, 10]
    n_total = len(all_items)

    def _bin_items(items):
        bins = [[] for _ in bin_centers]
        for div, fs_dt in items:
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= div < bin_edges[i + 1]:
                    bins[i].append(fs_dt)
                    break
        return bins

    conf_bins = _bin_items(confirmed)
    unc_bins  = _bin_items(uncertain)
    counts_confirmed = [len(b) for b in conf_bins]
    counts_uncertain = [len(b) for b in unc_bins]

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (cen, w) in enumerate(zip(bin_centers, widths)):
        ax.bar(cen, counts_confirmed[i], width=w, color=color, alpha=0.8, edgecolor="black")
        ax.bar(cen, counts_uncertain[i], width=w, bottom=counts_confirmed[i],
               color=color, alpha=0.4, edgecolor="black", hatch="//",
               label="uncertain" if i == 0 else None)
        all_dates = [(d, True) for d in conf_bins[i]] + [(d, False) for d in unc_bins[i]]
        total_bar_cnt = counts_confirmed[i] + counts_uncertain[i]
        if all_dates and total_bar_cnt > 0:
            for j, (fs_dt, _is_conf) in enumerate(all_dates):
                ax.text(
                    cen, j + 0.5,
                    fs_dt.strftime("%Y-%m-%d"),
                    ha="center", va="center",
                    fontsize=7, fontweight="bold", color="black",
                    clip_on=True,
                )
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Number of forecasts", fontsize=12)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2 = ax.twinx()
    ax2.set_ylim(0, ax.get_ylim()[1] / n_total * 100)
    ax2.set_ylabel("% of occurrence", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.3)
    if uncertain:
        ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")
    confirmed_divs = [d for d, _ in confirmed]
    uncertain_divs = [d for d, _ in uncertain]
    all_divs = confirmed_divs + uncertain_divs
    print(f"  Total: {n_total} ({len(confirmed)} confirmed, {len(uncertain)} uncertain), mean: {np.mean(all_divs):.2f} days, std: {np.std(all_divs):.2f} days")


def plot_separation_timing(
    confirmed_ref: List[Tuple[int, datetime]],
    uncertain_ref: List[Tuple[int, datetime]],
    confirmed_gliders: List[Tuple[int, datetime]],
    uncertain_gliders: List[Tuple[int, datetime]],
    output_dir: str,
    ref_label: str = "REF",
    gliders_label: str = "GLIDERS",
    aviso_sep_date: Optional[datetime] = None,
) -> None:
    """Grouped-bar histogram of final LC separation timing: REF and GLIDERS side by side per bin."""
    all_ref = confirmed_ref + uncertain_ref
    all_gl  = confirmed_gliders + uncertain_gliders
    if not all_ref and not all_gl:
        print("No separation timing values.")
        return
    aviso_str = f" (AVISO: {aviso_sep_date.strftime('%Y-%m-%d')})" if aviso_sep_date else ""
    bin_edges = [-float("inf"), -25, -15, -5, 5, 15, 25, float("inf")]
    bin_centers = [-30, -20, -10, 0, 10, 20, 30]
    bin_labels = ["-25+", "-25 to -15", "-15 to -5", "-5 to 5", "5 to 15", "15 to 25", "25+"]
    widths = [10, 9, 9, 10, 9, 9, 10]
    n_total = max(len(all_ref), len(all_gl), 1)

    def _bin_sep(confirmed, uncertain):
        conf_b = [[] for _ in bin_centers]
        unc_b  = [[] for _ in bin_centers]
        for items, dest in [(confirmed, conf_b), (uncertain, unc_b)]:
            for div, fs_dt in items:
                for i in range(len(bin_edges) - 1):
                    if bin_edges[i] <= div < bin_edges[i + 1]:
                        dest[i].append((fs_dt, div))
                        break
        return conf_b, unc_b

    conf_r, unc_r = _bin_sep(confirmed_ref, uncertain_ref)
    conf_g, unc_g = _bin_sep(confirmed_gliders, uncertain_gliders)

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (cen, w) in enumerate(zip(bin_centers, widths)):
        hw = w / 2 * 0.7
        # REF bar (left), GLIDERS bar (right) — touching at bin center
        cnt_rc = len(conf_r[i]); cnt_ru = len(unc_r[i])
        ax.bar(cen - hw / 2, cnt_rc, width=hw, color="steelblue", alpha=0.8, edgecolor="black",
               label=f"{ref_label} confirmed" if i == 0 else None)
        ax.bar(cen - hw / 2, cnt_ru, width=hw, bottom=cnt_rc, color="steelblue", alpha=0.4,
               edgecolor="black", hatch="//", label=f"{ref_label} uncertain" if i == 0 else None)
        for j, (fs_dt, div) in enumerate(conf_r[i] + unc_r[i]):
            if aviso_sep_date is not None:
                offset_days = (aviso_sep_date - fs_dt).days
                offset_str = f"{offset_days}d"  # line 2: f"\n{div:+d}d"
                print(f"[SepTiming REF] forecast={fs_dt.strftime('%Y-%m-%d')} aviso_sep={aviso_sep_date.strftime('%Y-%m-%d')} offset={offset_days}d div={div:+d}d")
            else:
                offset_str = fs_dt.strftime('%-m/%-d/%y')  # line 2: f"\n{div:+d}d"
            ax.text(cen - hw / 2, j + 0.5, offset_str,
                    ha="center", va="center", fontsize=6, fontweight="bold", color="white",
                    rotation=0, clip_on=True, linespacing=1.2)
        cnt_gc = len(conf_g[i]); cnt_gu = len(unc_g[i])
        ax.bar(cen + hw / 2, cnt_gc, width=hw, color="darkred", alpha=0.8, edgecolor="black",
               label=f"{gliders_label} confirmed" if i == 0 else None)
        ax.bar(cen + hw / 2, cnt_gu, width=hw, bottom=cnt_gc, color="darkred", alpha=0.4,
               edgecolor="black", hatch="//", label=f"{gliders_label} uncertain" if i == 0 else None)
        for j, (fs_dt, div) in enumerate(conf_g[i] + unc_g[i]):
            if aviso_sep_date is not None:
                offset_days = (aviso_sep_date - fs_dt).days
                offset_str = f"{offset_days}d"  # line 2: f"\n{div:+d}d"
                print(f"[SepTiming GLIDERS] forecast={fs_dt.strftime('%Y-%m-%d')} aviso_sep={aviso_sep_date.strftime('%Y-%m-%d')} offset={offset_days}d div={div:+d}d")
            else:
                offset_str = fs_dt.strftime('%-m/%-d/%y')  # line 2: f"\n{div:+d}d"
            ax.text(cen + hw / 2, j + 0.5, offset_str,
                    ha="center", va="center", fontsize=6, fontweight="bold", color="white",
                    rotation=0, clip_on=True, linespacing=1.2)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Days: model final separation − AVISO final separation\n← earlier than AVISO                                                    later than AVISO →", fontsize=12)
    ax.set_ylabel("Number of forecasts", fontsize=12)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2 = ax.twinx()
    ax2.set_ylim(0, ax.get_ylim()[1] / n_total * 100)
    ax2.set_ylabel("% of occurrence", fontsize=12)
    ax.set_title(f"Final LC separation timing vs AVISO{aviso_str}", fontsize=14, fontweight="bold", pad=28)
    ax.text(0.5, 1.02, "Bar label: days from forecast start to AVISO final separation",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontstyle="italic")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.3)
    # Mean annotations
    mean_lines = []
    if all_ref:
        divs_r = [d for d, _ in all_ref]
        mean_lines.append(f"{ref_label}: mean = {np.mean(divs_r):.1f} d, std = {np.std(divs_r):.1f} d (n={len(divs_r)})")
    if all_gl:
        divs_g = [d for d, _ in all_gl]
        mean_lines.append(f"{gliders_label}: mean = {np.mean(divs_g):.1f} d, std = {np.std(divs_g):.1f} d (n={len(divs_g)})")
    if mean_lines:
        ax.text(0.02, 0.97, "\n".join(mean_lines), transform=ax.transAxes,
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "histogram_final_separation_timing.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")
    for label, items in [(ref_label, all_ref), (gliders_label, all_gl)]:
        if items:
            divs = [d for d, _ in items]
            print(f"  {label}: n={len(items)}, mean={np.mean(divs):.2f} days, std={np.std(divs):.2f} days")


def plot_detachment_counts(
    counts_ref: List[Tuple[datetime, int, int]],
    counts_gliders: List[Tuple[datetime, int, int]],
    output_dir: str,
    ref_label: str = "REF",
    gliders_label: str = "GLIDERS",
    aviso_total: Optional[int] = None,
) -> None:
    """
    Grouped bar chart: detachment count per forecast start date.
    counts_ref / counts_gliders: list of (forecast_start, model_count, aviso_count).
    aviso_total: total AVISO detachments over the full time series (shown in legend label).
    """
    if not counts_ref and not counts_gliders:
        print("No detachment count data. Skip plot.")
        return

    all_dates = sorted(set(fs for fs, _, _ in counts_ref) | set(fs for fs, _, _ in counts_gliders))
    x = np.arange(len(all_dates))
    date_labels = [d.strftime("%Y-%m-%d") for d in all_dates]

    ref_map  = {fs: (m, a) for fs, m, a in counts_ref}
    gl_map   = {fs: (m, a) for fs, m, a in counts_gliders}

    ref_model = [ref_map.get(d, (0, 0))[0] for d in all_dates]
    ref_aviso = [ref_map.get(d, (0, 0))[1] for d in all_dates]
    gl_model  = [gl_map.get(d, (0, 0))[0]  for d in all_dates]
    gl_aviso  = [gl_map.get(d, (0, 0))[1]  for d in all_dates]

    has_ref     = bool(counts_ref)
    has_gliders = bool(counts_gliders)
    n_groups    = sum([has_ref, has_gliders, 1])  # +1 for AVISO
    width       = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(all_dates) * 0.6 + 2), 5))

    slot = 0
    if has_ref:
        off = (slot - (n_groups - 1) / 2) * width
        ax.bar(x + off, ref_model, width, label=ref_label, color="steelblue")
        slot += 1
    if has_gliders:
        off = (slot - (n_groups - 1) / 2) * width
        ax.bar(x + off, gl_model, width, label=gliders_label, color="darkred")
        slot += 1
    # AVISO — use REF aviso if available, else GLIDERS
    aviso_vals = ref_aviso if has_ref else gl_aviso
    aviso_label = f"AVISO (total={aviso_total})" if aviso_total is not None else "AVISO"
    off = (slot - (n_groups - 1) / 2) * width
    ax.bar(x + off, aviso_vals, width, label=aviso_label, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(date_labels, rotation=45, ha="right")
    ax.set_xlabel("Forecast start date", fontsize=11)
    ax.set_ylabel("Number of detachments", fontsize=11)
    ax.set_title("LCE detachment count per forecast", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "detachment_count_per_forecast.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def _sanitize_label_for_filename(label: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_")
    return out or "simulation"


def _style_for_multi_sim_label(label: str) -> Tuple[str, str]:
    """Return (color, linestyle) for multi-simulation timeseries labels."""
    name = str(label).lower()

    colors = {
        "tracks": "#1f77b4",
        "swot": "#d62728",
        "swot+tracks": "#2ca02c",
    }
    linestyles = {
        "9d": "-",
        "5d": "--",
        "3d": ":",
    }

    if "swot+tracks" in name or "swottracks" in name:
        color = colors["swot+tracks"]
    elif "tracks" in name and "swot+" not in name and "swottracks" not in name:
        color = colors["tracks"]
    else:
        color = colors["swot"]

    if "9d" in name:
        linestyle = linestyles["9d"]
    elif "5d" in name:
        linestyle = linestyles["5d"]
    else:
        linestyle = linestyles["3d"]

    return color, linestyle


def plot_timeseries_multi_simulation(
    simulation_results: Dict[str, List[Dict]],
    output_dir: str,
    max_lead_days: Optional[int] = MAX_LEAD_DAYS,
) -> None:
    """Plot MHD lead-time timeseries for multiple simulations (one mean line per simulation)."""
    if not simulation_results:
        print("No simulation results to plot.")
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    any_plotted = False
    for idx, (label, results) in enumerate(simulation_results.items()):
        valid = [r for r in results if r.get("success") and r.get("date")]
        if not valid:
            continue
        color, linestyle = _style_for_multi_sim_label(label)
        by_fs = defaultdict(list)
        for r in valid:
            fs = r.get("forecast_start")
            if fs is not None:
                by_fs[fs].append(r)
        lead_to_vals = defaultdict(list)
        # Faint per-forecast lines
        for fs, group in sorted(by_fs.items()):
            leads, mhds = [], []
            for r in group:
                lead = r.get("lead_time_days")
                if lead is None:
                    date_obj = datetime.strptime(r["date"], "%Y%m%d")
                    lead = (date_obj - fs).days
                if lead >= 0 and (max_lead_days is None or lead <= max_lead_days):
                    mhd_km = r["mhd"] * 111.0
                    leads.append(lead)
                    mhds.append(mhd_km)
                    lead_to_vals[lead].append(mhd_km)
            if leads:
                s = sorted(zip(leads, mhds))
                ax.plot(
                    [x for x, _ in s],
                    [y for _, y in s],
                    linestyle,
                    linewidth=1.2,
                    color=color,
                    alpha=0.2,
                )
        if lead_to_vals:
            lead_sorted = sorted(lead_to_vals.keys())
            mean_mhd = [np.mean(lead_to_vals[lt]) for lt in lead_sorted]
            ax.plot(
                lead_sorted,
                mean_mhd,
                linestyle,
                linewidth=2.5,
                color=color,
                label=f"Mean ({label})",
            )
            any_plotted = True
    ax.set_xlabel("Lead Time (days)", fontsize=11)
    if max_lead_days is None:
        observed = []
        for results in simulation_results.values():
            for r in results:
                if not r.get("success") or r.get("date") is None:
                    continue
                fs = r.get("forecast_start")
                if fs is None:
                    continue
                lead = r.get("lead_time_days")
                if lead is None:
                    date_obj = datetime.strptime(r["date"], "%Y%m%d")
                    lead = (date_obj - fs).days
                if lead >= 0:
                    observed.append(lead)
        ax.set_xlim(0, max(observed) if observed else MAX_LEAD_DAYS)
    else:
        ax.set_xlim(0, max_lead_days)
    ax.set_ylabel("Modified Hausdorff Distance (km)", fontsize=11)
    ax.set_title("MHD time series (all simulations, all forecasts)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if any_plotted:
        ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = os.path.join(output_dir, "mhd_timeseries_all_simulations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def plot_mean_std_multi_simulation(
    simulation_results: Dict[str, List[Dict]],
    output_dir: str,
    max_lead_days: Optional[int] = MAX_LEAD_DAYS,
) -> None:
    """Plot mean ± std MHD vs lead time for multiple simulations on one figure."""
    if not simulation_results:
        print("No simulation results to plot.")
        return
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("tab10")
    any_plotted = False
    for idx, (label, results) in enumerate(simulation_results.items()):
        lead_data = defaultdict(list)
        for r in results:
            if not r.get("success") or r.get("date") is None:
                continue
            fs = r.get("forecast_start")
            if fs is None:
                continue
            lead = r.get("lead_time_days")
            if lead is None:
                date_obj = datetime.strptime(r["date"], "%Y%m%d")
                lead = (date_obj - fs).days
            if lead >= 0 and (max_lead_days is None or lead <= max_lead_days):
                lead_data[lead].append(r["mhd"] * 111.0)
        if not lead_data:
            continue
        color = cmap(idx % 10)
        lead_sorted = sorted(lead_data.keys())
        mean_vals = np.array([np.mean(lead_data[lt]) for lt in lead_sorted])
        std_vals = np.array([np.std(lead_data[lt]) for lt in lead_sorted])
        ax.fill_between(lead_sorted, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.14)
        ax.plot(lead_sorted, mean_vals, "-", color=color, linewidth=2.2, label=label)
        any_plotted = True
    ax.set_xlabel("Lead Time (days)", fontsize=12)
    if max_lead_days is None:
        observed = []
        for results in simulation_results.values():
            for r in results:
                if not r.get("success") or r.get("date") is None:
                    continue
                fs = r.get("forecast_start")
                if fs is None:
                    continue
                lead = r.get("lead_time_days")
                if lead is None:
                    date_obj = datetime.strptime(r["date"], "%Y%m%d")
                    lead = (date_obj - fs).days
                if lead >= 0:
                    observed.append(lead)
        ax.set_xlim(0, max(observed) if observed else MAX_LEAD_DAYS)
    else:
        ax.set_xlim(0, max_lead_days)
    ax.set_ylabel("Modified Hausdorff Distance (km)", fontsize=12)
    ax.set_title("MHD: AVISO vs multiple simulations (mean ± std)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 140)
    ax.grid(True, alpha=0.3)
    if any_plotted:
        ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    path = os.path.join(output_dir, "plot_mhd_timeseries_mean_std_multi.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def plot_timing_distribution_multi_simulation(
    timing_by_sim: Dict[str, List[int]],
    output_dir: str,
) -> None:
    """Write one timing histogram per simulation."""
    if not timing_by_sim:
        print("No timing data for multi-simulation histogram.")
        return
    cmap = plt.get_cmap("tab10")
    for idx, (label, divergences) in enumerate(timing_by_sim.items()):
        if not divergences:
            print(f"No divergence values for timing distribution. Skip {label}.")
            continue
        safe = _sanitize_label_for_filename(label)
        color = cmap(idx % 10)
        _plot_one_timing_histogram(
            divergences,
            output_dir,
            title=f"Distribution of first LCE detachment timing: {label} vs AVISO (all forecasts)",
            xlabel=f"Divergence (days): {label} first LCE day − AVISO first LCE day",
            filename=f"histogram_first_lce_divergence_aviso_{safe}.png",
            color=color,
        )


def load_mhd_from_netcdf(path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load REF and GLIDERS MHD results from mhd_OSEs.nc. Returns (results_ref, results_gliders) with keys date, mhd (deg), forecast_start, lead_time_days (if in file), success."""
    if nc is None or not os.path.isfile(path):
        return [], []
    ref_list = []
    gliders_list = []
    lead_fill = -999
    try:
        with nc.Dataset(path, "r") as ds:
            if "n_ref" in ds.dimensions:
                n_ref = len(ds.dimensions["n_ref"])
                date_var = ds.variables["ref_date"]
                mhd_km = ds.variables["ref_mhd_km"][:]
                mhd_lc_km = ds.variables["ref_mhd_lc_only_km"][:] if "ref_mhd_lc_only_km" in ds.variables else None
                fs_var = ds.variables["ref_forecast_start"]
                lead_var = ds.variables["ref_lead_time_days"][:] if "ref_lead_time_days" in ds.variables else None
                for i in range(n_ref):
                    date_str = (date_var[i, :].tobytes().decode("ascii", errors="ignore") if hasattr(date_var[i, :], "tobytes") else "".join(str(c) for c in date_var[i, :])).strip()
                    if not date_str:
                        continue
                    mhd_deg = float(mhd_km[i]) / 111.0
                    mhd_lc_deg = float(mhd_lc_km[i]) / 111.0 if mhd_lc_km is not None else np.nan
                    fs_str = (fs_var[i, :].tobytes().decode("ascii", errors="ignore") if hasattr(fs_var[i, :], "tobytes") else "".join(str(c) for c in fs_var[i, :])).strip()
                    fs_dt = datetime.strptime(fs_str[:10], "%Y-%m-%d") if (fs_str and len(fs_str) >= 10 and fs_str[:10].replace("-", "").isdigit()) else None
                    lead_val = int(lead_var[i]) if lead_var is not None and i < len(lead_var) else None
                    if lead_val is not None and (lead_val == lead_fill or lead_val < 0 or lead_val > MAX_LEAD_DAYS):
                        lead_val = None
                    lead_days = lead_val
                    if fs_dt is None and lead_days is not None:
                        date_obj = datetime.strptime(date_str, "%Y%m%d")
                        fs_dt = date_obj - timedelta(days=lead_days)
                    rec = {"date": date_str, "mhd": mhd_deg, "mhd_lc_only": mhd_lc_deg, "forecast_start": fs_dt, "success": True}
                    if lead_days is not None:
                        rec["lead_time_days"] = lead_days
                    ref_list.append(rec)
            if "n_gliders" in ds.dimensions:
                n_gl = len(ds.dimensions["n_gliders"])
                date_var = ds.variables["gliders_date"]
                mhd_km = ds.variables["gliders_mhd_km"][:]
                mhd_lc_km = ds.variables["gliders_mhd_lc_only_km"][:] if "gliders_mhd_lc_only_km" in ds.variables else None
                fs_var = ds.variables["gliders_forecast_start"]
                lead_var = ds.variables["gliders_lead_time_days"][:] if "gliders_lead_time_days" in ds.variables else None
                for i in range(n_gl):
                    date_str = (date_var[i, :].tobytes().decode("ascii", errors="ignore") if hasattr(date_var[i, :], "tobytes") else "".join(str(c) for c in date_var[i, :])).strip()
                    if not date_str:
                        continue
                    mhd_deg = float(mhd_km[i]) / 111.0
                    mhd_lc_deg = float(mhd_lc_km[i]) / 111.0 if mhd_lc_km is not None else np.nan
                    fs_str = (fs_var[i, :].tobytes().decode("ascii", errors="ignore") if hasattr(fs_var[i, :], "tobytes") else "".join(str(c) for c in fs_var[i, :])).strip()
                    fs_dt = datetime.strptime(fs_str[:10], "%Y-%m-%d") if (fs_str and len(fs_str) >= 10 and fs_str[:10].replace("-", "").isdigit()) else None
                    lead_val = int(lead_var[i]) if lead_var is not None and i < len(lead_var) else None
                    if lead_val is not None and (lead_val == lead_fill or lead_val < 0 or lead_val > MAX_LEAD_DAYS):
                        lead_val = None
                    lead_days = lead_val
                    if fs_dt is None and lead_days is not None:
                        date_obj = datetime.strptime(date_str, "%Y%m%d")
                        fs_dt = date_obj - timedelta(days=lead_days)
                    rec = {"date": date_str, "mhd": mhd_deg, "mhd_lc_only": mhd_lc_deg, "forecast_start": fs_dt, "success": True}
                    if lead_days is not None:
                        rec["lead_time_days"] = lead_days
                    gliders_list.append(rec)
    except Exception as e:
        print(f"Warning: could not load {path}: {e}")
        return [], []
    return ref_list, gliders_list


def save_mhd_to_netcdf(
    results_ref: List[Dict],
    results_gliders: List[Dict],
    output_dir: str,
    filename: str = "mhd_OSEs.nc",
) -> None:
    """Save REF and GLIDERS MHD results to a single NetCDF file. Skips write if file already exists."""
    if nc is None:
        print("Warning: netCDF4 not installed. Skip writing mhd_OSEs.nc")
        return
    path = os.path.join(output_dir, filename)
    if os.path.isfile(path):
        print(f"{filename} exists, skipping write (MHD can be read from it).")
        return
    ref_ok = [r for r in results_ref if r.get("success") and r.get("date") is not None and np.isfinite(r.get("mhd", np.nan))]
    gliders_ok = [r for r in results_gliders if r.get("success") and r.get("date") is not None and np.isfinite(r.get("mhd", np.nan))]
    if not ref_ok and not gliders_ok:
        return
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncattr("title", "MHD OSEs: REF and GLIDERS results (date, forecast_start, lead_time_days, mhd_km)")
        fill_int = -999
        fill_float = np.nan
        n_ref = len(ref_ok)
        n_gliders = len(gliders_ok)
        # Use named dimensions for string lengths (netCDF4 requires dimension names, not numeric strings)
        ds.createDimension("date_strlen", 8)
        ds.createDimension("forecast_strlen", 10)
        if n_ref > 0:
            ds.createDimension("n_ref", n_ref)
            date_ref = ds.createVariable("ref_date", "S1", ("n_ref", "date_strlen"))
            date_ref.long_name = "date REF (YYYYMMDD)"
            fs_ref = ds.createVariable("ref_forecast_start", "S1", ("n_ref", "forecast_strlen"))
            fs_ref.long_name = "forecast start REF (YYYY-MM-DD)"
            lead_ref = ds.createVariable("ref_lead_time_days", "i4", "n_ref", fill_value=fill_int)
            lead_ref.long_name = "lead time in days; -999 if unknown"
            mhd_ref = ds.createVariable("ref_mhd_km", "f4", "n_ref")
            mhd_ref.long_name = "modified Hausdorff distance (km)"
            mhd_ref_lc = ds.createVariable("ref_mhd_lc_only_km", "f4", "n_ref")
            mhd_ref_lc.long_name = "LC-only modified Hausdorff distance (km, no LCE)"
            for i, r in enumerate(ref_ok):
                date_ref[i, :] = list(r["date"].ljust(8)[:8])
                fs = r.get("forecast_start")
                if fs is not None:
                    fs_str = fs.strftime("%Y-%m-%d") if hasattr(fs, "strftime") else str(fs)[:10]
                    fs_ref[i, :] = list(fs_str.ljust(10)[:10])
                    date_obj = datetime.strptime(r["date"], "%Y%m%d")
                    lead_ref[i] = (date_obj - fs).days if hasattr(fs, "strftime") else (date_obj - datetime.strptime(fs_str[:10], "%Y-%m-%d")).days
                else:
                    fs_ref[i, :] = list("          ")
                    lead_ref[i] = fill_int
                mhd_ref[i] = float(r["mhd"] * 111.0)
                mhd_ref_lc[i] = float(r.get("mhd_lc_only", np.nan) * 111.0)
        if n_gliders > 0:
            ds.createDimension("n_gliders", n_gliders)
            date_gl = ds.createVariable("gliders_date", "S1", ("n_gliders", "date_strlen"))
            date_gl.long_name = "date GLIDERS (YYYYMMDD)"
            fs_gl = ds.createVariable("gliders_forecast_start", "S1", ("n_gliders", "forecast_strlen"))
            fs_gl.long_name = "forecast start GLIDERS (YYYY-MM-DD)"
            lead_gl = ds.createVariable("gliders_lead_time_days", "i4", "n_gliders", fill_value=fill_int)
            lead_gl.long_name = "lead time in days; -999 if unknown"
            mhd_gl = ds.createVariable("gliders_mhd_km", "f4", "n_gliders")
            mhd_gl.long_name = "modified Hausdorff distance (km)"
            mhd_gl_lc = ds.createVariable("gliders_mhd_lc_only_km", "f4", "n_gliders")
            mhd_gl_lc.long_name = "LC-only modified Hausdorff distance (km, no LCE)"
            for i, r in enumerate(gliders_ok):
                date_gl[i, :] = list(r["date"].ljust(8)[:8])
                fs = r.get("forecast_start")
                if fs is not None:
                    fs_str = fs.strftime("%Y-%m-%d") if hasattr(fs, "strftime") else str(fs)[:10]
                    fs_gl[i, :] = list(fs_str.ljust(10)[:10])
                    date_obj = datetime.strptime(r["date"], "%Y%m%d")
                    lead_gl[i] = (date_obj - fs).days if hasattr(fs, "strftime") else (date_obj - datetime.strptime(fs_str[:10], "%Y-%m-%d")).days
                else:
                    fs_gl[i, :] = list("          ")
                    lead_gl[i] = fill_int
                mhd_gl[i] = float(r["mhd"] * 111.0)
                mhd_gl_lc[i] = float(r.get("mhd_lc_only", np.nan) * 111.0)
    print(f"Wrote {path} (REF: {n_ref}, GLIDERS: {n_gliders})")


def save_lce_timing_to_netcdf(
    timing_ref: List,
    timing_gliders: List,
    output_dir: str,
    filename: str = "lce_timing_OSEs.nc",
) -> None:
    """Save first-LCE timing (forecast_start, divergence_days[, aviso_offset_days]) to NetCDF."""
    if nc is None:
        return
    path = os.path.join(output_dir, filename)
    # If file already exists, keep it (timing is meant to be read, not constantly regenerated)
    if os.path.isfile(path):
        print(f"{filename} exists, skipping write (timing can be read from it).")
        return
    if not timing_ref and not timing_gliders:
        return
    _FILL_INT = -9999
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncattr("title", "LCE timing: forecast_start (YYYY-MM-DD), divergence_days (model first LCE day − AVISO first LCE day)")
        ds.createDimension("forecast_strlen", 10)
        n_ref = len(timing_ref)
        n_gl = len(timing_gliders)
        if n_ref > 0:
            ds.createDimension("n_ref", n_ref)
            fs_ref = ds.createVariable("ref_forecast_start", "S1", ("n_ref", "forecast_strlen"))
            fs_ref.long_name = "forecast start REF (YYYY-MM-DD)"
            div_ref = ds.createVariable("ref_divergence_days", "i4", "n_ref")
            div_ref.long_name = "divergence (days): REF first LCE day − AVISO first LCE day"
            aviso_off_ref = ds.createVariable("ref_aviso_offset_days", "i4", "n_ref")
            aviso_off_ref.long_name = "AVISO first LCE lead day within this forecast window"
            aviso_off_ref.missing_value = _FILL_INT
            for i, entry in enumerate(timing_ref):
                fs_dt, d = entry[0], entry[1]
                aviso_off = entry[2] if len(entry) > 2 else None
                fs_ref[i, :] = list(fs_dt.strftime("%Y-%m-%d").ljust(10)[:10])
                div_ref[i] = int(d)
                aviso_off_ref[i] = int(aviso_off) if aviso_off is not None else _FILL_INT
        if n_gl > 0:
            ds.createDimension("n_gliders", n_gl)
            fs_gl = ds.createVariable("gliders_forecast_start", "S1", ("n_gliders", "forecast_strlen"))
            fs_gl.long_name = "forecast start GLIDERS (YYYY-MM-DD)"
            div_gl = ds.createVariable("gliders_divergence_days", "i4", "n_gliders")
            div_gl.long_name = "divergence (days): GLIDERS first LCE day − AVISO first LCE day"
            aviso_off_gl = ds.createVariable("gliders_aviso_offset_days", "i4", "n_gliders")
            aviso_off_gl.long_name = "AVISO first LCE lead day within this forecast window"
            aviso_off_gl.missing_value = _FILL_INT
            for i, entry in enumerate(timing_gliders):
                fs_dt, d = entry[0], entry[1]
                aviso_off = entry[2] if len(entry) > 2 else None
                fs_gl[i, :] = list(fs_dt.strftime("%Y-%m-%d").ljust(10)[:10])
                div_gl[i] = int(d)
                aviso_off_gl[i] = int(aviso_off) if aviso_off is not None else _FILL_INT
    print(f"Wrote {path} (REF: {n_ref}, GLIDERS: {n_gl} timing values)")


def load_lce_timing_from_netcdf(
    path: str,
) -> Tuple[List, List, None]:
    """Load (forecast_start, divergence_days, aviso_offset_days) 3-tuples for REF and GLIDERS."""
    timing_ref: List = []
    timing_gliders: List = []
    _FILL_INT = -9999
    if nc is None or not os.path.isfile(path):
        return timing_ref, timing_gliders, None
    try:
        with nc.Dataset(path, "r") as ds:
            if "n_ref" in ds.dimensions and "ref_forecast_start" in ds.variables and "ref_divergence_days" in ds.variables:
                n_ref = len(ds.dimensions["n_ref"])
                fs_ref = ds.variables["ref_forecast_start"]
                div_ref = ds.variables["ref_divergence_days"][:]
                aviso_off_ref = ds.variables["ref_aviso_offset_days"][:] if "ref_aviso_offset_days" in ds.variables else None
                for i in range(n_ref):
                    fs_str = (
                        fs_ref[i, :].tobytes().decode("ascii", errors="ignore")
                        if hasattr(fs_ref[i, :], "tobytes")
                        else "".join(str(c) for c in fs_ref[i, :])
                    ).strip()
                    if not fs_str:
                        continue
                    try:
                        fs_dt = datetime.strptime(fs_str[:10], "%Y-%m-%d")
                    except ValueError:
                        continue
                    aviso_off = int(aviso_off_ref[i]) if aviso_off_ref is not None and int(aviso_off_ref[i]) != _FILL_INT else None
                    timing_ref.append((fs_dt, int(div_ref[i]), aviso_off))
            if "n_gliders" in ds.dimensions and "gliders_forecast_start" in ds.variables and "gliders_divergence_days" in ds.variables:
                n_gl = len(ds.dimensions["n_gliders"])
                fs_gl = ds.variables["gliders_forecast_start"]
                div_gl = ds.variables["gliders_divergence_days"][:]
                aviso_off_gl = ds.variables["gliders_aviso_offset_days"][:] if "gliders_aviso_offset_days" in ds.variables else None
                for i in range(n_gl):
                    fs_str = (
                        fs_gl[i, :].tobytes().decode("ascii", errors="ignore")
                        if hasattr(fs_gl[i, :], "tobytes")
                        else "".join(str(c) for c in fs_gl[i, :])
                    ).strip()
                    if not fs_str:
                        continue
                    try:
                        fs_dt = datetime.strptime(fs_str[:10], "%Y-%m-%d")
                    except ValueError:
                        continue
                    aviso_off = int(aviso_off_gl[i]) if aviso_off_gl is not None and int(aviso_off_gl[i]) != _FILL_INT else None
                    timing_gliders.append((fs_dt, int(div_gl[i]), aviso_off))
    except Exception as e:
        print(f"Warning: could not load timing from {path}: {e}")
        return [], [], None
    return timing_ref, timing_gliders, None


def save_separation_timing_to_netcdf(
    aviso_sep_date: datetime,
    confirmed_ref: List[Tuple[int, datetime]],
    uncertain_ref: List[Tuple[int, datetime]],
    confirmed_gliders: List[Tuple[int, datetime]],
    uncertain_gliders: List[Tuple[int, datetime]],
    output_dir: str,
    filename: str = "lce_timing_OSEs.nc",
) -> None:
    """Append final LC separation timing to lce_timing_OSEs.nc (creates file if absent)."""
    if nc is None:
        return
    path = os.path.join(output_dir, filename)
    mode = "a" if os.path.isfile(path) else "w"
    all_ref = [(d, fs, True) for d, fs in confirmed_ref] + [(d, fs, False) for d, fs in uncertain_ref]
    all_gl  = [(d, fs, True) for d, fs in confirmed_gliders] + [(d, fs, False) for d, fs in uncertain_gliders]
    try:
        with nc.Dataset(path, mode, format="NETCDF4") as ds:
            ds.setncattr("sep_aviso_date", aviso_sep_date.strftime("%Y-%m-%d"))
            if "forecast_strlen" not in ds.dimensions:
                ds.createDimension("forecast_strlen", 10)
            if all_ref:
                n = len(all_ref)
                if "n_sep_ref" not in ds.dimensions:
                    ds.createDimension("n_sep_ref", n)
                fs_v = ds.createVariable("sep_ref_forecast_start", "S1", ("n_sep_ref", "forecast_strlen"))
                fs_v.long_name = "forecast start REF separation (YYYY-MM-DD)"
                div_v = ds.createVariable("sep_ref_divergence_days", "i4", "n_sep_ref")
                div_v.long_name = "divergence (days): REF final separation − AVISO final separation"
                conf_v = ds.createVariable("sep_ref_confirmed", "i1", "n_sep_ref")
                conf_v.long_name = "1=confirmed (>=30 days), 0=uncertain"
                for i, (d, fs, is_conf) in enumerate(all_ref):
                    fs_v[i, :] = list(fs.strftime("%Y-%m-%d").ljust(10)[:10])
                    div_v[i] = int(d)
                    conf_v[i] = int(is_conf)
            if all_gl:
                n = len(all_gl)
                if "n_sep_gliders" not in ds.dimensions:
                    ds.createDimension("n_sep_gliders", n)
                fs_v = ds.createVariable("sep_gliders_forecast_start", "S1", ("n_sep_gliders", "forecast_strlen"))
                fs_v.long_name = "forecast start GLIDERS separation (YYYY-MM-DD)"
                div_v = ds.createVariable("sep_gliders_divergence_days", "i4", "n_sep_gliders")
                div_v.long_name = "divergence (days): GLIDERS final separation − AVISO final separation"
                conf_v = ds.createVariable("sep_gliders_confirmed", "i1", "n_sep_gliders")
                conf_v.long_name = "1=confirmed (>=30 days), 0=uncertain"
                for i, (d, fs, is_conf) in enumerate(all_gl):
                    fs_v[i, :] = list(fs.strftime("%Y-%m-%d").ljust(10)[:10])
                    div_v[i] = int(d)
                    conf_v[i] = int(is_conf)
        print(f"Wrote separation timing to {path} (REF: {len(all_ref)}, GLIDERS: {len(all_gl)})")
    except Exception as e:
        print(f"Warning: could not save separation timing to {path}: {e}")


def load_separation_timing_from_netcdf(
    path: str,
) -> Tuple[Optional[datetime], List[Tuple[int, datetime, bool]], List[Tuple[int, datetime, bool]]]:
    """Load separation timing from lce_timing_OSEs.nc. Returns (aviso_sep_date, ref_items, gliders_items).
    Each item is (divergence_days, forecast_start, is_confirmed)."""
    if nc is None or not os.path.isfile(path):
        return None, [], []
    try:
        with nc.Dataset(path, "r") as ds:
            aviso_str = ds.getncattr("sep_aviso_date") if "sep_aviso_date" in ds.ncattrs() else None
            if aviso_str is None:
                return None, [], []
            aviso_sep_date = datetime.strptime(aviso_str[:10], "%Y-%m-%d")

            def _read_sep(prefix, dim):
                items = []
                if dim not in ds.dimensions:
                    return items
                n = len(ds.dimensions[dim])
                fs_v = ds.variables.get(f"{prefix}_forecast_start")
                div_v = ds.variables.get(f"{prefix}_divergence_days")
                conf_v = ds.variables.get(f"{prefix}_confirmed")
                if fs_v is None or div_v is None:
                    return items
                div_arr = div_v[:]
                conf_arr = conf_v[:] if conf_v is not None else [1] * n
                for i in range(n):
                    fs_str = (
                        fs_v[i, :].tobytes().decode("ascii", errors="ignore")
                        if hasattr(fs_v[i, :], "tobytes")
                        else "".join(str(c) for c in fs_v[i, :])
                    ).strip()
                    if not fs_str:
                        continue
                    try:
                        fs_dt = datetime.strptime(fs_str[:10], "%Y-%m-%d")
                    except ValueError:
                        continue
                    items.append((int(div_arr[i]), fs_dt, bool(conf_arr[i])))
                return items

            ref_items = _read_sep("sep_ref", "n_sep_ref")
            gl_items  = _read_sep("sep_gliders", "n_sep_gliders")
            return aviso_sep_date, ref_items, gl_items
    except Exception as e:
        print(f"Warning: could not load separation timing from {path}: {e}")
        return None, [], []


def save_detachment_counts_to_netcdf(
    counts_ref: List[Tuple[datetime, int, int]],
    counts_gliders: List[Tuple[datetime, int, int]],
    output_dir: str,
    filename: str = "lce_timing_OSEs.nc",
    aviso_total: Optional[int] = None,
) -> None:
    """Append LCE detachment counts to lce_timing_OSEs.nc (creates file if absent).
    counts_*: list of (forecast_start, model_count, aviso_count_per_forecast).
    aviso_total: total AVISO detachments over the full time series (global attribute)."""
    if nc is None:
        return
    path = os.path.join(output_dir, filename)
    mode = "a" if os.path.isfile(path) else "w"
    try:
        with nc.Dataset(path, mode, format="NETCDF4") as ds:
            if aviso_total is not None:
                ds.setncattr("det_aviso_total", int(aviso_total))
            if "forecast_strlen" not in ds.dimensions:
                ds.createDimension("forecast_strlen", 10)
            if counts_ref:
                n = len(counts_ref)
                if "n_det_ref" not in ds.dimensions:
                    ds.createDimension("n_det_ref", n)
                if "det_ref_forecast_start" not in ds.variables:
                    v = ds.createVariable("det_ref_forecast_start", "S1", ("n_det_ref", "forecast_strlen"))
                    v.long_name = "forecast start REF detachment count (YYYY-MM-DD)"
                if "det_ref_model_count" not in ds.variables:
                    v = ds.createVariable("det_ref_model_count", "i4", "n_det_ref")
                    v.long_name = "REF model detachment count per forecast"
                if "det_ref_aviso_count" not in ds.variables:
                    v = ds.createVariable("det_ref_aviso_count", "i4", "n_det_ref")
                    v.long_name = "AVISO detachment count per forecast window (REF)"
                for i, (fs_dt, mc, ac) in enumerate(counts_ref):
                    ds.variables["det_ref_forecast_start"][i, :] = list(fs_dt.strftime("%Y-%m-%d").ljust(10)[:10])
                    ds.variables["det_ref_model_count"][i] = int(mc)
                    ds.variables["det_ref_aviso_count"][i] = int(ac)
            if counts_gliders:
                n = len(counts_gliders)
                if "n_det_gliders" not in ds.dimensions:
                    ds.createDimension("n_det_gliders", n)
                if "det_gliders_forecast_start" not in ds.variables:
                    v = ds.createVariable("det_gliders_forecast_start", "S1", ("n_det_gliders", "forecast_strlen"))
                    v.long_name = "forecast start GLIDERS detachment count (YYYY-MM-DD)"
                if "det_gliders_model_count" not in ds.variables:
                    v = ds.createVariable("det_gliders_model_count", "i4", "n_det_gliders")
                    v.long_name = "GLIDERS model detachment count per forecast"
                if "det_gliders_aviso_count" not in ds.variables:
                    v = ds.createVariable("det_gliders_aviso_count", "i4", "n_det_gliders")
                    v.long_name = "AVISO detachment count per forecast window (GLIDERS)"
                for i, (fs_dt, mc, ac) in enumerate(counts_gliders):
                    ds.variables["det_gliders_forecast_start"][i, :] = list(fs_dt.strftime("%Y-%m-%d").ljust(10)[:10])
                    ds.variables["det_gliders_model_count"][i] = int(mc)
                    ds.variables["det_gliders_aviso_count"][i] = int(ac)
        print(f"Wrote detachment counts to {path} (REF: {len(counts_ref)}, GLIDERS: {len(counts_gliders)}, AVISO total: {aviso_total})")
    except Exception as e:
        print(f"Warning: could not save detachment counts to {path}: {e}")


def load_detachment_counts_from_netcdf(
    path: str,
) -> Tuple[List[Tuple[datetime, int, int]], List[Tuple[datetime, int, int]], Optional[int]]:
    """Load detachment counts from lce_timing_OSEs.nc.
    Returns (counts_ref, counts_gliders, aviso_total) where counts_* are (forecast_start, model_count, aviso_count)."""
    counts_ref: List[Tuple[datetime, int, int]] = []
    counts_gliders: List[Tuple[datetime, int, int]] = []
    aviso_total: Optional[int] = None
    if nc is None or not os.path.isfile(path):
        return counts_ref, counts_gliders, aviso_total
    try:
        with nc.Dataset(path, "r") as ds:
            if "det_aviso_total" in ds.ncattrs():
                aviso_total = int(ds.getncattr("det_aviso_total"))

            def _read_det(prefix, dim):
                items = []
                if dim not in ds.dimensions:
                    return items
                n = len(ds.dimensions[dim])
                fs_v = ds.variables.get(f"{prefix}_forecast_start")
                mc_v = ds.variables.get(f"{prefix}_model_count")
                ac_v = ds.variables.get(f"{prefix}_aviso_count")
                if fs_v is None or mc_v is None:
                    return items
                mc_arr = mc_v[:]
                ac_arr = ac_v[:] if ac_v is not None else [0] * n
                for i in range(n):
                    fs_str = (
                        fs_v[i, :].tobytes().decode("ascii", errors="ignore")
                        if hasattr(fs_v[i, :], "tobytes")
                        else "".join(str(c) for c in fs_v[i, :])
                    ).strip()
                    if not fs_str:
                        continue
                    try:
                        fs_dt = datetime.strptime(fs_str[:10], "%Y-%m-%d")
                    except ValueError:
                        continue
                    items.append((fs_dt, int(mc_arr[i]), int(ac_arr[i])))
                return items

            counts_ref = _read_det("det_ref", "n_det_ref")
            counts_gliders = _read_det("det_gliders", "n_det_gliders")
    except Exception as e:
        print(f"Warning: could not load detachment counts from {path}: {e}")
        return [], [], None
    return counts_ref, counts_gliders, aviso_total


def save_forecast_summary_table(
    output_dir: str,
    ref_label: str,
    gliders_label: str,
    counts_ref: List[Tuple[datetime, int]],
    counts_gliders: List[Tuple[datetime, int]],
    aviso_total: Optional[int],
    no_sep_ref: List[datetime],
    no_sep_gliders: List[datetime],
    filename: str = "forecast_summary.txt",
) -> None:
    """Write a text table summarising forecasts with no model detachments and forecasts with no model separation."""
    path = os.path.join(output_dir, filename)
    lines = []
    lines.append("=" * 60)
    lines.append("FORECAST SUMMARY")
    lines.append("=" * 60)

    # --- Section 1: no model detachments ---
    aviso_str = f" (AVISO full-series total: {aviso_total})" if aviso_total is not None else ""
    lines.append(f"\n[1] Forecasts with NO model detachments{aviso_str}")
    lines.append("-" * 60)
    for label, counts in [(ref_label, counts_ref), (gliders_label, counts_gliders)]:
        no_det = [fs.strftime("%Y-%m-%d") for fs, mc, *_ in counts if mc == 0]
        if no_det:
            lines.append(f"  {label}:")
            for d in no_det:
                lines.append(f"    {d}")
        else:
            lines.append(f"  {label}: all forecasts had ≥1 detachment")

    # --- Section 2: no model final separation ---
    lines.append(f"\n[2] Forecasts with NO model final separation detected")
    lines.append("-" * 60)
    for label, no_sep in [(ref_label, no_sep_ref), (gliders_label, no_sep_gliders)]:
        if no_sep:
            lines.append(f"  {label}:")
            for fs in sorted(no_sep):
                lines.append(f"    {fs.strftime('%Y-%m-%d')}")
        else:
            lines.append(f"  {label}: all forecasts had a separation detected")

    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote forecast summary to {path}")


def extract_forecast_start_from_hycom_path(base_dir_ref: str) -> Optional[datetime]:
    """Parse forecast start from HYCOM path like .../Jun03-2025_REF/data -> 2025-06-03."""
    base = os.path.basename(os.path.dirname(base_dir_ref)).replace("_REF", "").replace("_GLIDERS", "")
    month_abbrev = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    for month_str, month_num in month_abbrev.items():
        if month_str in base:
            parts = base.split(month_str)
            if len(parts) == 2:
                day_year = parts[1].split("-")
                if len(day_year) == 2:
                    try:
                        day = int(day_year[0])
                        year = int(day_year[1])
                        return datetime(year, month_num, day)
                    except ValueError:
                        pass
    return None


def get_model_data_config_hycom_all() -> List[tuple]:
    """
    Discover all HYCOM forecasts under HYCOM_BASE_FORECAST_DIR (*2025_REF).
    Returns list of (files_ref, files_gliders, grid_ref, grid_gliders, aviso_dir, mdt_path, forecast_start_dt).
    """
    aviso_dir = AVISO_DIR
    mdt_path = MDT_PATH
    ref_dirs = sorted(glob.glob(os.path.join(HYCOM_BASE_FORECAST_DIR, "*2025_REF")))
    configs = []
    for ref_dir in ref_dirs:
        base_dir_ref = os.path.join(ref_dir, "data")
        forecast_start = extract_forecast_start_from_hycom_path(base_dir_ref)
        if forecast_start is None:
            continue
        grid_file_ref = HYCOM_GRID_FILE if HYCOM_GRID_FILE and os.path.isfile(HYCOM_GRID_FILE) else os.path.join(base_dir_ref, "grad", "regional.grid.a")
        if not os.path.isfile(grid_file_ref):
            continue
        pattern_ref = os.path.join(base_dir_ref, "tarm_125*", "*.a")
        files_ref = [f for f in sorted(glob.glob(pattern_ref)) if not f.endswith(".tar.gz") and f.endswith(".a")]
        if len(files_ref) == 0:
            continue
        gliders_dir = ref_dir.replace("_REF", "_GLIDERS")
        base_dir_gliders = os.path.join(gliders_dir, "data")
        grid_file_gliders = HYCOM_GRID_FILE if HYCOM_GRID_FILE and os.path.isfile(HYCOM_GRID_FILE) else os.path.join(base_dir_gliders, "grad", "regional.grid.a")
        pattern_gliders = os.path.join(base_dir_gliders, "tarm_125*", "*.a")
        files_gliders = [f for f in sorted(glob.glob(pattern_gliders)) if not f.endswith(".tar.gz") and f.endswith(".a")] if os.path.isdir(base_dir_gliders) else []
        if not os.path.isfile(grid_file_gliders):
            grid_file_gliders = grid_file_ref  # use REF grid for GLIDERS (same domain); keep files_gliders
        configs.append((files_ref, files_gliders, grid_file_ref, grid_file_gliders, aviso_dir, mdt_path, forecast_start))
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OSEs metrics: animation, time series, mean±std, timing distribution. Use --hycom or --no-hycom."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--hycom", action="store_true", help="Use HYCOM model data (paths in script)")
    mode.add_argument("--no-hycom", action="store_true", help="Use NetCDF model data (provide paths below)")
    mode.add_argument("--multi-simulation", action="store_true", dest="multi_simulation", help="Use multiple NetCDF simulation directories (compare many simulations vs AVISO).")
    parser.add_argument("--animate", action="store_true", help="Produce MHD contour animation (MP4)")
    parser.add_argument("--animate-all", action="store_true", dest="animate_all", help="With --animate: produce one animation per forecast/group (HYCOM forecasts or NetCDF subfolders).")
    parser.add_argument("--timeseries", action="store_true", help="Produce MHD time series plot (all forecasts, lead time vs MHD)")
    parser.add_argument("--timeseries-by-date", action="store_true", dest="timeseries_by_date", help="Produce MHD time series vs actual date (mean ± std across forecasts)")
    parser.add_argument("--no-lce-timeseries", action="store_true", dest="no_lce_timeseries", help="Plot LC-only MHD timeseries (no LCE detection); saves mhd_OSEs_lc_only.nc")
    parser.add_argument("--no-lce-timeseries-by-date", action="store_true", dest="no_lce_timeseries_by_date", help="Plot LC-only MHD timeseries by date (no LCE detection); saves mhd_OSEs_lc_only.nc")
    parser.add_argument("--no-lce-mean-std", action="store_true", dest="no_lce_mean_std", help="Plot LC-only MHD mean±std (no LCE detection); saves mhd_OSEs_lc_only.nc")
    parser.add_argument("--mean-std-by-date", action="store_true", dest="mean_std_by_date", help="Plot MHD mean±std vs actual date")
    parser.add_argument("--no-lce-mean-std-by-date", action="store_true", dest="no_lce_mean_std_by_date", help="Plot LC-only MHD mean±std vs actual date (no LCE detection); saves mhd_OSEs_lc_only.nc")
    parser.add_argument("--no-lon-span-filter", action="store_true", dest="no_lon_span_filter", help="Disable the min_lon_span=1.3° LCE filter (allow narrow LCEs)")
    parser.add_argument("--detachment-count", action="store_true", dest="detachment_count", help="Plot LCE detachment count per forecast (model vs AVISO grouped bar chart)")
    parser.add_argument("--separation-timing", action="store_true", dest="separation_timing", help="Plot final LC separation timing vs AVISO (days early/late) across all forecasts")
    parser.add_argument("--mean-std", action="store_true", help="Produce mean ± std MHD vs lead time (requires forecast start)")
    parser.add_argument("--timing-distribution", action="store_true", help="Produce histogram of first LCE detachment timing (requires forecast start)")
    parser.add_argument("--netcdf-dir", type=str, help="[--no-hycom] Directory containing NetCDF SSH files (REF run)")
    parser.add_argument("--netcdf-dir-gliders", type=str, help="[--no-hycom] Optional: second directory for GLIDERS run")
    parser.add_argument("--sim-dirs", nargs="+", default=None, help="[--multi-simulation] One or more simulation NetCDF directories.")
    parser.add_argument("--sim-labels", nargs="+", default=None, help="[--multi-simulation] Optional labels for --sim-dirs (same order/length).")
    parser.add_argument("--netcdf-pattern", type=str, default="*.nc", help="[--no-hycom] Glob pattern for files (default: *.nc)")
    parser.add_argument("--aviso-dir", type=str, help="[--no-hycom] AVISO gridded directory")
    parser.add_argument("--mdt", type=str, help="[--no-hycom] Path to MDT NetCDF file")
    parser.add_argument("--forecast-start", type=str, metavar="YYYY-MM-DD", help="[--no-hycom] Optional fallback forecast start date. If omitted, inferred from NetCDF parent folder name.")
    parser.add_argument("--lon-var", type=str, default="longitude", help="[--no-hycom] NetCDF longitude variable")
    parser.add_argument("--lat-var", type=str, default="latitude", help="[--no-hycom] NetCDF latitude variable")
    parser.add_argument("--ssh-var", type=str, default="ssh", help="[--no-hycom] NetCDF SSH variable")
    parser.add_argument("--ssh-scale", type=float, default=1.0, help="[--no-hycom] Scale SSH to meters (e.g. 0.01 if cm)")
    parser.add_argument("--model-label", type=str, default="no_GrASE_HYCOM", help="Legend label for REF run")
    parser.add_argument("--model-label-gliders", type=str, default="GrASE_HYCOM", help="Legend label for GLIDERS run")
    parser.add_argument("--grid-netcdf", type=str, default=None, help="[--no-hycom] NetCDF grid file with cell sizes (for area-weighted demean when --mercator)")
    parser.add_argument("--mercator", action="store_true", help="[--no-hycom] Use area-weighted demean (requires --grid-netcdf with dx/dy or pscx/pscy)")
    parser.add_argument("--grid-dx-var", type=str, default="pscx", help="[--no-hycom] Grid NetCDF variable for cell x-size (default: pscx)")
    parser.add_argument("--grid-dy-var", type=str, default="pscy", help="[--no-hycom] Grid NetCDF variable for cell y-size (default: pscy)")
    parser.add_argument("-o", "--output", type=str, default="animation_mhd_lc+lce.mp4", help="Output MP4 path (for --animate)")
    parser.add_argument("--max-forecasts", type=int, default=None, metavar="N", help="[--hycom] Use at most N forecasts (for testing). Default: all.")
    args = parser.parse_args()

    if not (args.animate or args.timeseries or args.timeseries_by_date or args.no_lce_timeseries or args.no_lce_timeseries_by_date or args.no_lce_mean_std or args.mean_std or args.mean_std_by_date or args.no_lce_mean_std_by_date or args.timing_distribution or args.detachment_count or args.separation_timing):
        parser.error("At least one of --animate, --timeseries, --mean-std, --timing-distribution is required.")
    lon_cutoff = -81.0
    lce_min_lon_span = 0.0 if args.no_lon_span_filter else 1.3
    lon_filter_suffix = "_no_lon_filter" if args.no_lon_span_filter else ""
    use_cutoff = True
    results_ref_all = None
    results_gliders_all = None
    timing_only_data_ref = None  # List of (forecast_start, divergence_days) when --timing-distribution (light path)
    timing_only_data_gliders = None
    detachment_count_data_ref = None     # List of (forecast_start, model_count, aviso_count) when --detachment-count
    detachment_count_data_gliders = None
    aviso_date_max_lat: Optional[Dict[datetime, float]] = None
    aviso_total_detachments_from_nc: Optional[int] = None
    animate_all_done = False  # True when --animate-all produced one MP4 per forecast

    if args.hycom:
        # ---------------------------------------------------------------------
        # HYCOM: animation uses one forecast only; timeseries/mean-std/timing use all forecasts
        # ---------------------------------------------------------------------
        def get_model_data_config_hycom_single():
            """Single forecast for animation (paths from HYCOM_SINGLE_REF_SUBDIR / HYCOM_SINGLE_GLIDERS_SUBDIR)."""
            base_dir_ref = os.path.join(HYCOM_BASE_FORECAST_DIR, HYCOM_SINGLE_REF_SUBDIR, "data")
            base_dir_gliders = os.path.join(HYCOM_BASE_FORECAST_DIR, HYCOM_SINGLE_GLIDERS_SUBDIR, "data")
            grid_file_ref = HYCOM_GRID_FILE if HYCOM_GRID_FILE and os.path.isfile(HYCOM_GRID_FILE) else os.path.join(base_dir_ref, "grad", "regional.grid.a")
            grid_file_gliders = HYCOM_GRID_FILE if HYCOM_GRID_FILE and os.path.isfile(HYCOM_GRID_FILE) else os.path.join(base_dir_gliders, "grad", "regional.grid.a")
            aviso_dir = AVISO_DIR
            mdt_path = MDT_PATH
            pattern_ref = os.path.join(base_dir_ref, "tarm_125*", "*.a")
            pattern_gliders = os.path.join(base_dir_gliders, "tarm_125*", "*.a")
            files_ref = [f for f in sorted(glob.glob(pattern_ref)) if not f.endswith(".tar.gz") and f.endswith(".a")]
            files_gliders = [f for f in sorted(glob.glob(pattern_gliders)) if not f.endswith(".tar.gz") and f.endswith(".a")]
            forecast_start = extract_forecast_start_from_hycom_path(base_dir_ref)
            return files_ref, files_gliders, grid_file_ref, grid_file_gliders, aviso_dir, mdt_path, forecast_start

        need_animation = args.animate
        need_rest = args.timeseries or args.timeseries_by_date or args.no_lce_timeseries or args.no_lce_timeseries_by_date or args.no_lce_mean_std or args.mean_std or args.mean_std_by_date or args.no_lce_mean_std_by_date or args.timing_distribution or args.detachment_count or args.separation_timing
        results_ref = []
        results_gliders = []
        forecast_start_dt = None

        ref_label = args.model_label
        gliders_label = args.model_label_gliders

        # If we only need timing/detachment/separation and an existing timing NetCDF is present, load and skip HYCOM.
        _light_only = not (args.timeseries or args.timeseries_by_date or args.mean_std or args.animate)
        _wants_nc = (args.timing_distribution or args.detachment_count or args.separation_timing) and _light_only
        if _wants_nc:
            timing_nc_path = os.path.join(OUTPUT_DIR, "lce_timing_OSEs.nc")
            if os.path.isfile(timing_nc_path):
                _all_loaded = True
                # --- timing distribution ---
                if args.timing_distribution:
                    loaded_ref, loaded_gliders, _ = load_lce_timing_from_netcdf(timing_nc_path)
                    if loaded_ref or loaded_gliders:
                        print(f"[Timing-only] Loaded from {timing_nc_path} (REF {len(loaded_ref)}, GLIDERS {len(loaded_gliders)})")
                        timing_only_data_ref = loaded_ref
                        timing_only_data_gliders = loaded_gliders
                    else:
                        print("[Timing-only] No timing data in nc. Will recompute.")
                        _all_loaded = False
                # --- detachment count ---
                if args.detachment_count and _all_loaded:
                    det_ref_nc, det_gl_nc, det_aviso_nc = load_detachment_counts_from_netcdf(timing_nc_path)
                    if det_ref_nc or det_gl_nc:
                        print(f"[Detachment] Loaded from {timing_nc_path} (REF {len(det_ref_nc)}, GLIDERS {len(det_gl_nc)}, AVISO total={det_aviso_nc})")
                        detachment_count_data_ref = det_ref_nc
                        detachment_count_data_gliders = det_gl_nc
                        aviso_total_detachments_from_nc = det_aviso_nc
                    else:
                        print("[Detachment] No detachment data in nc. Will recompute.")
                        _all_loaded = False
                # --- separation timing ---
                if args.separation_timing and _all_loaded:
                    aviso_sep_date_nc, ref_items_nc, gl_items_nc = load_separation_timing_from_netcdf(timing_nc_path)
                    if aviso_sep_date_nc is not None and (ref_items_nc or gl_items_nc):
                        print(f"[Separation] Loaded from {timing_nc_path} (AVISO sep {aviso_sep_date_nc.strftime('%Y-%m-%d')}, REF {len(ref_items_nc)}, GLIDERS {len(gl_items_nc)})")
                        # reconstruct confirmed/uncertain lists: items are (divergence_days, forecast_start, is_confirmed)
                        conf_ref_nc = [(d, fs) for d, fs, c in ref_items_nc if c]
                        unc_ref_nc  = [(d, fs) for d, fs, c in ref_items_nc if not c]
                        conf_gl_nc  = [(d, fs) for d, fs, c in gl_items_nc  if c]
                        unc_gl_nc   = [(d, fs) for d, fs, c in gl_items_nc  if not c]
                        plot_separation_timing(conf_ref_nc, unc_ref_nc, conf_gl_nc, unc_gl_nc,
                                               OUTPUT_DIR, ref_label=ref_label, gliders_label=gliders_label,
                                               aviso_sep_date=aviso_sep_date_nc)
                        save_forecast_summary_table(
                            OUTPUT_DIR, ref_label, gliders_label,
                            detachment_count_data_ref or [],
                            detachment_count_data_gliders or [],
                            aviso_total_detachments_from_nc if args.detachment_count else None,
                            [], [],  # no_sep lists not available from nc
                        )
                    else:
                        print("[Separation] No separation data in nc. Will recompute.")
                        _all_loaded = False
                if _all_loaded:
                    need_rest = False

        if need_animation and getattr(args, "animate_all", False):
            # One animation per forecast
            configs_anim = get_model_data_config_hycom_all()
            if getattr(args, "max_forecasts", None) is not None:
                configs_anim = configs_anim[: args.max_forecasts]
                print(f"[Animation – all forecasts] Limiting to {len(configs_anim)} forecast(s) (--max-forecasts)")
            if not configs_anim:
                print("Error: No HYCOM forecasts found for --animate-all. Check HYCOM_BASE_FORECAST_DIR.")
                sys.exit(1)
            print(f"[Animation – all forecasts] Found {len(configs_anim)} forecasts")
            print("=" * 60)
            ref_label_anim = "HYCOM_REF"
            gliders_label_anim = "HYCOM_GLIDERS"
            base_out = args.output if os.path.isabs(args.output) else os.path.join(OUTPUT_DIR, args.output)
            base_stem, base_ext = os.path.splitext(base_out)
            for cfg_idx, (files_ref, files_gliders, grid_ref, grid_gliders, aviso_dir, mdt_path, fs_dt) in enumerate(configs_anim):
                print(f"  Forecast {cfg_idx + 1}/{len(configs_anim)}: {fs_dt.strftime('%Y-%m-%d')} ({len(files_ref)} REF, {len(files_gliders)} GLIDERS)")
                anim_ref = []
                anim_gliders = []
                for i, hycom_file in enumerate(files_ref, 1):
                    if i % 10 == 0 or i == 1:
                        print(f"    [{i}/{len(files_ref)}] REF {os.path.basename(hycom_file)}...")
                    result = process_hycom_file_for_animation(
                        hycom_file=hycom_file, grid_file=grid_ref,
                        aviso_dir=aviso_dir, mdt_path=mdt_path,
                        lon_cutoff=lon_cutoff, use_cutoff=use_cutoff,
                        lce_min_lon_span=lce_min_lon_span,
                    )
                    anim_ref.append(result)
                    sys.stdout.flush()
                for i, hycom_file in enumerate(files_gliders, 1):
                    if i % 10 == 0 or i == 1:
                        print(f"    [{i}/{len(files_gliders)}] GLIDERS {os.path.basename(hycom_file)}...")
                    result = process_hycom_file_for_animation(
                        hycom_file=hycom_file, grid_file=grid_gliders,
                        aviso_dir=aviso_dir, mdt_path=mdt_path,
                        lon_cutoff=lon_cutoff, use_cutoff=use_cutoff,
                        lce_min_lon_span=lce_min_lon_span,
                    )
                    anim_gliders.append(result)
                    sys.stdout.flush()
                out_anim = base_stem + "_" + fs_dt.strftime("%Y-%m-%d") + base_ext
                if any(r.get("success") and r.get("date") for r in anim_ref):
                    create_animation(
                        results_ref=anim_ref,
                        results_gliders=anim_gliders,
                        output_file=out_anim,
                        fps=2,
                        ref_label=ref_label_anim,
                        gliders_label=gliders_label_anim,
                    )
                else:
                    print(f"    Skip animation (no valid REF results for {fs_dt.strftime('%Y-%m-%d')})")
            animate_all_done = True
        elif need_animation:
            hycom_files_ref, hycom_files_gliders, grid_file_ref, grid_file_gliders, aviso_dir, mdt_path, forecast_start_dt = get_model_data_config_hycom_single()
            if len(hycom_files_ref) == 0:
                print("Error: No HYCOM files found for animation. Check paths in get_model_data_config_hycom_single().")
                sys.exit(1)
            print(f"[Animation – one forecast] Found {len(hycom_files_ref)} REF, {len(hycom_files_gliders)} GLIDERS files")
            print("=" * 60)
            for i, hycom_file in enumerate(hycom_files_ref, 1):
                if i % 10 == 0 or i == 1:
                    print(f"[{i}/{len(hycom_files_ref)}] Processing {os.path.basename(hycom_file)}...")
                result = process_hycom_file_for_animation(
                    hycom_file=hycom_file, grid_file=grid_file_ref,
                    aviso_dir=aviso_dir, mdt_path=mdt_path,
                    lon_cutoff=lon_cutoff, use_cutoff=use_cutoff,
                    lce_min_lon_span=lce_min_lon_span,
                )
                if forecast_start_dt is not None:
                    result["forecast_start"] = forecast_start_dt
                results_ref.append(result)
                if result["success"] and (i % 10 == 0 or i == 1):
                    m_val = result.get("mhd", np.nan)
                    if np.isfinite(m_val):
                        print(f"  Date: {result['date']}, MHD: ~{m_val * 111.0:.2f} km")
                    else:
                        print(f"  Date: {result['date']}, MHD: NaN (no east-of-90W overlap)")
                sys.stdout.flush()
            for i, hycom_file in enumerate(hycom_files_gliders, 1):
                if i % 10 == 0 or i == 1:
                    print(f"[{i}/{len(hycom_files_gliders)}] Processing {os.path.basename(hycom_file)}...")
                result = process_hycom_file_for_animation(
                    hycom_file=hycom_file, grid_file=grid_file_gliders,
                    aviso_dir=aviso_dir, mdt_path=mdt_path,
                    lon_cutoff=lon_cutoff, use_cutoff=use_cutoff,
                    lce_min_lon_span=lce_min_lon_span,
                )
                if forecast_start_dt is not None:
                    result["forecast_start"] = forecast_start_dt
                results_gliders.append(result)
                if result["success"] and (i % 10 == 0 or i == 1):
                    m_val = result.get("mhd", np.nan)
                    if np.isfinite(m_val):
                        print(f"  Date: {result['date']}, MHD: ~{m_val * 111.0:.2f} km")
                    else:
                        print(f"  Date: {result['date']}, MHD: NaN (no east-of-90W overlap)")
                sys.stdout.flush()

        need_full_processing = args.timeseries or args.timeseries_by_date or args.mean_std or args.mean_std_by_date
        if need_rest:
            configs = get_model_data_config_hycom_all()
            if getattr(args, "max_forecasts", None) is not None:
                configs = configs[: args.max_forecasts]
                print(f"[Test] Limiting to {len(configs)} forecast(s) (--max-forecasts {args.max_forecasts})")
            if not configs:
                print("Error: No HYCOM forecasts found. Check HYCOM_BASE_FORECAST_DIR and get_model_data_config_hycom_all().")
                if not need_animation:
                    sys.exit(1)
            elif (args.timing_distribution or args.detachment_count or args.separation_timing or args.no_lce_timeseries or args.no_lce_timeseries_by_date or args.no_lce_mean_std or args.no_lce_mean_std_by_date) and not need_full_processing:
                # Light path: timing/detachment/lc-only – LC contours only, no LCE detection
                lc_only_loaded = False
                if args.no_lce_timeseries or args.no_lce_timeseries_by_date or args.no_lce_mean_std or args.no_lce_mean_std_by_date:
                    lc_only_nc_path = os.path.join(OUTPUT_DIR, "mhd_OSEs_lc_only.nc")
                    if os.path.isfile(lc_only_nc_path):
                        print(f"[LC-only] Loading from existing {lc_only_nc_path}")
                        lc_only_results_ref, lc_only_results_gliders = load_mhd_from_netcdf(lc_only_nc_path)
                        if lc_only_results_ref or lc_only_results_gliders:
                            print(f"  Loaded REF: {len(lc_only_results_ref)}, GLIDERS: {len(lc_only_results_gliders)}")
                            lc_only_loaded = True
                        else:
                            print("  Load failed or file empty. Recomputing...")
                    if not lc_only_loaded:
                        lc_only_results_ref = []
                        lc_only_results_gliders = []
                need_loop = (args.timing_distribution or args.detachment_count or args.separation_timing or not lc_only_loaded)
                if need_loop:
                    print(f"[Timing/detachment/LC-only – lightweight] Found {len(configs)} forecasts")
                    print("=" * 60)
                    if args.timing_distribution:
                        timing_only_data_ref = []
                        timing_only_data_gliders = []
                    if args.detachment_count:
                        detachment_count_data_ref = []
                        detachment_count_data_gliders = []
                    if args.separation_timing or args.detachment_count:
                        aviso_date_max_lat = {}
                        sep_series_ref: List[Tuple[datetime, List[Tuple[datetime, float]], Dict[datetime, bool]]] = []
                        sep_series_gliders: List[Tuple[datetime, List[Tuple[datetime, float]], Dict[datetime, bool]]] = []
                for cfg_idx, (files_ref, files_gliders, grid_ref, grid_gliders, aviso_dir, mdt_path, fs_dt) in (enumerate(configs) if need_loop else []):
                    print(f"  Forecast {cfg_idx + 1}/{len(configs)}: {fs_dt.strftime('%Y-%m-%d')} ({len(files_ref)} REF, {len(files_gliders)} GLIDERS files)")
                    series_ref_model, series_ref_aviso = [], []
                    lce_ref_by_lead: Dict[int, bool] = {}
                    lce_ref_by_lead_det: Dict[int, bool] = {}
                    sep_ref_model_this: List[Tuple[datetime, float]] = []
                    sep_ref_lce_this: Dict[datetime, bool] = {}
                    for hycom_file in files_ref:
                        if args.timing_distribution or args.detachment_count or args.separation_timing:
                            out = process_hycom_file_for_timing_only(
                                hycom_file, grid_ref, aviso_dir, mdt_path, fs_dt, lon_cutoff=lon_cutoff,
                                detect_lce=True, lce_min_lon_span=lce_min_lon_span,
                                sep_min_lon=-91.5 if args.separation_timing else None,
                                det_min_lon=-91.5 if args.detachment_count else None,
                            )
                            if out is not None:
                                if len(out) == 6:
                                    lead, max_m, max_a, has_lce_timing, has_lce_sep, has_lce_det = out
                                else:
                                    lead, max_m, max_a = out[:3]
                                    has_lce_timing = has_lce_sep = has_lce_det = False
                                if max_m is not None:
                                    series_ref_model.append((lead, max_m))
                                    lce_ref_by_lead[lead] = has_lce_timing
                                    lce_ref_by_lead_det[lead] = has_lce_det
                                if max_a is not None:
                                    series_ref_aviso.append((lead, max_a))
                                if args.separation_timing or args.detachment_count:
                                    actual_date = fs_dt + timedelta(days=lead)
                                    if args.separation_timing:
                                        if max_m is not None:
                                            sep_ref_model_this.append((actual_date, max_m))
                                            sep_ref_lce_this[actual_date] = has_lce_sep
                                    if max_a is not None and actual_date not in aviso_date_max_lat:
                                        aviso_date_max_lat[actual_date] = max_a
                        if args.no_lce_timeseries or args.no_lce_timeseries_by_date or args.no_lce_mean_std or args.no_lce_mean_std_by_date:
                            r = process_hycom_file_for_lc_only_mhd(
                                hycom_file, grid_ref, aviso_dir, mdt_path, fs_dt, lon_cutoff=lon_cutoff
                            )
                            if r is not None:
                                lc_only_results_ref.append(r)
                    series_ref_model.sort(key=lambda x: x[0])
                    series_ref_aviso.sort(key=lambda x: x[0])
                    if args.timing_distribution:
                        d = compute_divergence_from_series(series_ref_model, series_ref_aviso,
                                                           lead_has_lce_model=lce_ref_by_lead)
                        if d is not None:
                            aviso_offset_ref = first_detachment_day_from_max_lat_series(series_ref_aviso)
                            timing_only_data_ref.append((fs_dt, d, aviso_offset_ref))
                    if args.detachment_count:
                        detachment_count_data_ref.append((
                            fs_dt,
                            count_detachments_from_max_lat_series(series_ref_model, lead_has_lce=lce_ref_by_lead_det),
                            count_detachments_from_max_lat_series(series_ref_aviso),
                        ))
                    if args.separation_timing and sep_ref_model_this:
                        sep_series_ref.append((fs_dt, sep_ref_model_this, sep_ref_lce_this))
                    series_gliders_model, series_gliders_aviso = [], []
                    lce_gliders_by_lead: Dict[int, bool] = {}
                    lce_gliders_by_lead_det: Dict[int, bool] = {}
                    sep_gliders_model_this: List[Tuple[datetime, float]] = []
                    sep_gliders_lce_this: Dict[datetime, bool] = {}
                    for hycom_file in files_gliders:
                        if args.timing_distribution or args.detachment_count or args.separation_timing:
                            out = process_hycom_file_for_timing_only(
                                hycom_file, grid_gliders, aviso_dir, mdt_path, fs_dt, lon_cutoff=lon_cutoff,
                                detect_lce=True, lce_min_lon_span=lce_min_lon_span,
                                sep_min_lon=-91.5 if args.separation_timing else None,
                                det_min_lon=-91.5 if args.detachment_count else None,
                            )
                            if out is not None:
                                if len(out) == 6:
                                    lead, max_m, max_a, has_lce_timing, has_lce_sep, has_lce_det = out
                                else:
                                    lead, max_m, max_a = out[:3]
                                    has_lce_timing = has_lce_sep = has_lce_det = False
                                if max_m is not None:
                                    series_gliders_model.append((lead, max_m))
                                    lce_gliders_by_lead[lead] = has_lce_timing
                                    lce_gliders_by_lead_det[lead] = has_lce_det
                                if max_a is not None:
                                    series_gliders_aviso.append((lead, max_a))
                                if args.separation_timing or args.detachment_count:
                                    actual_date = fs_dt + timedelta(days=lead)
                                    if args.separation_timing:
                                        if max_m is not None:
                                            sep_gliders_model_this.append((actual_date, max_m))
                                            sep_gliders_lce_this[actual_date] = has_lce_sep
                                    if max_a is not None and actual_date not in aviso_date_max_lat:
                                        aviso_date_max_lat[actual_date] = max_a
                        if args.no_lce_timeseries or args.no_lce_timeseries_by_date or args.no_lce_mean_std or args.no_lce_mean_std_by_date:
                            r = process_hycom_file_for_lc_only_mhd(
                                hycom_file, grid_gliders, aviso_dir, mdt_path, fs_dt, lon_cutoff=lon_cutoff
                            )
                            if r is not None:
                                lc_only_results_gliders.append(r)
                    series_gliders_model.sort(key=lambda x: x[0])
                    series_gliders_aviso.sort(key=lambda x: x[0])
                    if args.timing_distribution:
                        d = compute_divergence_from_series(series_gliders_model, series_gliders_aviso,
                                                           lead_has_lce_model=lce_gliders_by_lead)
                        if d is not None:
                            aviso_offset_gl = first_detachment_day_from_max_lat_series(series_gliders_aviso)
                            timing_only_data_gliders.append((fs_dt, d, aviso_offset_gl))
                    if args.detachment_count:
                        detachment_count_data_gliders.append((
                            fs_dt,
                            count_detachments_from_max_lat_series(series_gliders_model, lead_has_lce=lce_gliders_by_lead_det),
                            count_detachments_from_max_lat_series(series_gliders_aviso),
                        ))
                    if args.separation_timing and sep_gliders_model_this:
                        sep_series_gliders.append((fs_dt, sep_gliders_model_this, sep_gliders_lce_this))
                    sys.stdout.flush()
                if (args.no_lce_timeseries or args.no_lce_timeseries_by_date or args.no_lce_mean_std or args.no_lce_mean_std_by_date):
                    if not lc_only_loaded:
                        save_mhd_to_netcdf(lc_only_results_ref, lc_only_results_gliders, OUTPUT_DIR, filename="mhd_OSEs_lc_only.nc")
                    if args.no_lce_timeseries:
                        plot_timeseries_all_forecasts(lc_only_results_ref, lc_only_results_gliders, OUTPUT_DIR, suffix="_lc_only")
                    if args.no_lce_timeseries_by_date:
                        plot_timeseries_by_date(lc_only_results_ref, lc_only_results_gliders, OUTPUT_DIR, suffix="_lc_only")
                    if args.no_lce_mean_std:
                        plot_mean_std_from_results(lc_only_results_ref, lc_only_results_gliders, None, OUTPUT_DIR, suffix="_lc_only")
                    if args.no_lce_mean_std_by_date:
                        plot_mean_std_by_date(lc_only_results_ref, lc_only_results_gliders, OUTPUT_DIR, suffix="_lc_only")
                if args.separation_timing:
                    aviso_series = sorted(aviso_date_max_lat.items())
                    aviso_sep_date, aviso_confirmed = detect_final_separation(aviso_series)
                    if aviso_sep_date is None or not aviso_confirmed:
                        print("[Separation timing] Could not confirm final LC separation in AVISO. Skip histogram.")
                    else:
                        print(f"[Separation timing] AVISO final separation: {aviso_sep_date.strftime('%Y-%m-%d')}")
                        def _compute_sep_divergences(sep_series_model, label):
                            confirmed_items = []  # (divergence_days, forecast_start_date)
                            uncertain_items = []
                            no_sep_forecasts = []  # forecast_start_date with no separation detected
                            for fs_dt_i, date_ml, lce_dict in sep_series_model:
                                sep_date, is_confirmed = detect_final_separation(date_ml, date_has_lce=lce_dict)
                                if sep_date is not None:
                                    div = (sep_date - aviso_sep_date).days
                                    if is_confirmed:
                                        confirmed_items.append((div, fs_dt_i))
                                    else:
                                        uncertain_items.append((div, fs_dt_i))
                                else:
                                    no_sep_forecasts.append(fs_dt_i)
                            all_items = sorted(confirmed_items + uncertain_items, key=lambda x: x[0])
                            for div, fs_dt_i in all_items:
                                tag = "confirmed" if (div, fs_dt_i) in confirmed_items else "uncertain"
                                print(f"  [{label}] {tag} separation: forecast {fs_dt_i.strftime('%Y-%m-%d')} (divergence {div:+d} days)")
                            if all_items:
                                all_divs = [d for d, _ in all_items]
                                print(f"  [{label}] {len(confirmed_items)} confirmed, {len(uncertain_items)} uncertain; mean divergence: {np.mean(all_divs):.1f} days, std: {np.std(all_divs):.1f} days")
                            else:
                                print(f"  [{label}] No separations detected.")
                            return confirmed_items, uncertain_items, no_sep_forecasts
                        conf_ref, unc_ref, no_sep_ref = _compute_sep_divergences(sep_series_ref, ref_label)
                        conf_gl, unc_gl, no_sep_gl = _compute_sep_divergences(sep_series_gliders, gliders_label)
                        plot_separation_timing(conf_ref, unc_ref, conf_gl, unc_gl, OUTPUT_DIR, ref_label=ref_label, gliders_label=gliders_label, aviso_sep_date=aviso_sep_date)
                        timing_nc_path = os.path.join(OUTPUT_DIR, "lce_timing_OSEs.nc")
                        save_separation_timing_to_netcdf(aviso_sep_date, conf_ref, unc_ref, conf_gl, unc_gl, OUTPUT_DIR)
                        _aviso_tot_for_table: Optional[int] = None
                        if aviso_date_max_lat is not None and aviso_date_max_lat:
                            _aviso_det_ord = [(d.toordinal(), lat) for d, lat in sorted(aviso_date_max_lat.items())]
                            _aviso_tot_for_table = count_detachments_from_max_lat_series(_aviso_det_ord)
                        save_forecast_summary_table(
                            OUTPUT_DIR, ref_label, gliders_label,
                            detachment_count_data_ref or [],
                            detachment_count_data_gliders or [],
                            _aviso_tot_for_table,
                            no_sep_ref, no_sep_gl,
                        )
            else:
                # Full path: timeseries and/or mean_std (and possibly timing from results)
                mhd_path = os.path.join(OUTPUT_DIR, f"mhd_OSEs{lon_filter_suffix}.nc")
                if os.path.isfile(mhd_path):
                    print(f"[Timeseries/mean-std] Loading MHD from existing {mhd_path}")
                    results_ref_all, results_gliders_all = load_mhd_from_netcdf(mhd_path)
                    if not results_ref_all and not results_gliders_all:
                        print("  Load failed or file empty (e.g. old format). Recomputing from model files...")
                        results_ref_all = None
                        results_gliders_all = None
                    else:
                        print(f"  Loaded REF: {len(results_ref_all)}, GLIDERS: {len(results_gliders_all)}")
                if results_ref_all is not None and results_gliders_all is not None:
                    if args.timing_distribution or args.detachment_count:
                        # No contours in file; run light path for timing/detachment
                        if args.timing_distribution:
                            timing_only_data_ref = []
                            timing_only_data_gliders = []
                        if args.detachment_count:
                            detachment_count_data_ref = []
                            detachment_count_data_gliders = []
                        for cfg_idx, (files_ref, files_gliders, grid_ref, grid_gliders, aviso_dir, mdt_path, fs_dt) in enumerate(configs):
                            print(f"  Forecast {cfg_idx + 1}/{len(configs)}: {fs_dt.strftime('%Y-%m-%d')} (timing)...")
                            series_ref_model, series_ref_aviso = [], []
                            lce_ref_by_lead_fp: Dict[int, bool] = {}
                            lce_ref_by_lead_fp_det: Dict[int, bool] = {}
                            for hycom_file in files_ref:
                                out = process_hycom_file_for_timing_only(
                                    hycom_file, grid_ref, aviso_dir, mdt_path, fs_dt, lon_cutoff=lon_cutoff,
                                    detect_lce=True, lce_min_lon_span=lce_min_lon_span,
                                    det_min_lon=-91.5 if args.detachment_count else None,
                                )
                                if out is not None:
                                    if len(out) == 6:
                                        lead, max_m, max_a, has_lce_timing, _, has_lce_det = out
                                    else:
                                        lead, max_m, max_a = out[:3]
                                        has_lce_timing = has_lce_det = False
                                    if max_m is not None:
                                        series_ref_model.append((lead, max_m))
                                        lce_ref_by_lead_fp[lead] = has_lce_timing
                                        lce_ref_by_lead_fp_det[lead] = has_lce_det
                                    if max_a is not None:
                                        series_ref_aviso.append((lead, max_a))
                            series_ref_model.sort(key=lambda x: x[0])
                            series_ref_aviso.sort(key=lambda x: x[0])
                            if args.timing_distribution:
                                d = compute_divergence_from_series(series_ref_model, series_ref_aviso,
                                                                   lead_has_lce_model=lce_ref_by_lead_fp)
                                if d is not None:
                                    timing_only_data_ref.append((fs_dt, d))
                            if args.detachment_count:
                                detachment_count_data_ref.append((
                                    fs_dt,
                                    count_detachments_from_max_lat_series(series_ref_model, lead_has_lce=lce_ref_by_lead_fp_det),
                                    count_detachments_from_max_lat_series(series_ref_aviso),
                                ))
                            series_gliders_model, series_gliders_aviso = [], []
                            lce_gliders_by_lead_fp: Dict[int, bool] = {}
                            lce_gliders_by_lead_fp_det: Dict[int, bool] = {}
                            for hycom_file in files_gliders:
                                out = process_hycom_file_for_timing_only(
                                    hycom_file, grid_gliders, aviso_dir, mdt_path, fs_dt, lon_cutoff=lon_cutoff,
                                    detect_lce=True, lce_min_lon_span=lce_min_lon_span,
                                    det_min_lon=-91.5 if args.detachment_count else None,
                                )
                                if out is not None:
                                    if len(out) == 6:
                                        lead, max_m, max_a, has_lce_timing, _, has_lce_det = out
                                    else:
                                        lead, max_m, max_a = out[:3]
                                        has_lce_timing = has_lce_det = False
                                    if max_m is not None:
                                        series_gliders_model.append((lead, max_m))
                                        lce_gliders_by_lead_fp[lead] = has_lce_timing
                                        lce_gliders_by_lead_fp_det[lead] = has_lce_det
                                    if max_a is not None:
                                        series_gliders_aviso.append((lead, max_a))
                            series_gliders_model.sort(key=lambda x: x[0])
                            series_gliders_aviso.sort(key=lambda x: x[0])
                            if args.timing_distribution:
                                d = compute_divergence_from_series(series_gliders_model, series_gliders_aviso,
                                                                   lead_has_lce_model=lce_gliders_by_lead_fp)
                                if d is not None:
                                    timing_only_data_gliders.append((fs_dt, d))
                            if args.detachment_count:
                                detachment_count_data_gliders.append((
                                    fs_dt,
                                    count_detachments_from_max_lat_series(series_gliders_model, lead_has_lce=lce_gliders_by_lead_fp_det),
                                    count_detachments_from_max_lat_series(series_gliders_aviso),
                                ))
                            sys.stdout.flush()
                    if need_animation and not results_ref:
                        forecast_start_dt = configs[0][6]
                    if not need_animation:
                        results_ref = results_ref_all
                        results_gliders = results_gliders_all
                else:
                    print(f"[Timeseries/mean-std/timing – all forecasts] Found {len(configs)} forecasts")
                    print("=" * 60)
                    results_ref_all = []
                    results_gliders_all = []
                    for cfg_idx, (files_ref, files_gliders, grid_ref, grid_gliders, aviso_dir, mdt_path, fs_dt) in enumerate(configs):
                        print(f"  Forecast {cfg_idx + 1}/{len(configs)}: {fs_dt.strftime('%Y-%m-%d')} ({len(files_ref)} REF, {len(files_gliders)} GLIDERS files)")
                        for hycom_file in files_ref:
                            result = process_hycom_file_for_animation(
                                hycom_file=hycom_file, grid_file=grid_ref,
                                aviso_dir=aviso_dir, mdt_path=mdt_path,
                                lon_cutoff=lon_cutoff, use_cutoff=use_cutoff,
                                lce_min_lon_span=lce_min_lon_span,
                            )
                            result["forecast_start"] = fs_dt
                            results_ref_all.append(result)
                        for hycom_file in files_gliders:
                            result = process_hycom_file_for_animation(
                                hycom_file=hycom_file, grid_file=grid_gliders,
                                aviso_dir=aviso_dir, mdt_path=mdt_path,
                                lon_cutoff=lon_cutoff, use_cutoff=use_cutoff,
                                lce_min_lon_span=lce_min_lon_span,
                            )
                            result["forecast_start"] = fs_dt
                            results_gliders_all.append(result)
                        sys.stdout.flush()
                    if need_animation and not results_ref:
                        forecast_start_dt = configs[0][6]
                    if not need_animation:
                        results_ref = results_ref_all
                        results_gliders = results_gliders_all


    elif args.multi_simulation:
        # ---------------------------------------------------------------------
        # Multi-simulation NetCDF: compare many simulation directories to AVISO
        # ---------------------------------------------------------------------
        if not args.sim_dirs or not args.aviso_dir or not args.mdt:
            parser.error("--multi-simulation requires --sim-dirs, --aviso-dir, and --mdt")
        if args.sim_labels is not None and len(args.sim_labels) != len(args.sim_dirs):
            parser.error("--sim-labels must have the same length as --sim-dirs")
        if getattr(args, "mercator", False) and not getattr(args, "grid_netcdf", None):
            parser.error("--mercator requires --grid-netcdf (path to NetCDF grid file with cell sizes)")

        need_animation = args.animate
        sim_labels = args.sim_labels if args.sim_labels is not None else [os.path.basename(p.rstrip("/")) or f"sim_{i+1}" for i, p in enumerate(args.sim_dirs)]
        fallback_fs = None
        if args.forecast_start:
            try:
                fallback_fs = datetime.strptime(args.forecast_start, "%Y-%m-%d")
            except ValueError:
                print(f"Warning: --forecast-start {args.forecast_start} invalid (use YYYY-MM-DD). Ignoring fallback.")
                fallback_fs = None

        def collect_netcdf_file_date_pairs(nc_dir):
            pattern = os.path.join(nc_dir, args.netcdf_pattern)
            nc_files = sorted(glob.glob(pattern))
            nc_files = [f for f in nc_files if os.path.isfile(f) and f.endswith(".nc")]
            nc_files = [f for f in nc_files if not os.path.basename(f).startswith("mhd_OSEs")]
            pairs = []
            for f in nc_files:
                date_str = extract_date_from_netcdf_path(f)
                if date_str:
                    pairs.append((f, date_str))
            return pairs

        def collect_netcdf_file_date_pairs_grouped(nc_dir):
            grouped = defaultdict(list)
            for f, date_str in collect_netcdf_file_date_pairs(nc_dir):
                group_name = os.path.basename(os.path.dirname(f)) or "root"
                grouped[group_name].append((f, date_str))
            for key in grouped:
                grouped[key] = sorted(grouped[key], key=lambda x: x[1])
            return grouped

        def infer_fs_from_pairs(pairs: List[Tuple[str, str]], fallback_dt: Optional[datetime]) -> Optional[datetime]:
            """Infer forecast start from parent folder; fallback to earliest file date; then CLI fallback."""
            if not pairs:
                return fallback_dt
            fs = infer_forecast_start_from_path(pairs[0][0])
            if fs is not None:
                return fs
            try:
                return datetime.strptime(min(d for _, d in pairs), "%Y%m%d")
            except Exception:
                return fallback_dt

        simulation_results: Dict[str, List[Dict]] = {}
        grouped_pairs_by_sim: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        for sim_dir, sim_label in zip(args.sim_dirs, sim_labels):
            file_date_pairs = collect_netcdf_file_date_pairs(sim_dir)
            if len(file_date_pairs) == 0:
                print(f"Warning: No NetCDF files with date token found for simulation {sim_label} in {sim_dir}. Skipping.")
                continue
            print(f"[{sim_label}] Found {len(file_date_pairs)} NetCDF files")
            sim_default_fs = infer_fs_from_pairs(file_date_pairs, fallback_fs)
            if sim_default_fs is not None:
                print(f"[{sim_label}] forecast_start used for grouping/timing: {sim_default_fs.strftime('%Y-%m-%d')}")
            results = []
            for i, (nc_path, date_str) in enumerate(file_date_pairs, 1):
                if i % 10 == 0 or i == 1:
                    print(f"  [{i}/{len(file_date_pairs)}] Processing {os.path.basename(nc_path)}...")
                fs_for_file = infer_forecast_start_from_path(nc_path) or sim_default_fs
                result = process_netcdf_file_for_animation(
                    nc_path=nc_path, date=date_str,
                    aviso_dir=args.aviso_dir, mdt_path=args.mdt,
                    lon_var=args.lon_var, lat_var=args.lat_var, ssh_var=args.ssh_var,
                    time_idx=0, ssh_scale=args.ssh_scale,
                    lon_cutoff=lon_cutoff, use_cutoff=use_cutoff,
                    grid_path_netcdf=getattr(args, "grid_netcdf", None),
                    mercator=getattr(args, "mercator", False),
                    grid_dx_var=getattr(args, "grid_dx_var", "pscx"),
                    grid_dy_var=getattr(args, "grid_dy_var", "pscy"),
                    lce_min_lon_span=lce_min_lon_span,
                )
                if fs_for_file is not None:
                    result["forecast_start"] = fs_for_file
                results.append(result)
                sys.stdout.flush()
            simulation_results[sim_label] = results
            grouped_pairs_by_sim[sim_label] = collect_netcdf_file_date_pairs_grouped(sim_dir)

        if not simulation_results:
            print("Error: No simulation results were produced in --multi-simulation mode.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("SUMMARY (MULTI-SIMULATION)")
        print("=" * 60)
        for label, results in simulation_results.items():
            n_ok = sum(1 for r in results if r.get("success"))
            print(f"{label}: {n_ok}/{len(results)} successful")

        # In multi-simulation mode, use the maximum available lead from data (not fixed 90 days).
        multi_max_lead_days = None
        observed_leads = []
        for results in simulation_results.values():
            for r in results:
                if not r.get("success") or r.get("date") is None:
                    continue
                fs = r.get("forecast_start")
                if fs is None:
                    continue
                lead = r.get("lead_time_days")
                if lead is None:
                    date_obj = datetime.strptime(r["date"], "%Y%m%d")
                    lead = (date_obj - fs).days
                if lead >= 0:
                    observed_leads.append(lead)
        if observed_leads:
            multi_max_lead_days = max(observed_leads)
            print(f"[Multi-simulation] Using max lead days from data: {multi_max_lead_days}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base_out = args.output if os.path.isabs(args.output) else os.path.join(OUTPUT_DIR, args.output)
        base_stem, base_ext = os.path.splitext(base_out)

        if need_animation:
            if getattr(args, "animate_all", False):
                for sim_label, grouped in grouped_pairs_by_sim.items():
                    safe_label = _sanitize_label_for_filename(sim_label)
                    print(f"[Animation – all groups] {sim_label}: {len(grouped)} group(s)")
                    for grp_idx, group_name in enumerate(sorted(grouped.keys()), 1):
                        pairs = grouped[group_name]
                        if not pairs:
                            continue
                        group_fs = infer_fs_from_pairs(pairs, sim_default_fs)
                        group_results = []
                        for nc_path, date_str in pairs:
                            result = process_netcdf_file_for_animation(
                                nc_path=nc_path, date=date_str,
                                aviso_dir=args.aviso_dir, mdt_path=args.mdt,
                                lon_var=args.lon_var, lat_var=args.lat_var, ssh_var=args.ssh_var,
                                time_idx=0, ssh_scale=args.ssh_scale,
                                lon_cutoff=lon_cutoff, use_cutoff=use_cutoff,
                                grid_path_netcdf=getattr(args, "grid_netcdf", None),
                                mercator=getattr(args, "mercator", False),
                                grid_dx_var=getattr(args, "grid_dx_var", "pscx"),
                                grid_dy_var=getattr(args, "grid_dy_var", "pscy"),
                                lce_min_lon_span=lce_min_lon_span,
                            )
                            if group_fs is not None:
                                result["forecast_start"] = group_fs
                            group_results.append(result)
                        safe_group = _sanitize_label_for_filename(group_name) or f"group_{grp_idx:03d}"
                        out_anim = f"{base_stem}_{safe_label}_{safe_group}{base_ext}"
                        if any(r.get("success") for r in group_results):
                            create_animation(
                                results_ref=group_results,
                                results_gliders=[],
                                output_file=out_anim,
                                fps=2,
                                ref_label=sim_label,
                                gliders_label="",
                            )
                        else:
                            print(f"  Skip animation for {sim_label}/{group_name}: no valid results.")
            else:
                for sim_label, results in simulation_results.items():
                    safe_label = _sanitize_label_for_filename(sim_label)
                    out_anim = f"{base_stem}_{safe_label}{base_ext}"
                    if any(r.get("success") for r in results):
                        create_animation(
                            results_ref=results,
                            results_gliders=[],
                            output_file=out_anim,
                            fps=2,
                            ref_label=sim_label,
                            gliders_label="",
                        )
                    else:
                        print(f"Skip animation for {sim_label}: no valid results.")

        if args.timeseries:
            plot_timeseries_multi_simulation(
                simulation_results,
                OUTPUT_DIR,
                max_lead_days=multi_max_lead_days,
            )
        if args.mean_std:
            plot_mean_std_multi_simulation(
                simulation_results,
                OUTPUT_DIR,
                max_lead_days=multi_max_lead_days,
            )
        if args.timing_distribution:
            timing_by_sim: Dict[str, List[int]] = {}
            for sim_label, results in simulation_results.items():
                by_fs = defaultdict(list)
                for r in results:
                    fs = r.get("forecast_start")
                    if fs is not None:
                        by_fs[fs].append(r)
                divergences = []
                n_groups = 0
                n_both = 0
                n_model_only = 0
                n_aviso_only = 0
                n_neither = 0
                for fs, group in by_fs.items():
                    n_groups += 1
                    series_model, series_aviso = _build_max_lat_series_from_results(
                        results=group,
                        forecast_start=fs,
                        lon_cutoff=lon_cutoff,
                        max_lead_days=multi_max_lead_days,
                    )
                    first_model = first_detachment_day_from_max_lat_series(series_model)
                    first_aviso = first_detachment_day_from_max_lat_series(series_aviso)
                    if first_model is not None and first_aviso is not None:
                        n_both += 1
                        divergences.append(first_model - first_aviso)
                    elif first_model is not None:
                        n_model_only += 1
                    elif first_aviso is not None:
                        n_aviso_only += 1
                    else:
                        n_neither += 1

                    if first_model is None or first_aviso is None:
                        model_drop = _max_drop_from_running_max(series_model)
                        aviso_drop = _max_drop_from_running_max(series_aviso)
                        valid_dates = sorted(
                            r["date"] for r in group if r.get("success") and r.get("date") is not None
                        )
                        d0 = valid_dates[0] if valid_dates else "NA"
                        d1 = valid_dates[-1] if valid_dates else "NA"
                        lm = f"{series_model[0][0]}-{series_model[-1][0]}" if series_model else "NA"
                        la = f"{series_aviso[0][0]}-{series_aviso[-1][0]}" if series_aviso else "NA"
                        print(
                            f"[Timing diag] {sim_label} fs={fs.strftime('%Y-%m-%d')} dates={d0}->{d1} "
                            f"model_pts={len(series_model)} aviso_pts={len(series_aviso)} "
                            f"model_leads={lm} aviso_leads={la} "
                            f"first_model={first_model} first_aviso={first_aviso} "
                            f"max_drop_model={model_drop:.2f} max_drop_aviso={aviso_drop:.2f} "
                            f"threshold={LC_NORTH_DROP_DEGREES:.2f}"
                        )
                timing_by_sim[sim_label] = divergences
                print(
                    f"[Timing summary] {sim_label}: groups={n_groups}, both={n_both}, "
                    f"model_only={n_model_only}, aviso_only={n_aviso_only}, neither={n_neither}, "
                    f"divergences={len(divergences)}"
                )
            plot_timing_distribution_multi_simulation(timing_by_sim, OUTPUT_DIR)

        sys.exit(0)

    else:
        # ---------------------------------------------------------------------
        # NetCDF: user provides --netcdf-dir (REF), optional --netcdf-dir-gliders, --aviso-dir, --mdt
        # ---------------------------------------------------------------------
        if not args.netcdf_dir or not args.aviso_dir or not args.mdt:
            parser.error("--no-hycom requires --netcdf-dir, --aviso-dir, and --mdt")
        if getattr(args, "mercator", False) and not getattr(args, "grid_netcdf", None):
            parser.error("--mercator requires --grid-netcdf (path to NetCDF grid file with cell sizes)")
        need_animation = args.animate
        need_rest = args.timeseries or args.timeseries_by_date or args.no_lce_timeseries or args.no_lce_timeseries_by_date or args.no_lce_mean_std or args.mean_std or args.mean_std_by_date or args.no_lce_mean_std_by_date or args.timing_distribution or args.detachment_count or args.separation_timing
        forecast_start_dt = None
        if args.forecast_start:
            try:
                forecast_start_dt = datetime.strptime(args.forecast_start, "%Y-%m-%d")
            except ValueError:
                print(f"Warning: --forecast-start {args.forecast_start} invalid (use YYYY-MM-DD).")
                forecast_start_dt = None
        def collect_netcdf_file_date_pairs(nc_dir):
            pattern = os.path.join(nc_dir, args.netcdf_pattern)
            nc_files = sorted(glob.glob(pattern))
            nc_files = [f for f in nc_files if os.path.isfile(f) and f.endswith(".nc")]
            # Do not treat our own MHD output as model input
            nc_files = [f for f in nc_files if not os.path.basename(f).startswith("mhd_OSEs")]
            pairs = []
            for f in nc_files:
                date_str = extract_date_from_netcdf_path(f)
                if date_str:
                    pairs.append((f, date_str))
            return pairs
        def collect_netcdf_file_date_pairs_grouped(nc_dir):
            grouped = defaultdict(list)
            for f, date_str in collect_netcdf_file_date_pairs(nc_dir):
                group_name = os.path.basename(os.path.dirname(f)) or "root"
                grouped[group_name].append((f, date_str))
            for key in grouped:
                grouped[key] = sorted(grouped[key], key=lambda x: x[1])
            return grouped

        # NetCDF animate-all: one animation per subfolder group.
        if need_animation and getattr(args, "animate_all", False):
            grouped_ref = collect_netcdf_file_date_pairs_grouped(args.netcdf_dir)
            if not grouped_ref:
                print("Error: No NetCDF files with YYYYMMDD in filename found in REF dir for --animate-all.")
                print("  Example: my_20250601.nc or 2025-06-01_ssh.nc")
                sys.exit(1)
            grouped_gliders = collect_netcdf_file_date_pairs_grouped(args.netcdf_dir_gliders) if getattr(args, "netcdf_dir_gliders", None) else {}
            base_out = args.output if os.path.isabs(args.output) else os.path.join(OUTPUT_DIR, args.output)
            base_stem, base_ext = os.path.splitext(base_out)
            print(f"[Animation – all groups] Found {len(grouped_ref)} group(s) in REF NetCDF dir")
            print("=" * 60)
            for grp_idx, group_name in enumerate(sorted(grouped_ref.keys()), 1):
                ref_pairs = grouped_ref[group_name]
                gl_pairs = grouped_gliders.get(group_name, [])
                group_fs = infer_forecast_start_from_path(ref_pairs[0][0]) if ref_pairs else None
                if group_fs is None:
                    group_fs = forecast_start_dt
                print(f"  Group {grp_idx}/{len(grouped_ref)}: {group_name} ({len(ref_pairs)} REF, {len(gl_pairs)} GLIDERS files)")
                anim_ref = []
                anim_gl = []
                for i, (nc_path, date_str) in enumerate(ref_pairs, 1):
                    if i % 10 == 0 or i == 1:
                        print(f"    [{i}/{len(ref_pairs)}] REF {os.path.basename(nc_path)}...")
                    result = process_netcdf_file_for_animation(
                        nc_path=nc_path, date=date_str,
                        aviso_dir=args.aviso_dir, mdt_path=args.mdt,
                        lon_var=args.lon_var, lat_var=args.lat_var, ssh_var=args.ssh_var,
                        time_idx=0, ssh_scale=args.ssh_scale,
                        lon_cutoff=lon_cutoff, use_cutoff=use_cutoff,
                        grid_path_netcdf=getattr(args, "grid_netcdf", None),
                        mercator=getattr(args, "mercator", False),
                        grid_dx_var=getattr(args, "grid_dx_var", "pscx"),
                        grid_dy_var=getattr(args, "grid_dy_var", "pscy"),
                        lce_min_lon_span=lce_min_lon_span,
                    )
                    if group_fs is not None:
                        result["forecast_start"] = group_fs
                    anim_ref.append(result)
                    sys.stdout.flush()
                for i, (nc_path, date_str) in enumerate(gl_pairs, 1):
                    if i % 10 == 0 or i == 1:
                        print(f"    [{i}/{len(gl_pairs)}] GLIDERS {os.path.basename(nc_path)}...")
                    result = process_netcdf_file_for_animation(
                        nc_path=nc_path, date=date_str,
                        aviso_dir=args.aviso_dir, mdt_path=args.mdt,
                        lon_var=args.lon_var, lat_var=args.lat_var, ssh_var=args.ssh_var,
                        time_idx=0, ssh_scale=args.ssh_scale,
                        lon_cutoff=lon_cutoff, use_cutoff=use_cutoff,
                        grid_path_netcdf=getattr(args, "grid_netcdf", None),
                        mercator=getattr(args, "mercator", False),
                        grid_dx_var=getattr(args, "grid_dx_var", "pscx"),
                        grid_dy_var=getattr(args, "grid_dy_var", "pscy"),
                        lce_min_lon_span=lce_min_lon_span,
                    )
                    if group_fs is not None:
                        result["forecast_start"] = group_fs
                    anim_gl.append(result)
                    sys.stdout.flush()
                safe_group = re.sub(r"[^A-Za-z0-9._-]+", "_", group_name).strip("_") or f"group_{grp_idx:03d}"
                out_anim = f"{base_stem}_{safe_group}{base_ext}"
                if any(r.get("success") for r in anim_ref):
                    create_animation(
                        results_ref=anim_ref,
                        results_gliders=anim_gl,
                        output_file=out_anim,
                        fps=2,
                        ref_label=args.model_label,
                        gliders_label=getattr(args, "model_label_gliders", "Model_GLIDERS"),
                    )
                else:
                    print(f"    Skip animation (no valid REF results for group {group_name})")
            animate_all_done = True

        if need_animation and getattr(args, "animate_all", False) and not need_rest:
            results_ref = []
            results_gliders = []
        else:
            file_date_pairs_ref = collect_netcdf_file_date_pairs(args.netcdf_dir)
            if len(file_date_pairs_ref) == 0:
                print("Error: No NetCDF files with YYYYMMDD in filename found in REF dir.")
                print("  Example: my_20250601.nc or 2025-06-01_ssh.nc")
                sys.exit(1)
            print(f"Found {len(file_date_pairs_ref)} NetCDF files (REF)")
            print("=" * 60)

            results_ref = []
            for i, (nc_path, date_str) in enumerate(file_date_pairs_ref, 1):
                if i % 10 == 0 or i == 1:
                    print(f"[{i}/{len(file_date_pairs_ref)}] Processing REF {os.path.basename(nc_path)}...")
                fs_for_file = infer_forecast_start_from_path(nc_path) or forecast_start_dt
                result = process_netcdf_file_for_animation(
                    nc_path=nc_path, date=date_str,
                    aviso_dir=args.aviso_dir, mdt_path=args.mdt,
                    lon_var=args.lon_var, lat_var=args.lat_var, ssh_var=args.ssh_var,
                    time_idx=0, ssh_scale=args.ssh_scale,
                    lon_cutoff=lon_cutoff, use_cutoff=use_cutoff,
                    grid_path_netcdf=getattr(args, "grid_netcdf", None),
                    mercator=getattr(args, "mercator", False),
                    grid_dx_var=getattr(args, "grid_dx_var", "pscx"),
                    grid_dy_var=getattr(args, "grid_dy_var", "pscy"),
                    lce_min_lon_span=lce_min_lon_span,
                )
                if fs_for_file is not None:
                    result["forecast_start"] = fs_for_file
                results_ref.append(result)
                if result['success'] and (i % 10 == 0 or i == 1):
                    print(f"  Date: {result['date']}, MHD: ~{result['mhd'] * 111.0:.2f} km")
                sys.stdout.flush()

            results_gliders = []
            if getattr(args, "netcdf_dir_gliders", None):
                file_date_pairs_gliders = collect_netcdf_file_date_pairs(args.netcdf_dir_gliders)
                if len(file_date_pairs_gliders) > 0:
                    print(f"Found {len(file_date_pairs_gliders)} NetCDF files (GLIDERS)")
                    for i, (nc_path, date_str) in enumerate(file_date_pairs_gliders, 1):
                        if i % 10 == 0 or i == 1:
                            print(f"[{i}/{len(file_date_pairs_gliders)}] Processing GLIDERS {os.path.basename(nc_path)}...")
                        fs_for_file = infer_forecast_start_from_path(nc_path) or forecast_start_dt
                        result = process_netcdf_file_for_animation(
                            nc_path=nc_path, date=date_str,
                            aviso_dir=args.aviso_dir, mdt_path=args.mdt,
                            lon_var=args.lon_var, lat_var=args.lat_var, ssh_var=args.ssh_var,
                            time_idx=0, ssh_scale=args.ssh_scale,
                            lon_cutoff=lon_cutoff, use_cutoff=use_cutoff,
                            grid_path_netcdf=getattr(args, "grid_netcdf", None),
                            mercator=getattr(args, "mercator", False),
                            grid_dx_var=getattr(args, "grid_dx_var", "pscx"),
                            grid_dy_var=getattr(args, "grid_dy_var", "pscy"),
                            lce_min_lon_span=lce_min_lon_span,
                        )
                        if fs_for_file is not None:
                            result["forecast_start"] = fs_for_file
                        results_gliders.append(result)
                        if result['success'] and (i % 10 == 0 or i == 1):
                            print(f"  Date: {result['date']}, MHD: ~{result['mhd'] * 111.0:.2f} km")
                        sys.stdout.flush()
        ref_label = args.model_label
        gliders_label = getattr(args, "model_label_gliders", "Model_GLIDERS")
        if (args.mean_std or args.timing_distribution) and any(r.get("forecast_start") is not None for r in results_ref):
            # Non-HYCOM multi-forecast run with inferred forecast starts: same downstream behavior as HYCOM all-forecast mode.
            results_ref_all = results_ref
            results_gliders_all = results_gliders

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if timing_only_data_ref is not None or timing_only_data_gliders is not None:
        n_ref = len(timing_only_data_ref) if timing_only_data_ref else 0
        n_gl = len(timing_only_data_gliders) if timing_only_data_gliders else 0
        print(f"Timing only (light path): REF {n_ref}, GLIDERS {n_gl} divergence values")
    else:
        print(f"Model (REF): {len(results_ref)} results")
        if len(results_gliders) > 0:
            print(f"Model (GLIDERS): {len(results_gliders)} results")
        if len(results_ref) == 0:
            if args.animate and animate_all_done and not (args.timeseries or args.mean_std or args.timing_distribution or args.detachment_count):
                print("Animation-all completed; skipping REF-results check for non-plot run.")
            elif args.detachment_count and not (args.timeseries or args.timeseries_by_date or args.mean_std or args.animate):
                pass  # lightweight detachment path: results_ref is empty by design
            elif (args.no_lce_timeseries or args.no_lce_timeseries_by_date or args.no_lce_mean_std or args.no_lce_mean_std_by_date or args.separation_timing) and not (args.timeseries or args.timeseries_by_date or args.mean_std or args.mean_std_by_date or args.animate):
                pass  # lightweight lc-only / separation-timing path: results_ref is empty by design
            else:
                print("Error: No REF results.")
                sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_anim = args.output if os.path.isabs(args.output) else os.path.join(OUTPUT_DIR, args.output)

    if args.animate and not animate_all_done:
        if len(results_ref) == 0:
            print("Skipping animation: no REF results.")
        else:
            create_animation(
                results_ref=results_ref,
                results_gliders=results_gliders,
                output_file=out_anim,
                fps=2,
                ref_label=ref_label,
                gliders_label=gliders_label,
            )
    ref_for_rest = results_ref_all if results_ref_all is not None else results_ref
    gliders_for_rest = results_gliders_all if results_gliders_all is not None else results_gliders

    if need_full_processing:
        save_mhd_to_netcdf(ref_for_rest, gliders_for_rest, OUTPUT_DIR, filename=f"mhd_OSEs{lon_filter_suffix}.nc")

    if args.timeseries:
        plot_timeseries_all_forecasts(
            ref_for_rest, gliders_for_rest, OUTPUT_DIR, ref_label=ref_label, gliders_label=gliders_label,
            suffix=lon_filter_suffix,
        )
    if args.timeseries_by_date:
        plot_timeseries_by_date(
            ref_for_rest, gliders_for_rest, OUTPUT_DIR, ref_label=ref_label, gliders_label=gliders_label,
            suffix=lon_filter_suffix,
        )
    if args.mean_std_by_date:
        plot_mean_std_by_date(
            ref_for_rest, gliders_for_rest, OUTPUT_DIR, ref_label=ref_label, gliders_label=gliders_label,
            suffix=lon_filter_suffix,
        )
    if args.mean_std:
        if results_ref_all is not None:
            plot_mean_std_from_results(
                ref_for_rest, gliders_for_rest, None, OUTPUT_DIR,
                ref_label=ref_label, gliders_label=gliders_label,
                suffix=lon_filter_suffix,
            )
        elif forecast_start_dt is not None:
            plot_mean_std_from_results(
                results_ref, results_gliders, forecast_start_dt, OUTPUT_DIR,
                ref_label=ref_label, gliders_label=gliders_label,
                suffix=lon_filter_suffix,
            )
        else:
            print("Skipping mean-std: no forecast start date (set --forecast-start for NetCDF or check HYCOM path).")
    if args.no_lce_mean_std or args.no_lce_mean_std_by_date or args.no_lce_timeseries or args.no_lce_timeseries_by_date:
        lc_only_ref = _lc_only_results(ref_for_rest)
        lc_only_gl  = _lc_only_results(gliders_for_rest)
        save_mhd_to_netcdf(lc_only_ref, lc_only_gl, OUTPUT_DIR, filename="mhd_OSEs_lc_only.nc")
        if args.no_lce_timeseries:
            plot_timeseries_all_forecasts(lc_only_ref, lc_only_gl, OUTPUT_DIR, ref_label=ref_label, gliders_label=gliders_label, suffix="_lc_only")
        if args.no_lce_timeseries_by_date:
            plot_timeseries_by_date(lc_only_ref, lc_only_gl, OUTPUT_DIR, ref_label=ref_label, gliders_label=gliders_label, suffix="_lc_only")
        if args.no_lce_mean_std:
            plot_mean_std_from_results(lc_only_ref, lc_only_gl, None, OUTPUT_DIR, ref_label=ref_label, gliders_label=gliders_label, suffix="_lc_only")
        if args.no_lce_mean_std_by_date:
            plot_mean_std_by_date(lc_only_ref, lc_only_gl, OUTPUT_DIR, ref_label=ref_label, gliders_label=gliders_label, suffix="_lc_only")
    if args.timing_distribution:
        # If a timing NetCDF already exists, read from it and do not recompute
        timing_nc_path = os.path.join(OUTPUT_DIR, "lce_timing_OSEs.nc")
        loaded_from_nc = False
        if os.path.isfile(timing_nc_path):
            timing_ref_loaded, timing_gliders_loaded, _ = load_lce_timing_from_netcdf(timing_nc_path)
            if timing_ref_loaded or timing_gliders_loaded:
                plot_timing_distribution(
                    timing_ref_loaded,
                    timing_gliders_loaded,
                    OUTPUT_DIR,
                    ref_label=ref_label,
                    gliders_label=gliders_label,
                )
                # Use loaded data for summary and skip recomputation entirely
                timing_only_data_ref = timing_ref_loaded
                timing_only_data_gliders = timing_gliders_loaded
                loaded_from_nc = True
        if not loaded_from_nc:
            timing_data_ref = []
            timing_data_gliders = []
            if timing_only_data_ref is not None or timing_only_data_gliders is not None:
                plot_timing_distribution(
                    timing_only_data_ref or [],
                    timing_only_data_gliders or [],
                    OUTPUT_DIR,
                    ref_label=ref_label,
                    gliders_label=gliders_label,
                )
                timing_data_ref = timing_only_data_ref or []
                timing_data_gliders = timing_only_data_gliders or []
            elif results_ref_all is not None:
                # All forecasts (full path): one divergence per forecast per run type
                ref_by_fs = defaultdict(list)
                for r in ref_for_rest:
                    if r.get("forecast_start") is not None:
                        ref_by_fs[r["forecast_start"]].append(r)
                gliders_by_fs = defaultdict(list)
                for r in gliders_for_rest:
                    if r.get("forecast_start") is not None:
                        gliders_by_fs[r["forecast_start"]].append(r)
                for fs in ref_by_fs:
                    d = compute_divergence_from_results(ref_by_fs[fs], fs, lon_cutoff=lon_cutoff)
                    if d is not None:
                        timing_data_ref.append((fs, d))
                for fs in gliders_by_fs:
                    d = compute_divergence_from_results(gliders_by_fs[fs], fs, lon_cutoff=lon_cutoff)
                    if d is not None:
                        timing_data_gliders.append((fs, d))
                plot_timing_distribution(
                    timing_data_ref,
                    timing_data_gliders,
                    OUTPUT_DIR,
                    ref_label=ref_label, gliders_label=gliders_label,
                )
            elif forecast_start_dt is not None:
                div_ref = compute_divergence_from_results(results_ref, forecast_start_dt, lon_cutoff=lon_cutoff)
                div_gliders = compute_divergence_from_results(results_gliders, forecast_start_dt, lon_cutoff=lon_cutoff)
                timing_data_ref = [(forecast_start_dt, div_ref)] if div_ref is not None else []
                timing_data_gliders = [(forecast_start_dt, div_gliders)] if div_gliders is not None else []
                plot_timing_distribution(
                    timing_data_ref,
                    timing_data_gliders,
                    OUTPUT_DIR,
                    ref_label=ref_label,
                    gliders_label=gliders_label,
                )
            else:
                print("Skipping timing-distribution: no forecast start date (set --forecast-start for NetCDF or check HYCOM path).")
            if timing_data_ref or timing_data_gliders:
                save_lce_timing_to_netcdf(timing_data_ref, timing_data_gliders, OUTPUT_DIR, filename="lce_timing_OSEs.nc")
    if args.detachment_count:
        if detachment_count_data_ref is not None or detachment_count_data_gliders is not None:
            aviso_total_detachments: Optional[int] = aviso_total_detachments_from_nc
            if aviso_date_max_lat is not None:
                aviso_series_full = sorted(aviso_date_max_lat.items())
                if aviso_series_full:
                    aviso_det_by_ord = [(d.toordinal(), lat) for d, lat in aviso_series_full]
                    aviso_total_detachments = count_detachments_from_max_lat_series(aviso_det_by_ord)
                    print(f"[Detachment count] AVISO total detachments (full time series): {aviso_total_detachments}")
            plot_detachment_counts(
                detachment_count_data_ref or [],
                detachment_count_data_gliders or [],
                OUTPUT_DIR,
                ref_label=ref_label,
                gliders_label=gliders_label,
                aviso_total=aviso_total_detachments,
            )
            save_detachment_counts_to_netcdf(
                detachment_count_data_ref or [],
                detachment_count_data_gliders or [],
                OUTPUT_DIR,
                aviso_total=aviso_total_detachments,
            )
        else:
            print("Skipping detachment-count: no data computed (check HYCOM path).")

