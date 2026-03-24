"""
HYCOM-specific I/O: read SSH and grid from HYCOM binary (.a) archive and grid files.

This is the only module that reads HYCOM native format. Other groups using different
models (e.g. NetCDF) can omit this file and provide their own loader that returns
(lon, lat, ssh) with the same interface.

Supports both archv and archm: SSH variable index is read from the archive .b file.
Required dependency: io_py (read_hycom_grid, sub_var2), and info (read_field_names) for archm.
"""

from typing import Tuple, List
import os
import numpy as np

try:
    from io_py import read_hycom_grid, sub_var2
    _HAS_IO = True
except ImportError:
    _HAS_IO = False

# SSH field names used in HYCOM archive .b files (archv vs archm may differ)
_SSH_FIELD_NAMES = ("srfhgt", "srfh", "montg1", "montg", "steric", "surf_el", "ssh", "SSH")


def _ssh_variable_index(archive_path: str) -> Tuple[int, str, List[str]]:
    """Return (1-based variable index, field name, all names) for SSH from archive .b file. Falls back to (2, '', []) if not found."""
    try:
        from info import read_field_names
    except ImportError:
        return 2, "", []
    base = archive_path.rstrip(".a").rstrip(".b")
    b_path = base + ".b"
    if not os.path.isfile(b_path):
        return 2, "", []
    names = read_field_names(b_path)
    for candidate in _SSH_FIELD_NAMES:
        if candidate in names:
            return names.index(candidate) + 1, candidate, names
    return 2, "", names


def load_hycom_ssh_and_grid(
    archv_file: str,
    grid_file: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SSH and 2D lon/lat from HYCOM archive (.a) and grid (.a) files.

    Parameters
    ----------
    archv_file : str
        Path to HYCOM archive file (e.g. 080_archv.2010_001_00.a).
    grid_file : str
        Path to HYCOM grid file (e.g. regional.grid.a). Reads plon, plat.

    Returns
    -------
    lon, lat, ssh : 2D arrays (jdm, idm). SSH is in HYCOM archive units (usually decimetres);
        caller should convert to meters if needed.
    """
    if not _HAS_IO:
        raise ImportError(
            "hycom_io requires io_py (read_hycom_grid, sub_var2) to read HYCOM .a files. "
            "Other groups: replace this module with your own loader that returns (lon, lat, ssh)."
        )
    grid_data = read_hycom_grid(grid_file, ["plon", "plat"])
    lon = np.asarray(grid_data["plon"])
    lat = np.asarray(grid_data["plat"])
    jdm, idm = lon.shape[0], lon.shape[1]
    ivar, ssh_field_name, b_file_names = _ssh_variable_index(archv_file)
    ssh = sub_var2(archv_file, idm, jdm, ivar)
    ssh = np.asarray(ssh)
    return lon, lat, ssh
