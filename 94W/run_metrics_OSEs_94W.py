#!/usr/bin/env python3
"""
Run metrics_OSEs.py. Edit the DEFAULT_ARGS list below to match your data and desired
outputs, then run: python run_metrics_OSEs.py
"""
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_SCRIPT = os.path.join(SCRIPT_DIR, "metrics_OSEs_94W.py")

# Edit these to your needs. Uncomment and set values as desired.
# All possible arguments (commented out):
#
# --- Mode ---
#    "--hycom",                    # Use HYCOM (REF + GLIDERS) paths
#    "--no-hycom",                 # Use NetCDF model paths instead of HYCOM
#
# --- Actions / outputs ---
#    "--timeseries",               # Plot MHD timeseries (0–90 days)
#    "--mean-std",                 # Plot mean±std MHD
#    "--timing-distribution",      # Plot LCE timing distribution (histograms)
#    "--animate",                  # Create animation for one forecast
#    "--animate-all",              # Create animations for all forecasts
#    "--animate-forecast", "YYYY-MM-DD",   # Only animate this forecast (with --animate / --animate-all)
#
# --- Plot labels ---
#    "--model-label", "Model",
#    "--ref-label", "REF",
#    "--gliders-label", "GLIDERS",
#
# --- Paths (AVISO / MDT) ---
#    "--aviso-dir", "/path/to/AVISO/GRIDDED",
#    "--mdt-path", "/path/to/mdt.nc",
#
# --- HYCOM paths (when --hycom) ---
#    "--hycom-ref-dir", "/path/to/HYCOM_REF",
#    "--hycom-gliders-dir", "/path/to/HYCOM_GLIDERS",
#    "--hycom-grid-file", "/path/to/regional.grid.a",
#
# --- NetCDF paths (when --no-hycom) ---
#    "--netcdf-ref-dir", "/path/to/NetCDF_REF",
#    "--netcdf-gliders-dir", "/path/to/NetCDF_GLIDERS",
# Optional when using NetCDF: set plot/legend labels to match (e.g. --ref-label "NetCDF_REF", --gliders-label "NetCDF_GLIDERS")
#
# --- NetCDF variable names (when --no-hycom). Defaults: longitude, latitude, ssh. Set if your NetCDF uses different names. ---
#    "--netcdf-lon-var", "longitude",       # e.g. "nav_lon", "lon"
#    "--netcdf-lat-var", "latitude",       # e.g. "nav_lat", "lat"
#    "--netcdf-ssh-var", "ssh",            # e.g. "zos", "sossheig", "height"
#    "--netcdf-ssh-scale", 1.0,             # scale factor (1.0 = meters, 0.1 = dm)
#
# --- NetCDF grid: regular vs Mercator (when --no-hycom) ---
# For NetCDF, demean is either:
#   - Regular (unweighted): default if no grid file is given.
#   - Mercator (area-weighted): use if your NetCDF is on a Mercator grid and you have a grid file with cell sizes (dx, dy).
# If metrics_OSEs.py supports these, uncomment and set:
#    "--grid-netcdf", "/path/to/grid_with_pscx_pscy.nc",   # NetCDF with cell sizes for area-weighted demean
#    "--mercator",                                         # Use area-weighted demean (requires --grid-netcdf)
#    "--grid-dx-var", "pscx",                              # Variable name for cell x-size (default pscx)
#    "--grid-dy-var", "pscy",                              # Variable name for cell y-size (default pscy)
#
# --- Environment variables (alternative to passing paths above) ---
# HYCOM mode:  HYCOM_REF_DIR, HYCOM_GLIDERS_DIR, HYCOM_GRID_FILE
# NetCDF mode: NETCDF_REF_DIR, NETCDF_GLIDERS_DIR
# Output:      MHD_OUTPUT_DIR (default output dir for plots and .nc files)
#
# --- Output ---
#    "--output-dir", "/path/to/output",
#    "--max-lead-days", "90",
#
DEFAULT_ARGS = [
    "--hycom",
    "--animate",
    "--animate-all",
    #"--timeseries",
    #"--mean-std",
    #"--timing-distribution",
]


def main():
    if not os.path.isfile(METRICS_SCRIPT):
        print(f"Error: metrics_OSEs.py not found at {METRICS_SCRIPT}")
        sys.exit(1)
    cmd = [sys.executable, METRICS_SCRIPT] + DEFAULT_ARGS
    print("Running:", " ".join(cmd))
    sys.stdout.flush()
    rc = subprocess.call(cmd, cwd=SCRIPT_DIR)
    sys.exit(rc)


if __name__ == "__main__":
    main()
