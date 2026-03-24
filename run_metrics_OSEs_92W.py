#!/usr/bin/env python3
"""
Run metrics_OSEs_92W.py. Edit DEFAULT_ARGS below to match your data and desired
outputs, then run: python run_metrics_OSEs_92W.py
"""
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_SCRIPT = os.path.join(SCRIPT_DIR, "metrics_OSEs_92W.py")

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
#    "--model-label-gliders", "Model_GLIDERS",
#
# --- Paths (AVISO / MDT) ---
#    "--aviso-dir", "/path/to/AVISO/GRIDDED",
#    "--mdt", "/path/to/mdt.nc",
#
# --- HYCOM paths (when --hycom) ---
# Paths are configured inside metrics_OSEs_92W.py:
#    HYCOM_BASE_FORECAST_DIR, HYCOM_SINGLE_REF_SUBDIR, HYCOM_SINGLE_GLIDERS_SUBDIR, HYCOM_GRID_FILE
#
# --- NetCDF paths (when --no-hycom) ---
#    "--netcdf-dir", "/path/to/NetCDF_REF",
#    "--netcdf-dir-gliders", "/path/to/NetCDF_GLIDERS",
#    "--netcdf-pattern", "*/*.nc"
# Optional when using NetCDF: set plot labels to match (e.g. --model-label "NetCDF_REF", --model-label-gliders "NetCDF_GLIDERS")
#
# --- NetCDF variable names (when --no-hycom). Defaults: longitude, latitude, ssh. Set if your NetCDF uses different names. ---
#    "--lon-var", "longitude",              # e.g. "nav_lon", "lon"
#    "--lat-var", "latitude",               # e.g. "nav_lat", "lat"
#    "--ssh-var", "ssh",                    # e.g. "zos", "sossheig", "height"
#    "--ssh-scale", "1.0",                  # scale factor (1.0 = meters, 0.1 = dm)
#
# --- NetCDF grid: regular vs Mercator (when --no-hycom) ---
# For NetCDF, demean is either:
#   - Regular (unweighted): default if no grid file is given.
#   - Mercator (area-weighted): use if your NetCDF is on a Mercator grid and you have a grid file with cell sizes (dx, dy).
# If metrics_OSEs_92W.py supports these, uncomment and set:
#    "--grid-netcdf", "/path/to/grid_with_pscx_pscy.nc",   # NetCDF with cell sizes for area-weighted demean
#    "--mercator",                                         # Use area-weighted demean (requires --grid-netcdf)
#    "--grid-dx-var", "pscx",                              # Variable name for cell x-size (default pscx)
#    "--grid-dy-var", "pscy",                              # Variable name for cell y-size (default pscy)
#
# --- Environment variables (alternative to passing paths above) ---
# Output:      MHD_OUTPUT_DIR (default output dir for plots and .nc files)
#
# --- Output ---
#    "-o", "animation_mhd_with_mean_lce.mp4",
#    "--forecast-start", "YYYY-MM-DD",      # Optional fallback for --no-hycom (if folder inference fails)
#    "--max-forecasts", "10",               # HYCOM only
#
#DEFAULT_ARGS = [
#    "--hycom",
#    #"--animate",
#    #"--animate-all",
#    "--timeseries",
#    "--mean-std",
#    #"--timing-distribution",
#]

DEFAULT_ARGS = [
    "--no-hycom",
    #"--timeseries",
    #"--mean-std",
    "--timing-distribution",
    "--animate","--animate-all",
    "--model-label", "MODEL_no_grace",
    "--model-label-gliders", "MODEL_grace",
    "--aviso-dir", "/gpfs/research/coaps/abozec/HYCOM2.3-TSIS/AVISO/GRIDDED",
    "--mdt", "/gpfs/research/coaps/abozec/HYCOM2.3-TSIS/AVISO/clim/mdt_cnes_cls22_global.nc",
    "--netcdf-dir", "/gpfs/research/coaps/nntaganou/OSEs_GrASE/workdir/netcdf_test_nograse",
    "--netcdf-dir-gliders", "/gpfs/research/coaps/nntaganou/OSEs_GrASE/workdir/netcdf_test_grase",
    "--netcdf-pattern", "*/*.nc",
    "--lon-var", "Longitude",             # e.g. "nav_lon", "lon"
    "--lat-var", "Latitude",              # e.g. "nav_lat", "lat"
    "--ssh-var", "ssh",
    ]


def main():
    if not os.path.isfile(METRICS_SCRIPT):
        print(f"Error: metrics_OSEs_92W.py not found at {METRICS_SCRIPT}")
        sys.exit(1)
    cmd = [sys.executable, METRICS_SCRIPT] + DEFAULT_ARGS
    print("Running:", " ".join(cmd))
    sys.stdout.flush()
    rc = subprocess.call(cmd, cwd=SCRIPT_DIR)
    sys.exit(rc)


if __name__ == "__main__":
    main()
