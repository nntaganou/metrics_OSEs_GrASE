# OSEs GrASE Metrics (92W)

This directory contains the 92W configuration of the OSE metrics workflow.

Main script:
- `metrics_OSEs_92W.py`

Runner helper:
- `run_metrics_OSEs_92W.py`

## What the script produces

Depending on flags, it can produce:
- MHD animation MP4 (`--animate`)
- MHD time series plot (`--timeseries`)
- Mean +- std MHD vs lead time (`--mean-std`)
- Timing distribution histograms (`--timing-distribution`)
- Cached NetCDF outputs in `MHD_OUTPUT_DIR` (or current directory):
  - `mhd_OSEs.nc`
  - `lce_timing_OSEs.nc`

## 92W setting

In this file, 92W is active:
- `MHD_LON_MIN = -92.0`

This affects:
- LCE inclusion/exclusion for MHD
- MHD contour clipping west boundary
- map west boundary in animation (set from `MHD_LON_MIN`)

To switch back to 90W:
- comment `-92.0` and enable `-90.0` line in `metrics_OSEs_92W.py`.

## CLI options

### Modes
- `--hycom`: use HYCOM archive inputs (paths configured inside script)
- `--no-hycom`: use NetCDF inputs (paths passed by arguments)

Exactly one mode is required.

### Actions / outputs
- `--animate`: create animation MP4
- `--animate-all`: with `--animate`, create one MP4 per forecast/group
- `--timeseries`: create `mhd_timeseries_all_forecasts.png`
- `--mean-std`: create mean+-std plot(s)
- `--timing-distribution`: create timing histograms

At least one action flag is required.

### NetCDF mode arguments (`--no-hycom`)
- `--netcdf-dir PATH`: REF directory (required)
- `--netcdf-dir-gliders PATH`: GLIDERS directory (optional)
- `--netcdf-pattern GLOB`: file pattern under each netcdf dir (default `*.nc`)
- `--aviso-dir PATH`: AVISO gridded directory (required)
- `--mdt PATH`: MDT file path (required)
- `--forecast-start YYYY-MM-DD`: optional fallback forecast start (used if folder inference fails)
- `--lon-var NAME`: longitude variable name (default `longitude`)
- `--lat-var NAME`: latitude variable name (default `latitude`)
- `--ssh-var NAME`: ssh variable name (default `ssh`)
- `--ssh-scale FLOAT`: scale to meters (default `1.0`)
- `--model-label TEXT`: label for REF (default `Model`)
- `--model-label-gliders TEXT`: label for GLIDERS (default `Model_GLIDERS`)
- `--grid-netcdf PATH`: grid NetCDF with cell size vars (for Mercator weighted demean)
- `--mercator`: enable area-weighted demean
- `--grid-dx-var NAME`: dx var (default `pscx`)
- `--grid-dy-var NAME`: dy var (default `pscy`)

### HYCOM mode arguments (`--hycom`)
- HYCOM paths are configured in script constants (`HYCOM_BASE_FORECAST_DIR`, `HYCOM_SINGLE_*`, `HYCOM_GRID_FILE`, etc.)
- `--max-forecasts N`: limit number of HYCOM forecasts for testing

### General arguments
- `-o`, `--output FILE`: MP4 output path/name for animation base (default `animation_mhd_with_mean_lce.mp4`)

### Environment variable
- `MHD_OUTPUT_DIR`: output directory for plots, MP4 (if relative), and cached NetCDF files.

If unset, output defaults to current working directory (`.`).

## Naming conventions

### NetCDF file naming (date extraction)
For each NetCDF input file, date is parsed from the file basename and normalized to `YYYYMMDD`.
Accepted patterns in filename:
- `YYYYMMDD` (example: `model_20250415.nc`)
- `YYYY-MM-DD` (example: `model_2025-04-15.nc`)
- `YYYY_MM_DD` (example: `model_2025_04_15.nc`)

Only `.nc` files are processed.
`mhd_OSEs.nc` is excluded from model input scanning.

### Forecast-start inference (non-HYCOM)
Forecast start is inferred from the **parent folder name** of each file.
Accepted parent folder patterns:
- `YYYYMMDD`
- `YYYY-MM-DD`
- `YYYY_MM_DD`
- `Apr15-2025`, `april15_2025`, `Apr_15_2025`, etc. (month name + day + year)

If parent folder inference fails, `--forecast-start` (if provided) is used as fallback.

### Grouping behavior for `--no-hycom --animate --animate-all`
- Files are grouped by immediate parent folder name.
- One animation is generated per group.
- Output file names are `<output_stem>_<group>.mp4`.

Example:
- REF files in `/data/ref/april15_2025/*.nc`
- GLIDERS in `/data/gliders/april15_2025/*.nc`
- Output: `animation_mhd_with_mean_lce_april15_2025.mp4`

## Important behavior notes

- `--animate-all` works for both HYCOM and non-HYCOM.
- In non-HYCOM, when forecast starts are inferred per file, `--mean-std` and `--timing-distribution` can run across all forecasts in one run (similar to HYCOM all-forecast behavior).
- If no REF files are found for processing, the script exits with an error.
- For non-HYCOM recursive structures, set pattern accordingly, e.g. `--netcdf-pattern "*/*.nc"`.

## Usage examples

### 1) HYCOM: all-forecast static products
```bash
python metrics_OSEs_92W.py --hycom --timeseries --mean-std --timing-distribution
```

### 2) HYCOM: animation for all forecasts
```bash
python metrics_OSEs_92W.py --hycom --animate --animate-all -o animation_mhd_with_mean_lce.mp4
```

### 3) Non-HYCOM: one run across all forecast folders
```bash
python metrics_OSEs_92W.py   --no-hycom   --animate --animate-all   --timeseries --mean-std --timing-distribution   --netcdf-dir /path/ref   --netcdf-dir-gliders /path/gliders   --netcdf-pattern "*/*.nc"   --aviso-dir /path/AVISO/GRIDDED   --mdt /path/mdt.nc   --lon-var Longitude --lat-var Latitude --ssh-var ssh   --model-label REF --model-label-gliders GLIDERS   -o animation_mhd_with_mean_lce.mp4
```

### 4) Non-HYCOM with fallback forecast-start
```bash
python metrics_OSEs_92W.py   --no-hycom --timeseries --mean-std   --netcdf-dir /path/ref --netcdf-pattern "*.nc"   --aviso-dir /path/AVISO/GRIDDED --mdt /path/mdt.nc   --forecast-start 2025-04-15
```

## Runner helper

`run_metrics_OSEs_92W.py` is a convenience wrapper that calls `metrics_OSEs_92W.py` with `DEFAULT_ARGS`.

To use it:
1. Edit `DEFAULT_ARGS` in `run_metrics_OSEs_92W.py`.
2. Run:
```bash
python run_metrics_OSEs_92W.py
```

The wrapper does not add extra logic; it forwards arguments to `metrics_OSEs_92W.py`.
