# Contributing

Thanks for contributing to `metrics_OSEs_GrASE`.

## Scope

This repo contains OSE metric workflows (92W focus), including:
- `metrics_OSEs_92W.py`
- `run_metrics_OSEs_92W.py`
- helper modules (`loop_current_contour.py`, `lce_contours.py`, `hycom_io.py`, `mhd.py`)

## Getting started

1. Fork the repo (or create a branch if you have write access).
2. Clone your fork/repo.
3. Create a branch for your change.

```bash
git checkout -b feature/my-change
```

## Running locally

Use either:
- direct script execution (`metrics_OSEs_92W.py`), or
- the wrapper (`run_metrics_OSEs_92W.py`) by editing `DEFAULT_ARGS`.

### HYCOM example

```bash
python metrics_OSEs_92W.py --hycom --timeseries --mean-std
```

### Non-HYCOM example

```bash
python metrics_OSEs_92W.py \
  --no-hycom \
  --timeseries --mean-std --timing-distribution \
  --netcdf-dir /path/ref \
  --netcdf-dir-gliders /path/gliders \
  --netcdf-pattern "*/*.nc" \
  --aviso-dir /path/AVISO/GRIDDED \
  --mdt /path/mdt.nc
```

See `README.md` for full option docs and naming conventions.

## Naming conventions (important)

- NetCDF file names should include one of:
  - `YYYYMMDD`
  - `YYYY-MM-DD`
  - `YYYY_MM_DD`
- For non-HYCOM multi-forecast runs, forecast start is inferred from parent folder names; fallback is `--forecast-start`.

## Code style

- Keep edits focused and minimal.
- Preserve behavior unless explicitly changing it.
- Keep 92W setting intact in `metrics_OSEs_92W.py` unless proposal says otherwise.
- Update `README.md` when CLI behavior or conventions change.

## Before opening a PR

- Run at least one representative command for the path you changed (HYCOM or non-HYCOM).
- Confirm no obvious runtime errors.
- Confirm generated outputs are not added to git.

Recommended ignored outputs include `*.nc`, `*.mp4`, `*.png`, cache directories, and temporary files.

## Pull request checklist

Please include:
- What changed
- Why it changed
- How to run/verify
- Any assumptions about input file/folder naming

## Git workflow

```bash
git add .
git commit -m "Short, descriptive message"
git push origin feature/my-change
```

Open a PR to `main` and request review.

## Reporting issues

If something fails, include:
- exact command used
- mode (`--hycom` or `--no-hycom`)
- relevant path pattern (`--netcdf-pattern`)
- key log lines/error message
