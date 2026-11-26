# Dataset Harmonizer

A Python tool for harmonizing oceanographic NetCDF datasets.
It interpolates variables to a common grid, combines multiple datasets, renames variables and coordinates, adds attributes and subsets data for a region of interest. It is particularly useful for NEMO ocean model outputs and for preparing data for tools like OpenDrift (see the companion project [Oil-Spill](https://github.com/novaisc/Oil-Spill)).

---

## Features

- Interpolates variables from different grids (U, V, W) to a common target grid (T grid by default).
- Combines multiple datasets into a single harmonized dataset.
- Renames variables and coordinates to standardized names.
- Adds file and variable attributes for CF-compliance and documentation.
- Subsets data to a region of interest.
- Fully configuration-driven using a YAML file.

---

## Installation

Python >= 3.13.

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"      # or: make install   (pip install -r requirements.txt also works)
```

## Usage

1. Prepare your input NetCDF datasets.
2. Edit `config.yaml` (paths, variables and transformations). The code does not need to change to run on other datasets.
3. Run from the repository root:

```bash
python app.py                 # uses config.yaml
python app.py my_config.yaml  # another configuration file
make run                      # same as `python app.py`
make run-bg                   # in the background, log in output.txt
```

4. The harmonized dataset is written to `output.directory` / `output.file_name`.

## Configuration

Transformations are applied in this order: **interpolation → combination → renaming → attributes → coordinates → cleaning (`keep`) → slicing**. Each step receives the result of the previous one, so e.g. names in `attributes_to_add` and `keep` are the names *after* renaming.

| Section | Purpose |
| --- | --- |
| `input_data.data_paths` | Dataset name -> file path or glob pattern (required). |
| `transformations.interpolation` | `target_grid` (a key of `data_paths`), `target_reference_variable` and `variables` (`source_var` and `dims_mapping` per grid to interpolate). |
| `transformations.combination` | Other datasets/variables merged into the target dataset, optionally with `dims_mapping`. |
| `transformations.renaming` | Variable/coordinate/dimension renames. Names that do not exist are skipped with a warning. |
| `transformations.attributes_to_add` | `file` and `variables` attributes. |
| `transformations.coords_to_add` | Coordinates assigned to each variable. |
| `transformations.slicing` | `region_of_interest`: `{dim: [start, end]}` (index based). |
| `transformations.keep` | `variables` and `coords` to keep. If omitted, everything is kept. |
| `xarray` | `engine` (default `netcdf4`) and `reader.combine` / `reader.data_vars` for `open_mfdataset`. |
| `output` | `directory` and `file_name` of the harmonized file. |

Everything except `input_data` and `transformations.interpolation` is optional. Unknown keys and malformed sections are rejected with an error naming the file and the section.

## Development

```bash
make test        # or: python -m pytest
```

The tests build a small synthetic NEMO-like dataset, so they need no real data.
