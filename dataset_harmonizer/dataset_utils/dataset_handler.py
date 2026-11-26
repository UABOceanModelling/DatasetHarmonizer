import logging
from contextlib import contextmanager
from time import time
from typing import Dict

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@contextmanager
def _step(description: str):
    """Log the start and the duration of a pipeline step."""
    logger.info("Starting %s...", description)
    start = time()
    yield
    logger.info("%s took %.2f seconds", description.capitalize(), time() - start)


class DatasetHandler:
    def __init__(self, interpolation, combination, renaming, attributes_to_add, coords_to_add, slicing, keep):
        self.target_grid = interpolation.target_grid
        self.target_reference_variable = interpolation.target_reference_variable
        self.interpolation_variables = interpolation.variables
        self.combination = combination
        self.renaming_map = renaming
        self.region_of_interest = slicing.region_of_interest
        self.vars_to_keep = keep.variables
        self.coords_to_keep = keep.coords
        self.file_attributes_to_add = attributes_to_add.file
        self.variable_attributes_to_add = attributes_to_add.variables
        self.coords_to_add = coords_to_add

    def _select_subset_of_data(self, ds: xr.Dataset) -> xr.Dataset:
        variables_mapping = {key: slice(start, end) for key, (start, end) in self.region_of_interest.items()}
        ds = ds.isel(
            **variables_mapping,
        )

        return ds

    # NOTE: not called by the pipeline (and `mask_ds` is not configurable yet); kept as-is.
    def _add_mask(self, ds: xr.Dataset, mask_ds: xr.Dataset) -> xr.Dataset:
        mask = mask_ds["tmask"].fillna(0).astype(int)

        time_len = ds["time_counter"].size
        depth_len = ds["deptht"].size
        Y_len = ds["Y"].size
        X_len = ds["X"].size

        mask_broadcasted = xr.DataArray(
            np.broadcast_to(mask.values, (time_len, depth_len, Y_len, X_len)),
            dims=["time_counter", "deptht", "Y", "X"],
            coords={
                "time_counter": ds["time_counter"],
                "deptht": ds["deptht"],
                "Y": ds["Y"],
                "X": ds["X"],
            },
            attrs={"standard_name": "land_binary_mask"},
        )

        ds["land_binary_mask"] = mask_broadcasted
        ds["land_binary_mask"] = ds["land_binary_mask"].fillna(0).astype(int)

        return ds

    def _clean_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        # `keep` sections that are not configured mean "keep everything".
        if self.vars_to_keep is not None:
            vars_to_drop = [v for v in ds.data_vars if v not in self.vars_to_keep]
            ds = ds.drop_vars(vars_to_drop)

        if self.coords_to_keep is not None:
            coords_to_drop = [c for c in ds.coords if c not in self.coords_to_keep]
            ds = ds.drop_vars(coords_to_drop)

        return ds

    def interpolate(self, da: xr.DataArray, dimension_slice_start: dict, dimension_slice_end: dict) -> xr.DataArray:
        interpolated_array = 0.5 * (da.isel(**dimension_slice_start) + da.isel(**dimension_slice_end))

        return interpolated_array

    def _fill_values(self, arr: xr.DataArray, dimension: str, target_dim_size: int) -> xr.DataArray:
        times_to_add = target_dim_size - arr[dimension].size

        if times_to_add <= 0:
            return arr

        last_col = arr.isel({dimension: -1})
        repeats = [last_col] * times_to_add
        arr = xr.concat([arr, *repeats], dim=dimension)

        return arr

    def _add_attributes(self, ds: xr.Dataset) -> xr.Dataset:
        for attr, value in self.file_attributes_to_add.items():
            ds.attrs[attr] = value

        for var, attrs in self.variable_attributes_to_add.items():
            if var in ds:
                ds[var].attrs.update(attrs)
            else:
                logger.warning("Variable '%s' not found, skipping its attributes", var)

        return ds

    def _add_coords(self, ds: xr.Dataset) -> xr.Dataset:
        for var, coords in self.coords_to_add.items():
            if var in ds:
                for coord in coords:
                    ds[var] = ds[var].assign_coords({coord: ds[coord].data})
        return ds

    def _rename_variables(self, ds: xr.Dataset) -> xr.Dataset:
        renaming_map = {}
        for var, new_name in self.renaming_map.items():
            if (var not in ds) and (var not in ds.coords) and (var not in ds.dims):
                logger.warning("Variable '%s' not found in dataset for renaming. Skipping...", var)
                continue
            renaming_map[var] = new_name

        return ds.rename(renaming_map)

    def _run_interpolation(self, datasets: Dict[str, xr.Dataset]):
        target_ds = datasets[self.target_grid]
        datasets_to_pop = []
        for var, ds in datasets.items():
            if not var == self.target_grid and var in self.interpolation_variables:
                variable_dim, target_dim = next(iter(self.interpolation_variables[var].dims_mapping.items()))
                dimension_slice_start = {variable_dim: slice(0, -1)}
                dimension_slice_end = {variable_dim: slice(1, None)}

                interpolated_data_array = self.interpolate(
                    da=ds[self.interpolation_variables[var].source_var],
                    dimension_slice_start=dimension_slice_start,
                    dimension_slice_end=dimension_slice_end,
                )

                interpolated_data_array = self._fill_values(
                    interpolated_data_array, variable_dim, target_ds[target_dim].size
                )

                target_ds[var.lower()] = xr.DataArray(
                    interpolated_data_array,
                    dims=target_ds[self.target_reference_variable].dims,
                    coords=target_ds[self.target_reference_variable].coords,
                )

                datasets_to_pop.append(var)

        return datasets_to_pop

    def _unify_datasets(self, datasets: Dict[str, xr.Dataset]) -> xr.Dataset:
        combined_ds = datasets[self.target_grid]

        if not self.combination:
            return combined_ds

        for dataset in self.combination.keys():
            if dataset == self.target_grid:
                logger.warning("Target grid cannot be in combination section.")
                continue
            if dataset not in datasets:
                logger.warning("Dataset '%s' not found for combination. Skipping...", dataset)
                continue

            data_variables = self.combination[dataset]["variables"]

            for var in data_variables:
                if var not in datasets[dataset]:
                    raise ValueError(f"Variable '{var}' not found in dataset '{dataset}' for combination.")

                da = datasets[dataset][var]

                if data_variables[var] is not None and data_variables[var].get("dims_mapping"):
                    da = da.rename(data_variables[var]["dims_mapping"])

                combined_ds[var.lower()] = da

        return combined_ds

    def create_combined_dataset(self, datasets: Dict[str, xr.Dataset]) -> xr.Dataset:
        if self.target_grid not in datasets:
            raise ValueError(
                f"Target grid '{self.target_grid}' is not among the input datasets: {sorted(datasets)}"
            )

        # Work on copies: the caller's dict and the target dataset are left untouched.
        # (`copy()` is shallow, so no data is loaded or duplicated.)
        datasets = dict(datasets)
        datasets[self.target_grid] = datasets[self.target_grid].copy()

        with _step("interpolation"):
            datasets_to_pop = self._run_interpolation(datasets)

        for name in datasets_to_pop:
            datasets.pop(name)

        with _step("dataset unification"):
            ds = self._unify_datasets(datasets)

        with _step("renaming"):
            ds = self._rename_variables(ds)

        with _step("adding attributes"):
            ds = self._add_attributes(ds)

        with _step("adding coords"):
            ds = self._add_coords(ds)

        with _step("cleaning dataset"):
            ds = self._clean_dataset(ds)

        with _step("subset selection"):
            ds = self._select_subset_of_data(ds)

        return ds
