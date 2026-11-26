import numpy as np
import pytest
import xarray as xr

from dataset_harmonizer.config import ConfigReader
from dataset_harmonizer.main import main, read_data
from dataset_harmonizer.reader import close_datasets

from helpers import build_inputs, config_dict, write_config


@pytest.fixture
def run(tmp_path):
    raw = build_inputs(tmp_path)
    config = write_config(tmp_path)
    out = main(str(config))
    with xr.open_dataset(out) as ds:
        yield raw, ds.load()


def test_output_structure(run):
    _, ds = run
    assert set(ds.data_vars) == {"temperature", "u", "v", "w", "ice_u", "ice_concentration"}
    assert ds.sizes == {"time_counter": 2, "depth": 3, "Y": 2, "X": 3}
    assert ds.attrs["title"] == "t"
    assert ds["u"].attrs["standard_name"] == "x_sea_water_velocity"
    assert {"lat", "lon", "X", "Y", "depth"} <= set(ds.coords)


def test_u_is_averaged_along_x_and_last_column_repeated(run):
    raw, ds = run
    u = raw["u"]
    avg = 0.5 * (u[..., :-1] + u[..., 1:])
    expected = np.concatenate([avg, avg[..., -1:]], axis=-1)  # last column repeated
    np.testing.assert_allclose(ds["u"].values, expected[:, :, 1:3, 1:4])


def test_w_is_averaged_along_depth(run):
    raw, ds = run
    w = raw["w"]
    avg = 0.5 * (w[:, :-1] + w[:, 1:])
    expected = np.concatenate([avg, avg[:, -1:]], axis=1)
    np.testing.assert_allclose(ds["w"].values, expected[:, :, 1:3, 1:4])


def test_non_interpolated_variables_are_passed_through(run):
    raw, ds = run
    np.testing.assert_allclose(ds["temperature"].values, raw["temp"][:, :, 1:3, 1:4])
    np.testing.assert_allclose(ds["ice_concentration"].values, (raw["ice"] * 2)[:, 1:3, 1:4])


def test_inputs_are_not_mutated(tmp_path):
    build_inputs(tmp_path)
    config = ConfigReader.from_yaml(write_config(tmp_path))
    dh, datasets = read_data(config)
    try:
        before = {name: set(ds.variables) for name, ds in datasets.items()}
        dh.create_combined_dataset(datasets)
        assert set(datasets) == {"T", "U", "V", "W", "icemod"}
        assert {name: set(ds.variables) for name, ds in datasets.items()} == before
    finally:
        close_datasets(datasets)


def test_missing_target_grid_is_a_clear_error(tmp_path):
    build_inputs(tmp_path)
    cfg = config_dict(tmp_path)
    cfg["transformations"]["interpolation"]["target_grid"] = "NOPE"
    config = ConfigReader.from_yaml(write_config(tmp_path, cfg))
    dh, datasets = read_data(config)
    try:
        with pytest.raises(ValueError, match="Target grid 'NOPE'"):
            dh.create_combined_dataset(datasets)
    finally:
        close_datasets(datasets)


def test_renaming_map_is_not_consumed(tmp_path):
    build_inputs(tmp_path)
    cfg = config_dict(tmp_path)
    cfg["transformations"]["renaming"]["does_not_exist"] = "whatever"
    config = ConfigReader.from_yaml(write_config(tmp_path, cfg))
    dh, datasets = read_data(config)
    try:
        dh.create_combined_dataset(datasets)
        dh.create_combined_dataset(datasets)  # second run must behave the same
        assert "does_not_exist" in dh.renaming_map
    finally:
        close_datasets(datasets)


def test_keep_none_keeps_everything(tmp_path):
    build_inputs(tmp_path)
    cfg = config_dict(tmp_path)
    del cfg["transformations"]["keep"]
    out = main(str(write_config(tmp_path, cfg)))
    with xr.open_dataset(out) as ds:
        assert "salinity" in ds or "vosaline" in ds
