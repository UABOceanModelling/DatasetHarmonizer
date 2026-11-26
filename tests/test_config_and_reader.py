import pytest
import xarray as xr

from dataset_harmonizer.config import ConfigError, ConfigReader
from dataset_harmonizer.config.config_reader import XarrayParameters
from dataset_harmonizer.reader import DatasetReader, close_datasets
from dataset_harmonizer.utils import create_path

from helpers import config_dict, write_config


def test_real_example_config_loads():
    from pathlib import Path

    cfg = ConfigReader.from_yaml(Path(__file__).resolve().parent.parent / "config.yaml")
    assert cfg.transformations.interpolation.target_grid == "T"
    assert "U" in cfg.transformations.interpolation.variables


def test_minimal_config_gets_defaults(tmp_path):
    cfg = {
        "input_data": {"data_paths": {"T": "x.nc"}},
        "transformations": {"interpolation": {"target_grid": "T"}},
    }
    c = ConfigReader.from_yaml(write_config(tmp_path, cfg))
    assert c.xarray_parameters.engine == "netcdf4"
    assert c.xarray_parameters.reader.combine == "by_coords"
    assert c.transformations.keep.variables is None
    assert c.transformations.renaming == {}
    assert c.output.file_name == "harmonized.nc"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.pop("input_data"), "input_data"),
        (lambda c: c["input_data"].update(data_paths={}), "data_paths"),
        (lambda c: c.pop("transformations"), "transformations"),
        (lambda c: c["transformations"].pop("interpolation"), "interpolation"),
        (lambda c: c["transformations"].update(bogus=1), "unknown keys"),
        (lambda c: c["transformations"]["interpolation"]["variables"]["U"].pop("dims_mapping"), "dims_mapping|invalid"),
        (lambda c: c["transformations"]["interpolation"].pop("target_reference_variable"), "target_reference_variable"),
        (lambda c: c["transformations"]["slicing"]["region_of_interest"].update(Y=[1]), r"\[start, end\]"),
        (lambda c: c["transformations"]["combination"].update(bad={}), "variables"),
        (lambda c: c["output"].update(bogus=1), "unknown keys"),
    ],
)
def test_invalid_configs(tmp_path, mutate, message):
    cfg = config_dict(tmp_path)
    mutate(cfg)
    with pytest.raises(ConfigError, match=message):
        ConfigReader.from_yaml(write_config(tmp_path, cfg))


def test_reader_reports_missing_files_and_closes_opened(tmp_path):
    xr.Dataset({"v": ("t", [1, 2])}).to_netcdf(tmp_path / "a_1.nc")
    reader = DatasetReader(XarrayParameters())
    with pytest.raises(FileNotFoundError, match="nothing"):
        reader.read_multi_files({"A": str(tmp_path / "a_*.nc"), "B": str(tmp_path / "nothing_*.nc")})


def test_reader_combines_globbed_files(tmp_path):
    for i in (1, 2):
        xr.Dataset({"v": ("t", [i])}, coords={"t": [i]}).to_netcdf(tmp_path / f"a_{i}.nc")
    datasets = DatasetReader(XarrayParameters()).read_multi_files({"A": str(tmp_path / "a_*.nc")})
    assert datasets["A"]["v"].values.tolist() == [1, 2]
    close_datasets(datasets)


def test_create_path(tmp_path):
    path = create_path(tmp_path / "a" / "b", "f.nc")
    assert path.endswith("a/b/f.nc") and (tmp_path / "a" / "b").is_dir()
