"""Small NEMO-like synthetic dataset used by the tests."""

import numpy as np
import xarray as xr
import yaml

T, D, Y, X = 2, 3, 4, 5
DEPTHS = np.array([1.0, 5.0, 10.0])


def build_inputs(root, seed=0):
    """Write *_gridT/U/V/W and *_icemod files (2 files each); return the raw arrays."""
    rng = np.random.default_rng(seed)
    tc = np.arange(T, dtype="float64")
    raw = {
        "u": rng.random((T, D, Y, X)),
        "v": rng.random((T, D, Y, X)),
        "w": rng.random((T, D, Y, X)),
        "temp": rng.random((T, D, Y, X)),
        "ice": rng.random((T, Y, X)),
    }

    def write(name, ds):
        for day in (1, 2):  # two files per grid -> exercises glob + by_coords combine
            ds.isel(time_counter=slice(day - 1, day)).to_netcdf(root / f"EXP_y2006m01d0{day}_{name}.nc")

    coords = {"time_counter": tc}
    write("gridT", xr.Dataset(
        {
            "votemper": (("time_counter", "deptht", "y_grid_T", "x_grid_T"), raw["temp"]),
            "vosaline": (("time_counter", "deptht", "y_grid_T", "x_grid_T"), raw["temp"] + 1),
        },
        coords={**coords, "deptht": DEPTHS,
                "nav_lat_grid_T": (("y_grid_T", "x_grid_T"), rng.random((Y, X))),
                "nav_lon_grid_T": (("y_grid_T", "x_grid_T"), rng.random((Y, X)))},
    ))
    dims4 = ("time_counter", "deptht", "y_grid_U", "x_grid_U")
    write("gridU", xr.Dataset({"vozocrtx": (dims4, raw["u"])}, coords={**coords, "deptht": DEPTHS}))
    write("gridV", xr.Dataset(
        {"vomecrty": (("time_counter", "deptht", "y_grid_V", "x_grid_V"), raw["v"])},
        coords={**coords, "deptht": DEPTHS}))
    write("gridW", xr.Dataset(
        {"vovecrtz": (("time_counter", "grid_W_3D_inner", "y_grid_W", "x_grid_W"), raw["w"])},
        coords=coords))
    write("icemod", xr.Dataset(
        {
            "iicevelu": (("time_counter", "y_grid_T", "x_grid_T"), raw["ice"]),
            "ileadfra": (("time_counter", "y_grid_T_2D_inner", "x_grid_T_2D_inner"), raw["ice"] * 2),
        },
        coords=coords))
    return raw


def config_dict(root):
    return {
        "input_data": {"data_paths": {
            "T": str(root / "EXP_y2006m01d0*_gridT.nc"),
            "U": str(root / "EXP_y2006m01d0*_gridU.nc"),
            "V": str(root / "EXP_y2006m01d0*_gridV.nc"),
            "W": str(root / "EXP_y2006m01d0*_gridW.nc"),
            "icemod": str(root / "EXP_y2006m01d0*_icemod.nc"),
        }},
        "transformations": {
            "interpolation": {
                "target_grid": "T",
                "target_reference_variable": "votemper",
                "variables": {
                    "U": {"source_var": "vozocrtx", "dims_mapping": {"x_grid_U": "x_grid_T"}},
                    "V": {"source_var": "vomecrty", "dims_mapping": {"x_grid_V": "x_grid_T"}},
                    "W": {"source_var": "vovecrtz", "dims_mapping": {"grid_W_3D_inner": "deptht"}},
                },
            },
            "combination": {"icemod": {"variables": {
                "iicevelu": None,
                "ileadfra": {"dims_mapping": {"y_grid_T_2D_inner": "y_grid_T",
                                              "x_grid_T_2D_inner": "x_grid_T"}},
            }}},
            "renaming": {"deptht": "depth", "x_grid_T": "X", "y_grid_T": "Y",
                         "nav_lat_grid_T": "lat", "nav_lon_grid_T": "lon",
                         "votemper": "temperature", "iicevelu": "ice_u", "ileadfra": "ice_concentration"},
            "attributes_to_add": {"file": {"title": "t"}, "variables": {"u": {"standard_name": "x_sea_water_velocity"}}},
            "coords_to_add": {"u": ["X", "Y"], "temperature": ["X", "Y"]},
            "slicing": {"region_of_interest": {"Y": [1, 3], "X": [1, 4]}},
            "keep": {"variables": ["u", "v", "w", "temperature", "ice_u", "ice_concentration"],
                     "coords": ["X", "Y", "depth", "time_counter", "lon", "lat"]},
        },
        "xarray": {"engine": "netcdf4", "reader": {"combine": "by_coords", "data_vars": "all"}},
        "output": {"directory": str(root / "out"), "file_name": "result.nc"},
    }


def write_config(root, cfg=None):
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg or config_dict(root), sort_keys=False))
    return path
