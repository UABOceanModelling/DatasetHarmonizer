import glob
import logging
from pathlib import Path
from typing import Dict, Mapping

import xarray as xr

logger = logging.getLogger(__name__)


class DatasetReader:
    def __init__(self, xarray_parameters):
        self.xarray_parameters = xarray_parameters

    def read_multi_files(self, data_paths: Mapping[str, str]) -> Dict[str, xr.Dataset]:
        """Open every entry of `data_paths` ({name: path or glob pattern}) as one dataset."""
        datasets: Dict[str, xr.Dataset] = {}
        engine = self.xarray_parameters.engine
        reader_params = self.xarray_parameters.reader

        try:
            for name, pattern in data_paths.items():
                # sorted: glob order is arbitrary, keep runs deterministic
                paths = sorted(glob.glob(str(Path(pattern).expanduser())))
                if not paths:
                    raise FileNotFoundError(
                        f"No files found for variable '{name}' with pattern '{pattern}'"
                    )
                logger.info("Opening '%s': %d file(s)", name, len(paths))
                datasets[name] = xr.open_mfdataset(
                    paths,
                    engine=engine,
                    combine=reader_params.combine,
                    data_vars=reader_params.data_vars,
                )
        except Exception:
            close_datasets(datasets)
            raise

        return datasets

    def read_single_file(self, path: str) -> xr.Dataset:
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"File not found at path: {path}")
        return xr.open_dataset(path, engine=self.xarray_parameters.engine)


def close_datasets(datasets: Mapping[str, xr.Dataset]) -> None:
    for ds in datasets.values():
        ds.close()
