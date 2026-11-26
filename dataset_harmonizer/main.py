import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

from dataset_harmonizer.config import ConfigError, ConfigReader
from dataset_harmonizer.dataset_utils import DatasetHandler
from dataset_harmonizer.reader import DatasetReader, close_datasets
from dataset_harmonizer.utils import create_path, save_dataset

logger = logging.getLogger("dataset_harmonizer")

DEFAULT_CONFIG_FILE = "config.yaml"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def read_data(config: ConfigReader):
    dr = DatasetReader(
        xarray_parameters=config.xarray_parameters,
    )
    dh = DatasetHandler(
        interpolation=config.transformations.interpolation,
        combination=config.transformations.combination,
        renaming=config.transformations.renaming,
        attributes_to_add=config.transformations.attributes_to_add,
        coords_to_add=config.transformations.coords_to_add,
        slicing=config.transformations.slicing,
        keep=config.transformations.keep,
    )
    datasets = dr.read_multi_files(config.input_data.data_paths)
    return dh, datasets


def main(config_file: str = DEFAULT_CONFIG_FILE) -> str:
    """Harmonize the datasets described in `config_file`; returns the output file path."""
    config = ConfigReader.from_yaml(config_file)
    dh, datasets = read_data(config)

    try:
        combined_ds = dh.create_combined_dataset(datasets)

        output_path = create_path(config.output.directory, config.output.file_name)
        save_dataset(combined_ds, output_path=output_path)
    finally:
        close_datasets(datasets)

    return output_path


def cli(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Harmonize oceanographic NetCDF datasets.")
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG_FILE,
        help="configuration YAML file (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    try:
        main(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        sys.exit(1)
