import logging

logger = logging.getLogger(__name__)


def save_dataset(ds, output_path):
    logger.info("Saving combined dataset to %s", output_path)
    ds.to_netcdf(output_path)
    logger.info("Combined dataset saved")
