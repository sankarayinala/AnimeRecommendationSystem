import os
from typing import Any, Dict

import pandas as pd
import yaml

from src.custom_exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


def read_yaml_file(file_path: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(file_path):
            raise CustomException(f"YAML file not found at path: {file_path}")

        if not os.path.isfile(file_path):
            raise CustomException(f"Path exists but is not a file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

            if config is None:
                raise CustomException(f"YAML file is empty or invalid: {file_path}")

            logger.info(f"Successfully read YAML file: {file_path}")
            return config

    except yaml.YAMLError as ye:
        logger.error(f"Invalid YAML format in {file_path}: {ye}")
        raise CustomException(f"Invalid YAML format in {file_path}: {ye}") from ye
    except CustomException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading YAML file {file_path}: {e}")
        raise CustomException(f"Failed to read YAML file {file_path}: {e}") from e


def load_data(path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(path):
            raise CustomException(f"Data file not found at path: {path}")

        if not os.path.isfile(path):
            raise CustomException(f"Path exists but is not a file: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext == ".parquet":
            data = pd.read_parquet(path)
        elif ext in {".csv", ".txt"}:
            data = pd.read_csv(path)
        else:
            raise CustomException(f"Unsupported file format: {ext}")

        logger.info(f"Successfully loaded data from: {path}")
        return data

    except pd.errors.EmptyDataError as ede:
        logger.error(f"Data file is empty: {path}: {ede}")
        raise CustomException(f"Data file is empty: {path}: {ede}") from ede
    except pd.errors.ParserError as pe:
        logger.error(f"Error parsing data file {path}: {pe}")
        raise CustomException(f"Error parsing data file {path}: {pe}") from pe
    except CustomException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data from {path}: {e}")
        raise CustomException(f"Failed to load data from {path}: {e}") from e


def save_data(df: pd.DataFrame, path: str, index: bool = False) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ext = os.path.splitext(path)[1].lower()

        if ext == ".parquet":
            df.to_parquet(path, index=index)
        elif ext in {".csv", ".txt"}:
            df.to_csv(path, index=index)
        else:
            raise CustomException(f"Unsupported file format: {ext}")

        logger.info(f"Successfully saved data to: {path}")

    except Exception as e:
        logger.error(f"Failed to save data to {path}: {e}")
        raise CustomException(f"Failed to save data to {path}: {e}") from e