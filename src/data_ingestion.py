"""
Data Ingestion Module

Responsible for downloading raw data files from Google Cloud Storage and
preparing train/test splits for the animelist dataset.
"""

import os
import sys
import warnings
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import RAW_DIR, TRAIN_FILE_PATH, TEST_FILE_PATH
from utils.common_functions import read_yaml_file

warnings.filterwarnings("ignore", category=UserWarning, module="google.auth")

logger = get_logger("data_ingestion")


class DataIngestion:
    """
    Handles data ingestion from GCS and prepares train/test datasets.
    Always processes the full dataset.
    """

    def __init__(self, config):
        try:
            if isinstance(config, (str, os.PathLike)):
                logger.info(f"Loading config from: {config}")
                config = read_yaml_file(config)

            self.config = config
            ingestion_cfg = self.config["data_ingestion"]

            self.bucket_name = ingestion_cfg["bucket_name"]
            self.bucket_file_names = ingestion_cfg["bucket_file_names"]
            self.train_ratio = ingestion_cfg.get("train_ratio", 0.8)

            self.gcp_credentials = ingestion_cfg.get("gcp_credentials")
            self.gcp_credentials_env = ingestion_cfg.get("gcp_credentials_env")

            os.makedirs(RAW_DIR, exist_ok=True)

            logger.info("DataIngestion initialized")
            logger.info(f"Bucket: {self.bucket_name}")
            logger.info(f"Files: {self.bucket_file_names}")
            logger.info(f"Train ratio: {self.train_ratio}")
            logger.info("Mode: FULL DATASET")

        except Exception as e:
            raise CustomException(f"Initialization failed: {e}") from e

    def _resolve_gcp_credentials(self):
        env_path = os.getenv(self.gcp_credentials_env, "") if self.gcp_credentials_env else ""
        mounted_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

        if mounted_path and os.path.exists(mounted_path):
            logger.info("Using mounted GCP credentials from GOOGLE_APPLICATION_CREDENTIALS")
            return mounted_path

        if env_path and os.path.exists(env_path):
            logger.info(f"Using GCP credentials from environment variable: {self.gcp_credentials_env}")
            return env_path

        if self.gcp_credentials and os.path.exists(self.gcp_credentials):
            logger.info("Using legacy configured GCP credentials path")
            return self.gcp_credentials

        return None

    def _get_gcp_client(self):
        try:
            cred_path = self._resolve_gcp_credentials()
            if cred_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
            return storage.Client()
        except Exception as e:
            raise CustomException(f"GCP client init failed: {e}") from e

    def download_data_from_gcs(self):
        try:
            logger.info("Connecting to GCS...")
            client = self._get_gcp_client()
            bucket = client.bucket(self.bucket_name)

            for file_name in self.bucket_file_names:
                local_path = os.path.join(RAW_DIR, file_name)
                logger.info(f"Downloading: {file_name}")
                blob = bucket.blob(file_name)
                blob.download_to_filename(local_path)
                logger.info(f"Downloaded: {file_name}")

        except Exception as e:
            raise CustomException(f"GCS download failed: {e}") from e

    def split_data_into_train_test(self):
        try:
            raw_file = os.path.join(RAW_DIR, "animelist.csv")

            if not os.path.exists(raw_file):
                raise FileNotFoundError(f"Missing file: {raw_file}")

            logger.info("Reading full dataset...")
            df = pd.read_csv(raw_file)

            logger.info(f"Total rows loaded: {len(df):,}")

            train_df, test_df = train_test_split(
                df,
                test_size=1 - self.train_ratio,
                random_state=42
            )

            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Train rows: {len(train_df):,}")
            logger.info(f"Test rows: {len(test_df):,}")

        except Exception as e:
            raise CustomException(f"Train/test split failed: {e}") from e

    def run(self):
        try:
            logger.info("Starting Data Ingestion Pipeline")
            self.download_data_from_gcs()
            self.split_data_into_train_test()
            logger.info("Pipeline completed successfully")
        except Exception as e:
            raise CustomException(f"Pipeline failed: {e}") from e


if __name__ == "__main__":
    try:
        ingestion = DataIngestion(config="config/config.yaml")
        ingestion.run()
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)