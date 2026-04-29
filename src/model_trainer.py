import json
import os
import sys
from datetime import datetime

import comet_ml
import joblib
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

from config.paths_config import (
    ANIME2ANIME_DECODED_PATH,
    ANIME2ANIME_ENCODED_PATH,
    ANIME_WEIGHTS_PATH,
    CHECKPOINT_FILE_PATH,
    MODEL_DIR,
    MODEL_PATH,
    PROCESSED_DIR,
    USER2USER_DECODED_PATH,
    USER2USER_ENCODED_PATH,
    USER_WEIGHTS_PATH,
    WEIGHTS_DIR,
    X_TEST_ARRAY,
    X_TRAIN_ARRAY,
    Y_TEST,
    Y_TRAIN,
)
from src.base_model import BaseModel
from src.custom_exception import CustomException
from src.logger import get_logger
from utils.common_functions import read_yaml_file

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml", experiment_name: str | None = None):
        self.config_path = config_path
        self.experiment = None
        self.config = None
        self.comet_config = None
        self.run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

        self.epochs = 12
        self.batch_size = 8192
        self.patience = 4

        try:
            logger.info(f"📄 Loading configuration from: {config_path}")
            self.config = read_yaml_file(config_path)

            model_config = self.config.get("model", {})
            self.epochs = model_config.get("epochs", self.epochs)
            self.batch_size = model_config.get("batch_size", self.batch_size)
            self.patience = model_config.get("patience", self.patience)

            self.comet_config = self.config.get("comet_ml", {})
            exp_name = experiment_name or self.comet_config.get(
                "experiment_name",
                f"anime-recommender-{self.run_id}"
            )

            comet_enabled = self.comet_config.get("enabled", True)
            comet_api_key = os.getenv(
                self.comet_config.get("api_key_env", "COMET_API_KEY"),
                ""
            )

            if comet_enabled and comet_api_key:
                self.experiment = comet_ml.Experiment(
                    api_key=comet_api_key,
                    project_name=self.comet_config.get("project_name", "anime-recommender"),
                    workspace=self.comet_config.get("workspace"),
                )
                self.experiment.set_name(exp_name)
                logger.info(f"✅ Comet ML experiment started: {exp_name}")
                logger.info(f"🔗 Experiment URL: {self.experiment.url}")
            else:
                logger.warning("⚠️ Comet ML tracking is disabled or COMET_API_KEY is missing.")

        except Exception as exc:
            logger.warning(f"⚠️ Could not initialize Comet ML or load config: {exc}. Continuing without tracking.")
            self.experiment = None

        self.X_train_array = None
        self.X_test_array = None
        self.y_train = None
        self.y_test = None
        self.n_users = None
        self.n_anime = None
        self.model = None
        self.history = None

        logger.info("✅ ModelTrainer initialized successfully")
        logger.info(f"   Run ID      : {self.run_id}")

    def load_data(self):
        try:
            self.X_train_array = joblib.load(X_TRAIN_ARRAY)
            self.X_test_array = joblib.load(X_TEST_ARRAY)
            self.y_train = joblib.load(Y_TRAIN)
            self.y_test = joblib.load(Y_TEST)

            user_encoded = joblib.load(USER2USER_ENCODED_PATH)
            anime_encoded = joblib.load(ANIME2ANIME_ENCODED_PATH)

            self.n_users = len(user_encoded)
            self.n_anime = len(anime_encoded)
        except Exception as exc:
            raise CustomException(f"Data loading failed: {exc}") from exc

    def get_lr_scheduler(self):
        def lrfn(epoch):
            start_lr = 1e-5
            max_lr = 5e-4
            min_lr = 1e-6
            rampup = 5
            sustain = 2
            exp_decay = 0.8

            if epoch < rampup:
                return (max_lr - start_lr) / rampup * epoch + start_lr
            if epoch < rampup + sustain:
                return max_lr
            return (max_lr - min_lr) * exp_decay ** (epoch - rampup - sustain) + min_lr

        return LearningRateScheduler(lrfn, verbose=0)

    def _log_to_comet(self):
        if self.experiment is None or self.history is None:
            return
        try:
            for epoch in range(len(self.history.history.get("loss", []))):
                self.experiment.log_metric("train_loss", self.history.history["loss"][epoch], step=epoch)
                if "val_loss" in self.history.history:
                    self.experiment.log_metric("val_loss", self.history.history["val_loss"][epoch], step=epoch)
                if "mae" in self.history.history:
                    self.experiment.log_metric("train_mae", self.history.history["mae"][epoch], step=epoch)
                if "mse" in self.history.history:
                    self.experiment.log_metric("train_mse", self.history.history["mse"][epoch], step=epoch)
        except Exception as exc:
            logger.warning(f"⚠️ Failed to log metrics to Comet ML: {exc}")

    def extract_weights(self, layer_name: str):
        try:
            weights = self.model.get_layer(layer_name).get_weights()[0]
            norms = np.linalg.norm(weights, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return weights / norms
        except Exception as exc:
            raise CustomException(f"Failed to extract {layer_name} weights: {exc}") from exc

    def _write_artifact_manifest(self):
        manifest = {
            "run_id": self.run_id,
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "n_users": int(self.n_users),
            "n_anime": int(self.n_anime),
            "files": {
                "user_weights": USER_WEIGHTS_PATH,
                "anime_weights": ANIME_WEIGHTS_PATH,
                "user2user_encoded": USER2USER_ENCODED_PATH,
                "user2user_decoded": USER2USER_DECODED_PATH,
                "anime2anime_encoded": ANIME2ANIME_ENCODED_PATH,
                "anime2anime_decoded": ANIME2ANIME_DECODED_PATH,
            },
        }

        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        manifest_path = os.path.join(WEIGHTS_DIR, "artifact_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def save_model_and_weights(self):
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(WEIGHTS_DIR, exist_ok=True)

            self.model.save(MODEL_PATH)

            user_weights = self.extract_weights("user_embedding")
            anime_weights = self.extract_weights("anime_embedding")

            joblib.dump(user_weights, USER_WEIGHTS_PATH)
            joblib.dump(anime_weights, ANIME_WEIGHTS_PATH)

            self._write_artifact_manifest()
        except Exception as exc:
            raise CustomException(f"Model saving failed: {exc}") from exc

    def train_model(self, epochs: int | None = None, batch_size: int | None = None, patience: int | None = None):
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        patience = patience or self.patience

        try:
            self.load_data()
            base_model = BaseModel(config_path=self.config_path)
            self.model = base_model.RecommenderNet(n_users=self.n_users, n_anime=self.n_anime)

            callbacks = [
                ModelCheckpoint(
                    filepath=CHECKPOINT_FILE_PATH,
                    save_weights_only=True,
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True,
                    verbose=1,
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
                self.get_lr_scheduler(),
            ]

            self.history = self.model.fit(
                x=self.X_train_array,
                y=self.y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(self.X_test_array, self.y_test),
                callbacks=callbacks,
            )

            self.model.load_weights(CHECKPOINT_FILE_PATH)
            self._log_to_comet()
            self.save_model_and_weights()
        except Exception as exc:
            raise CustomException(f"Model training failed: {exc}") from exc

    def run(self, epochs: int | None = None, batch_size: int | None = None, patience: int | None = None):
        try:
            self.train_model(epochs=epochs, batch_size=batch_size, patience=patience)
        finally:
            if self.experiment:
                self.experiment.end()


if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        trainer.run()
    except Exception as exc:
        logger.error(f"💥 Failed to start ModelTrainer: {exc}")
        sys.exit(1)