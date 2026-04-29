"""Base model definition for the anime recommendation system.

This module loads model configuration, builds the collaborative filtering
network, and keeps the model architecture isolated from training code.
"""

from typing import Dict, Any

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

from src.logger import get_logger
from src.custom_exception import CustomException

# Fixed import - use the actual function name from your utils
from utils.common_functions import read_yaml_file   # ← Changed here

logger = get_logger(__name__)


class BaseModel:
    """
    Base class for building the Anime Recommendation Model.

    The class is responsible for reading model configuration and constructing
    the Keras recommendation network used by the training pipeline.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Load the YAML configuration and initialize model hyperparameters.

        Args:
            config_path: Path to the model configuration file.
        """
        try:
            self.config = read_yaml_file(config_path)
            self.model_config = self.config.get("model", {})

            self.embedding_size = self.model_config.get("embedding_size", 128)
            self.learning_rate = self.model_config.get("learning_rate", 0.001)

            logger.info(f"✅ BaseModel initialized | Embedding size: {self.embedding_size} | LR: {self.learning_rate}")

        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}")
            raise CustomException(f"Error loading configuration: {e}") from e

    def RecommenderNet(self, n_users: int, n_anime: int) -> Model:
        """
        Build the neural collaborative filtering model.

        The model learns separate embeddings for users and anime titles, then
        combines them with a normalized dot product followed by a small dense
        head for final rating prediction.

        Args:
            n_users: Number of unique encoded users.
            n_anime: Number of unique encoded anime items.

        Returns:
            Compiled Keras model ready for training.
        """
        try:
            logger.info(f"Building RecommenderNet | Users: {n_users:,} | Anime: {n_anime:,} | Embedding dim: {self.embedding_size}")

            # User input branch: maps user IDs into learned latent embeddings.
            user_input = Input(shape=[1], name="user")
            user_embedding = Embedding(
                input_dim=n_users,
                output_dim=self.embedding_size,
                name="user_embedding"
            )(user_input)

            # Anime input branch: maps anime IDs into the same embedding space.
            anime_input = Input(shape=[1], name="anime")
            anime_embedding = Embedding(
                input_dim=n_anime,
                output_dim=self.embedding_size,
                name="anime_embedding"
            )(anime_input)

            # Similarity signal is computed as a normalized dot product.
            x = Dot(name="dot_product", normalize=True, axes=2)([user_embedding, anime_embedding])
            x = Flatten(name="flatten")(x)
            x = Dense(1, kernel_initializer='he_normal', name="dense_output")(x)
            x = BatchNormalization(name="batch_norm")(x)
            x = Activation("sigmoid", name="sigmoid_output")(x)

            model = Model(inputs=[user_input, anime_input], outputs=x, name="RecommenderNet")

            # Compile with the configured optimizer and loss function.
            model.compile(
                loss=self.model_config.get("loss", "binary_crossentropy"),
                optimizer=Adam(learning_rate=self.learning_rate),
                metrics=self.model_config.get("metrics", ["mae", "mse"])
            )

            logger.info("✅ RecommenderNet model built and compiled successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise CustomException(f"Failed to create RecommenderNet: {e}") from e


if __name__ == "__main__":
    try:
        base_model = BaseModel()
        logger.info("BaseModel loaded successfully")
    except Exception as e:
        logger.error(f"BaseModel test failed: {e}")