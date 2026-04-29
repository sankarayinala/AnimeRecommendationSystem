import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

RAW_DIR = os.path.join(ARTIFACTS_DIR, "raw")
PROCESSED_DIR = os.path.join(ARTIFACTS_DIR, "processed")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "model")
WEIGHTS_DIR = os.path.join(ARTIFACTS_DIR, "weights")

for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, WEIGHTS_DIR]:
    os.makedirs(d, exist_ok=True)

ANIMELIST_CSV = os.path.join(RAW_DIR, "animelist.csv")
ANIME_CSV = os.path.join(RAW_DIR, "anime.csv")
ANIMESYNOPSIS_CSV = os.path.join(RAW_DIR, "anime_with_synopsis.csv")

X_TRAIN_ARRAY = os.path.join(PROCESSED_DIR, "X_train_array.pkl")
X_TEST_ARRAY = os.path.join(PROCESSED_DIR, "X_test_array.pkl")
Y_TRAIN = os.path.join(PROCESSED_DIR, "y_train.pkl")
Y_TEST = os.path.join(PROCESSED_DIR, "y_test.pkl")

RATING_DF = os.path.join(PROCESSED_DIR, "rating_df.parquet")
DF = os.path.join(PROCESSED_DIR, "anime_df.parquet")
SYNOPSIS_DF = os.path.join(PROCESSED_DIR, "synopsis_df.parquet")

TRAIN_FILE_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(PROCESSED_DIR, "test.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "recommender_model.h5")
CHECKPOINT_FILE_PATH = os.path.join(WEIGHTS_DIR, "best_model.weights.h5")

USER_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "user_weights.pkl")
ANIME_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "anime_weights.pkl")

ANIME2ANIME_ENCODED_PATH = os.path.join(WEIGHTS_DIR, "anime2anime_encoded.pkl")
ANIME2ANIME_DECODED_PATH = os.path.join(WEIGHTS_DIR, "anime2anime_decoded.pkl")

USER2USER_ENCODED_PATH = os.path.join(WEIGHTS_DIR, "user2user_encoded.pkl")
USER2USER_DECODED_PATH = os.path.join(WEIGHTS_DIR, "user2user_decoded.pkl")


def verify_file(path: str, name: str, required: bool = True):
    if not os.path.exists(path):
        msg = f"Missing {name} at: {path}"
        if required:
            raise FileNotFoundError(msg)
        print(f"⚠️ WARNING: {msg}")
    return path


def verify_critical_paths():
    verify_file(RATING_DF, "rating dataframe")
    verify_file(DF, "anime metadata dataframe")
    verify_file(SYNOPSIS_DF, "synopsis dataframe")
    verify_file(USER_WEIGHTS_PATH, "user weights")
    verify_file(ANIME_WEIGHTS_PATH, "anime weights")
    verify_file(USER2USER_ENCODED_PATH, "user2user encoded map")
    verify_file(USER2USER_DECODED_PATH, "user2user decoded map")
    verify_file(ANIME2ANIME_ENCODED_PATH, "anime2anime encoded map")
    verify_file(ANIME2ANIME_DECODED_PATH, "anime2anime decoded map")