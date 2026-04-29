import os

import pandas as pd
import requests
import streamlit as st
from rapidfuzz import fuzz, process

from config.paths_config import DF
from ui.utils_ui import anime_card, explanation_card

API_HOST = os.getenv("API_HOST", "backend-service")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

AUTH_TIMEOUT_SECONDS = int(os.getenv("AUTH_TIMEOUT_SECONDS", "10"))
READ_TIMEOUT_SECONDS = int(os.getenv("READ_TIMEOUT_SECONDS", "180"))


@st.cache_data
def load_anime_df():
    """Load anime metadata once and cache it for the Streamlit session."""
    return pd.read_csv(DF)


anime_df = load_anime_df()


def get_token(username="demo", password="demo"):
    """Request a JWT token from the backend login endpoint."""
    try:
        resp = requests.post(
            f"{API_URL}/auth/login",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=AUTH_TIMEOUT_SECONDS,
        )

        if resp.status_code == 200:
            return resp.json()["access_token"]

        st.error(f"Authentication failed: {resp.text}")
        return None

    except requests.exceptions.RequestException as exc:
        st.error(f"Authentication failed: {exc}")
        return None


def get_recommendations(user_id, token, user_weight, content_weight, top_k):
    """Call the backend recommendation endpoint and return the JSON payload."""
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "user_weight": user_weight,
        "content_weight": content_weight,
        "top_k": top_k,
    }

    try:
        resp = requests.get(
            f"{API_URL}/recommend/{user_id}",
            headers=headers,
            params=params,
            timeout=(10, READ_TIMEOUT_SECONDS),
        )

        if resp.status_code == 200:
            return resp.json()

        st.error(f"❌ API Error: {resp.text}")
        return {}

    except requests.exceptions.ReadTimeout:
        st.error(
            f"API Error: request exceeded {READ_TIMEOUT_SECONDS}s. "
            "The backend is reachable, but recommendation generation is taking too long."
        )
        return {}

    except requests.exceptions.RequestException as exc:
        st.error(f"❌ Could not contact API: {exc}")
        return {}


def search_anime(query, limit=10):
    """Perform fuzzy title matching against the anime metadata."""
    names = anime_df["eng_version"].fillna("").tolist()
    matches = process.extract(query, names, scorer=fuzz.WRatio, limit=limit)
    return [m[0] for m in matches if m[1] > 60]


st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")

st.title("🎌 Anime Recommendation System")
st.caption("FAISS • Hybrid • Embeddings • MMR • Explanations")

st.sidebar.header("⚙️ Engine Settings")
user_weight = st.sidebar.slider("User-Based Weight", 0.0, 1.0, 0.6)
content_weight = st.sidebar.slider("Content-Based Weight", 0.0, 1.0, 0.4)
top_k = st.sidebar.slider("Top K Recommendations", 5, 25, 10)

st.subheader("🔍 Search Anime")
search_query = st.text_input("Search for an anime title:")

if search_query:
    results = search_anime(search_query)
    for r in results[:5]:
        st.write("• " + r)

st.write("---")
st.subheader("🎯 Get Personalized Recommendations")

user_id = st.number_input("Enter User ID", min_value=1, value=11880)

if st.button("🎌 Recommend!", use_container_width=True):
    token = get_token()
    if not token:
        st.stop()

    with st.spinner("Generating recommendations... This can take longer on a cold request."):
        result = get_recommendations(user_id, token, user_weight, content_weight, top_k)

    if not result:
        st.warning("No results returned.")
        st.stop()

    recs = result.get("recommendations", [])
    explanations = result.get("explanations", {})

    st.subheader(f"✅ Top {len(recs)} Recommendations")
    st.write("---")

    cols = st.columns(3)

    for i, anime_name in enumerate(recs):
        try:
            anime_id = int(anime_df[anime_df["eng_version"] == anime_name]["anime_id"].values[0])
        except Exception:
            anime_id = None

        exp = explanations.get(str(anime_id), explanations.get(anime_id, {}))

        with cols[i % 3]:
            anime_card(anime_name, anime_df)
            explanation_card(exp)