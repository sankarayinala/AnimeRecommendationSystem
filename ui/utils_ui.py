"""Reusable Streamlit UI helpers for rendering anime cards and explanations."""

import streamlit as st
import pandas as pd

from ui.jikan_client import get_poster_url


def anime_card(anime_name: str, anime_df: pd.DataFrame):
    """Render a single anime card with metadata and poster."""
    row = anime_df[anime_df["eng_version"] == anime_name]
    if row.empty:
        st.subheader(anime_name)
        st.write("Metadata unavailable.")
        return

    row = row.iloc[0]
    poster = get_poster_url(row.get("anime_id"))

    if poster:
        st.image(poster, use_column_width=True)

    st.subheader(anime_name)
    st.write(f"**Score:** {row.get('Score', 'N/A')}")
    st.write(f"**Genres:** {row.get('Genres', 'N/A')}")
    st.write(f"**Episodes:** {row.get('Episodes', 'N/A')}")
    st.write(f"**Type:** {row.get('Type', 'N/A')}")
    st.write(f"**Premiered:** {row.get('Premiered', 'N/A')}")
    st.write(f"**Members:** {row.get('Members', 'N/A')}")


def explanation_card(exp: dict):
    """Render explanation details for one recommendation."""
    if not exp:
        st.info("No explanation available.")
        return

    st.markdown("**Why this was recommended**")
    st.write(f"User score: {exp.get('raw_user_score', 0):.3f}")
    st.write(f"Content score: {exp.get('raw_content_score', 0):.3f}")
    st.write(f"Popularity boost: {exp.get('popularity', 0):.3f}")
    st.write(f"Genre overlap: {exp.get('genre_overlap', 0):.3f}")