"""Jikan API helper for fetching anime poster URLs."""

import requests

JIKAN_URL = "https://api.jikan.moe/v4/anime/"


def get_poster_url(mal_id):
    """Return the large poster URL for a MAL anime ID."""
    try:
        resp = requests.get(f"{JIKAN_URL}{mal_id}", timeout=10)
        if resp.status_code != 200:
            return None
        return resp.json()["data"]["images"]["jpg"]["large_image_url"]
    except requests.exceptions.RequestException:
        return None
    except (KeyError, TypeError):
        return None