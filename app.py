import os
from concurrent.futures import ThreadPoolExecutor

import pickle
import requests
import streamlit as st
import zstandard as zstd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "data/pkl_data"
MOVIES_PATH = os.path.join(DATA_DIR, "movies_df.zstd")
SIMILARITY_PATH = os.path.join(DATA_DIR, "similarity.zstd")
FALLBACK_POSTER = "no-poster.png"

load_dotenv()

# Falls back to the key you provided if OMDB_API_KEY isn't set in .env.
# For real use, put your key in a .env file instead of leaving it here:
#   OMDB_API_KEY=your_key_here
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "ea592f08")
OMDB_URL = "http://www.omdbapi.com/"


# ---------------------------------------------------------------------------
# Data loading (cached so it only runs once per session, not on every click)
# ---------------------------------------------------------------------------
def decompress_with_zstd(file_path):
    with open(file_path, "rb") as f:
        compressed_data = f.read()
    dc = zstd.ZstdDecompressor()
    decompressed_data = dc.decompress(compressed_data)
    return pickle.loads(decompressed_data)


@st.cache_resource(show_spinner="Loading movie database...")
def load_data():
    """Load and decompress the movie dataframe + similarity matrix once."""
    if not os.path.exists(MOVIES_PATH) or not os.path.exists(SIMILARITY_PATH):
        return None, None, (
            f"Data files not found. Expected:\n- {MOVIES_PATH}\n- {SIMILARITY_PATH}\n\n"
            "Make sure you've run the notebook to generate these files and that "
            "'data/pkl_data' exists relative to where you launch Streamlit."
        )
    try:
        movies_df = decompress_with_zstd(MOVIES_PATH)
        similarity = decompress_with_zstd(SIMILARITY_PATH)
        return movies_df, similarity, None
    except Exception as e:
        return None, None, f"Failed to load data files: {e}"


# ---------------------------------------------------------------------------
# Poster fetching via OMDb (looked up by title, cached per title)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_poster_by_title(title):
    """Look up a movie by title on OMDb and return its poster URL."""
    try:
        response = requests.get(
            OMDB_URL,
            params={"t": title, "apikey": OMDB_API_KEY},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "False":
            # OMDb couldn't find this title, or the key/quota failed
            return FALLBACK_POSTER

        poster_url = data.get("Poster")
        if not poster_url or poster_url == "N/A":
            return FALLBACK_POSTER

        return poster_url
    except requests.RequestException:
        return FALLBACK_POSTER


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------
def recommend(movie, movies_df, similarity, top_n=5):
    matches = movies_df[movies_df["title"] == movie]
    if matches.empty:
        return [], []

    movie_index = matches.index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1: top_n + 1]

    recommended_titles = [movies_df.iloc[i].title for i, _ in movies_list]

    # Fetch all posters concurrently instead of one by one
    with ThreadPoolExecutor(max_workers=min(15, len(recommended_titles)) or 1) as executor:
        recommended_posters = list(executor.map(fetch_poster_by_title, recommended_titles))

    return recommended_titles, recommended_posters


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

    hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        button[title="View fullscreen"] {visibility: hidden;}
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.title("🎬 Movie Recommendation System")
    st.caption("Posters powered by OMDb.")

    movies_df, similarity, load_error = load_data()

    if load_error:
        st.error(load_error)
        st.stop()

    selected_movie = st.selectbox(
        label="Select a movie",
        options=movies_df["title"],
        placeholder="Choose a movie",
        index=None,
    )

    if st.button("Recommend", type="primary"):
        if selected_movie is None:
            st.error("No movie selected.")
            st.stop()

        with st.spinner("Finding movies you'll like..."):
            names, posters = recommend(selected_movie, movies_df, similarity)

        if not names:
            st.error("Currently we don't have any information about this movie.")
            st.stop()

        cols = st.columns(5)
        for idx, (name, poster) in enumerate(zip(names, posters)):
            with cols[idx % 5]:
                st.image(poster, use_container_width=True)
                st.caption(name)


if __name__ == "__main__":
    main()