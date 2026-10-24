
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)

st.title("🎬 Movie Recommendation System")
st.write("Discover movies you may enjoy!")

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    movies = movies.dropna(subset=["movieId", "title"])
    movies = movies.drop_duplicates("movieId").reset_index(drop=True)
    movies["genres"] = movies["genres"].fillna("")

    ratings = ratings.dropna(
        subset=["userId", "movieId", "rating"]
    )

    vectorizer = TfidfVectorizer(
        token_pattern=r"[^|]+"
    )
    genre_matrix = vectorizer.fit_transform(movies["genres"])

    rating_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    return movies, rating_matrix, genre_matrix

movies, rating_matrix, genre_matrix = load_data()

selected_movie = st.selectbox(
    "Choose a movie",
    movies["title"].tolist()
)

method = st.selectbox(
    "Recommendation method",
    ["Content-Based", "Collaborative", "Hybrid"]
)

top_n = st.slider("Number of recommendations", 3, 10, 5)

if st.button("Recommend Movies"):
    movie_index = movies.index[
        movies["title"] == selected_movie
    ][0]

    content_scores = cosine_similarity(
        genre_matrix[movie_index],
        genre_matrix
    ).flatten()

    collaborative_scores = np.zeros(len(movies))

    if method in ["Collaborative", "Hybrid"]:
        movie_id = int(movies.iloc[movie_index]["movieId"])

        if movie_id in rating_matrix.columns:
            movie_ratings = rating_matrix[movie_id]
            rated_users = movie_ratings[movie_ratings > 0].index

            if len(rated_users) > 0:
                user_scores = rating_matrix.loc[rated_users].mean(axis=0)

                for i, mid in enumerate(movies["movieId"]):
                    if mid in user_scores.index:
                        collaborative_scores[i] = user_scores[mid]

    if method == "Content-Based":
        final_scores = content_scores
    elif method == "Collaborative":
        final_scores = collaborative_scores
    else:
        def normalize(scores):
            maximum = scores.max()
            return scores / maximum if maximum > 0 else scores

        final_scores = (
            0.5 * normalize(content_scores)
            + 0.5 * normalize(collaborative_scores)
        )

    final_scores[movie_index] = -1
    top_positions = np.argsort(final_scores)[::-1][:top_n]

    results = movies.iloc[top_positions][
        ["title", "genres"]
    ].copy()

    results["Score"] = np.round(
        final_scores[top_positions], 3
    )

    st.subheader("Recommended Movies")
    st.dataframe(results.reset_index(drop=True), use_container_width=True)
