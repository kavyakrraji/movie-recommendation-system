import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

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

    movies = movies.dropna(subset=["movieId", "title"]).drop_duplicates("movieId")
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    movies["movieId"] = movies["movieId"].astype(int)
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    movies["genres"] = movies["genres"].fillna("")
    return movies, ratings

movies, ratings = load_data()

@st.cache_data
def build_content_model(genres):
    vectorizer = TfidfVectorizer(token_pattern=r"[^|]+")
    matrix = vectorizer.fit_transform(genres)
    return cosine_similarity(matrix)

@st.cache_data
def build_rating_matrix(ratings):
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

content_similarity = build_content_model(movies["genres"].tolist())
rating_matrix = build_rating_matrix(ratings)

movie_titles = movies["title"].tolist()
selected_movie = st.selectbox("Choose a movie", movie_titles)
method = st.selectbox(
    "Recommendation method",
    ["Content-Based", "Collaborative", "Hybrid"]
)
top_n = st.slider("Number of recommendations", 3, 10, 5)

if st.button("Recommend Movies"):
    selected_position = movies.index[
        movies["title"] == selected_movie
    ][0]
    movie_id = int(movies.loc[selected_position, "movieId"])

    content_scores = content_similarity[
        movies.index.get_loc(selected_position)
    ].copy()

    user_id = None
    collaborative_scores = np.zeros(len(movies))

    if method in ["Collaborative", "Hybrid"]:
        user_id = st.selectbox(
            "Select a sample user ID",
            rating_matrix.index.tolist()
        ) if False else rating_matrix.index[0]

        user_position = rating_matrix.index.get_loc(user_id)
        user_similarity = cosine_similarity(
            csr_matrix(rating_matrix.values),
            dense_output=False
        )
        similarities = user_similarity.getrow(
            user_position
        ).toarray().ravel()

        similar_users = [
            i for i in similarities.argsort()[::-1]
            if i != user_position and similarities[i] > 0
        ][:20]

        watched = set(
            rating_matrix.loc[user_id][
                rating_matrix.loc[user_id] > 0
            ].index
        )
        scores_by_id = {}

        for other_position in similar_users:
            other_ratings = rating_matrix.iloc[other_position]
            for mid, rating in other_ratings[other_ratings > 0].items():
                if mid not in watched:
                    scores_by_id[mid] = (
                        scores_by_id.get(mid, 0)
                        + similarities[other_position] * rating
                    )

        for i, mid in enumerate(movies["movieId"]):
            collaborative_scores[i] = scores_by_id.get(mid, 0)

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

    final_scores[movies.index.get_loc(selected_position)] = -1
    top_positions = np.argsort(final_scores)[::-1][:top_n]

    results = movies.iloc[top_positions][
        ["title", "genres"]
    ].copy()
    results["Score"] = np.round(final_scores[top_positions], 3)

    st.subheader("Recommended Movies")
    st.dataframe(results.reset_index(drop=True), use_container_width=True)
