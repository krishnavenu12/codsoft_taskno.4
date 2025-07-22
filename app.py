# app.py
import streamlit as st
import pandas as pd

# --- Load Data ---
@st.cache_data
def load_data():
    ratings = pd.read_csv("u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
    movies = pd.read_csv("u.item", sep="|", encoding="latin-1", names=[
        "movie_id", "title", "release_date", "video_release", "imdb_url",
        "unknown", "action", "adventure", "animation", "childrens", "comedy", "crime",
        "documentary", "drama", "fantasy", "film_noir", "horror", "musical", "mystery",
        "romance", "sci_fi", "thriller", "war", "western"
    ])[["movie_id", "title"]]
    
    df = pd.merge(ratings, movies, on="movie_id")
    return df

df = load_data()

# --- Build Pivot Table ---
user_movie_matrix = df.pivot_table(index="user_id", columns="title", values="rating")

# --- Recommend Similar Movies ---
def get_similar_movies(movie_name, min_ratings=50):
    if movie_name not in user_movie_matrix.columns:
        return pd.DataFrame(columns=["title", "correlation", "rating_count"])
    
    movie_ratings = user_movie_matrix[movie_name]
    similar_movies = user_movie_matrix.corrwith(movie_ratings)
    
    corr_df = pd.DataFrame(similar_movies, columns=["correlation"])
    corr_df.dropna(inplace=True)

    rating_counts = df.groupby("title")["rating"].count()
    corr_df = corr_df.join(rating_counts.rename("rating_count"))
    
    result = corr_df[corr_df["rating_count"] >= min_ratings].sort_values("correlation", ascending=False).head(10)
    result = result.reset_index().rename(columns={"title": "Movie Title"})
    return result

# --- Streamlit UI ---
st.set_page_config("🎬 Movie Recommender")
st.title("🎬 Simple Movie Recommender")
st.write("This system uses collaborative filtering to find similar movies based on ratings.")

movie_list = sorted(df["title"].unique())
selected_movie = st.selectbox("Select a movie to get recommendations:", movie_list)

min_ratings = st.slider("Minimum number of ratings to consider", 20, 100, 50)

if st.button("Get Recommendations"):
    results = get_similar_movies(selected_movie, min_ratings)
    
    if not results.empty:
        st.success(f"Top 10 movies similar to **{selected_movie}**:")
        st.dataframe(results[["Movie Title", "correlation", "rating_count"]], use_container_width=True)
    else:
        st.warning("Not enough data for this movie. Try another one.")

