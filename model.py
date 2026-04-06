"""
Movie Recommendation Model
Uses collaborative filtering (SVD) on the MovieLens dataset.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

DATA_URL_RATINGS = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def load_data(data_dir: str = "data"):
    """Load MovieLens ratings and movies CSVs."""
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    return ratings, movies


def build_user_item_matrix(ratings: pd.DataFrame):
    """Build a user-item matrix from ratings dataframe."""
    matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)
    return matrix


def train_model(matrix: pd.DataFrame, n_components: int = 50):
    """
    Train a TruncatedSVD model on the user-item matrix.
    Returns the SVD model and the item (movie) embeddings.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_embeddings = svd.fit_transform(matrix)
    item_embeddings = svd.components_.T  # shape: (n_movies, n_components)
    return svd, user_embeddings, item_embeddings


def get_recommendations(
    liked_movie_ids: list[int],
    matrix: pd.DataFrame,
    item_embeddings: np.ndarray,
    movies: pd.DataFrame,
    top_n: int = 10,
) -> list[dict]:
    """
    Given a list of liked movieIds, return top_n recommendations.
    Strategy: average the embeddings of liked movies, then find nearest neighbors.
    """
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(matrix.columns)}

    # Filter to known movies
    valid_ids = [mid for mid in liked_movie_ids if mid in movie_id_to_idx]
    if not valid_ids:
        return []

    indices = [movie_id_to_idx[mid] for mid in valid_ids]
    liked_vector = item_embeddings[indices].mean(axis=0, keepdims=True)

    similarities = cosine_similarity(liked_vector, item_embeddings)[0]

    # Exclude already liked movies
    for idx in indices:
        similarities[idx] = -1

    top_indices = np.argsort(similarities)[::-1][:top_n]
    recommended_ids = [matrix.columns[i] for i in top_indices]

    results = []
    for movie_id, score in zip(recommended_ids, similarities[top_indices]):
        movie_row = movies[movies["movieId"] == movie_id]
        if not movie_row.empty:
            results.append({
                "movieId": int(movie_id),
                "title": movie_row.iloc[0]["title"],
                "genres": movie_row.iloc[0]["genres"],
                "score": round(float(score), 4),
            })
    return results


def save_artifacts(svd, matrix, item_embeddings, movies, out_dir: str = "artifacts"):
    """Persist trained artifacts to disk."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "svd.pkl"), "wb") as f:
        pickle.dump(svd, f)
    with open(os.path.join(out_dir, "item_embeddings.pkl"), "wb") as f:
        pickle.dump(item_embeddings, f)
    matrix.to_parquet(os.path.join(out_dir, "matrix.parquet"))
    movies.to_parquet(os.path.join(out_dir, "movies.parquet"))
    print(f"✅ Artifacts saved to '{out_dir}/'")


def load_artifacts(out_dir: str = "artifacts"):
    """Load persisted artifacts from disk."""
    with open(os.path.join(out_dir, "svd.pkl"), "rb") as f:
        svd = pickle.load(f)
    with open(os.path.join(out_dir, "item_embeddings.pkl"), "rb") as f:
        item_embeddings = pickle.load(f)
    matrix = pd.read_parquet(os.path.join(out_dir, "matrix.parquet"))
    movies = pd.read_parquet(os.path.join(out_dir, "movies.parquet"))
    return svd, matrix, item_embeddings, movies


if __name__ == "__main__":
    print("📥 Loading data...")
    ratings, movies = load_data()

    print("🔧 Building user-item matrix...")
    matrix = build_user_item_matrix(ratings)

    print(f"   Matrix shape: {matrix.shape}")

    print("🧠 Training SVD model...")
    svd, user_embeddings, item_embeddings = train_model(matrix)

    print("💾 Saving artifacts...")
    save_artifacts(svd, matrix, item_embeddings, movies)

    # Quick sanity check
    # MovieLens ID 1 = Toy Story
    recs = get_recommendations([1, 260, 296], matrix, item_embeddings, movies, top_n=5)
    print("\n🎬 Sample recommendations for [Toy Story, Star Wars, Pulp Fiction]:")
    for r in recs:
        print(f"  - {r['title']} ({r['genres']}) — score: {r['score']}")
