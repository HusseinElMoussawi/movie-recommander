"""
Movie Recommender API
FastAPI application exposing recommendation endpoints.
Run locally: uvicorn app:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import os

from model import load_artifacts, get_recommendations, load_data, build_user_item_matrix, train_model, save_artifacts

# ── Global state ─────────────────────────────────────────────────────────────
state = {}

ARTIFACTS_DIR = "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    if not os.path.exists(ARTIFACTS_DIR):
        print("⚙️  No artifacts found. Training from scratch...")
        ratings, movies = load_data()
        matrix = build_user_item_matrix(ratings)
        svd, user_embeddings, item_embeddings = train_model(matrix)
        save_artifacts(svd, matrix, item_embeddings, movies)

    state["svd"], state["matrix"], state["item_embeddings"], state["movies"] = (
        load_artifacts(ARTIFACTS_DIR)
    )
    print("✅ Model loaded successfully.")
    yield
    state.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🎬 Movie Recommender API",
    description=(
        "Collaborative filtering recommender built on the MovieLens dataset. "
        "Pass a list of movie IDs you like and get personalised recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    liked_movie_ids: list[int] = Field(
        ...,
        min_length=1,
        example=[1, 260, 296],
        description="List of MovieLens movie IDs you have enjoyed.",
    )
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations to return.")


class MovieResult(BaseModel):
    movieId: int
    title: str
    genres: str
    score: float


class RecommendResponse(BaseModel):
    recommendations: list[MovieResult]
    input_movie_ids: list[int]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Movie Recommender API is running 🎬"}


@app.get("/movies/search", tags=["Movies"])
def search_movies(q: str, limit: int = 10):
    """Search for movies by title to find their IDs."""
    movies = state["movies"]
    results = movies[movies["title"].str.contains(q, case=False, na=False)].head(limit)
    return results[["movieId", "title", "genres"]].to_dict(orient="records")


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(body: RecommendRequest):
    """
    Get movie recommendations based on a list of liked movie IDs.
    Use /movies/search to find movie IDs by title first.
    """
    recs = get_recommendations(
        liked_movie_ids=body.liked_movie_ids,
        matrix=state["matrix"],
        item_embeddings=state["item_embeddings"],
        movies=state["movies"],
        top_n=body.top_n,
    )
    if not recs:
        raise HTTPException(
            status_code=404,
            detail="None of the provided movie IDs were found in the dataset.",
        )
    return RecommendResponse(
        recommendations=recs,
        input_movie_ids=body.liked_movie_ids,
    )
