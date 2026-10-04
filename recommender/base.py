"""The contract every recommender must follow. The UI and chatbot only use these methods,
so a teammate's model can be dropped in without touching them."""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

RESULT_COLS = ["id", "title", "release_year", "genres", "vote_average", "vote_count",
               "overview", "poster_path", "language", "runtime", "directors", "cast", "score"]


class BaseRecommender(ABC):
    name = "base"

    def __init__(self, movies: pd.DataFrame):
        self.movies = movies

    # ---- must implement ------------------------------------------------------------
    @abstractmethod
    def similarity_to(self, row_ids: list) -> np.ndarray:
        """Similarity of EVERY movie to the mean of the given rows (shape: n_movies)."""

    @abstractmethod
    def similarity_to_text(self, text: str) -> np.ndarray:
        """Similarity of every movie to a free-text description (zeros if unsupported)."""

    # ---- shared logic (rarely needs overriding) ---------------------------------------
    def _mask(self, genres=None, year_range=None, language=None, max_runtime=None,
              min_runtime=None, min_rating=None, director=None, actor=None, min_votes=0):
        m = self.movies
        keep = np.ones(len(m), bool)
        if genres:
            keep &= m["genres_l"].map(lambda g: all(x in g for x in genres)).to_numpy()
        if year_range:
            y = m["release_year"].astype("float")
            keep &= ((y >= year_range[0]) & (y <= year_range[1])).to_numpy()
        if language:
            keep &= (m["language"] == language).to_numpy()
        if max_runtime:
            keep &= (m["runtime"] <= max_runtime).to_numpy()
        if min_runtime:
            keep &= (m["runtime"] >= min_runtime).to_numpy()
        if min_rating:
            keep &= (m["wr"] >= min_rating).to_numpy()
        if director:
            keep &= m["directors"].str.contains(director, case=False, regex=False).to_numpy()
        if actor:
            keep &= m["cast"].str.contains(actor, case=False, regex=False).to_numpy()
        if min_votes:
            keep &= (m["vote_count"] >= min_votes).to_numpy()
        return keep

    def filter_mask(self, **kw):
        """Public boolean mask over all movies (used by the Browse page)."""
        return self._mask(**kw)

    def _finish(self, order, scores, n):
        idx = order[:n]
        out = self.movies.iloc[idx].copy()
        out["score"] = scores[idx]
        return out[RESULT_COLS].reset_index(drop=True)

    def recommend_similar(self, row_ids, n=10, beta=0.15, min_votes=50, **filters):
        """Movies similar to one or several rows (a title, or a user's favourites)."""
        row_ids = [int(r) for r in row_ids]
        sim = self.similarity_to(row_ids)
        wr = self.movies["wr"].to_numpy()
        wr_n = (wr - wr.min()) / (wr.max() - wr.min() + 1e-9)
        score = sim + beta * wr_n
        keep = self._mask(min_votes=min_votes, **filters)
        keep[row_ids] = False
        score = np.where(keep, score, -np.inf)
        order = np.argsort(-score)
        order = order[np.isfinite(score[order])]
        return self._finish(order, score, n)

    def recommend_by_filters(self, genres=None, year_range=None, language=None, max_runtime=None,
                             min_runtime=None, min_rating=None, director=None, actor=None,
                             text=None, n=10, min_votes=50, sort_by="wr"):
        """Movies matching structured preferences, optionally ranked by a text theme."""
        keep = self._mask(genres, year_range, language, max_runtime, min_runtime, min_rating,
                          director, actor, min_votes)
        wr = self.movies["wr"].to_numpy()
        wr_n = (wr - wr.min()) / (wr.max() - wr.min() + 1e-9)
        if text:
            sim = self.similarity_to_text(text)
            score = sim + 0.15 * wr_n
            if sim.max() <= 0:              # nothing matched the theme words
                keep &= False
        elif sort_by == "popularity":
            score = self.movies["popularity"].to_numpy()
        elif sort_by == "newest":
            score = self.movies["release_year"].astype(float).fillna(0).to_numpy()
        else:
            score = wr
        score = np.where(keep, score, -np.inf)
        order = np.argsort(-score)
        order = order[np.isfinite(score[order])]
        return self._finish(order, score, n)
