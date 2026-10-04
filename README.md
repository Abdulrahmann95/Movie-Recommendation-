# 🎬 CineMatch: movie recommender (Streamlit)

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
First launch takes ~15 s (builds the TF-IDF baseline, cached afterwards in `models/`).

## What's inside
| Part | Where |
|---|---|
| Login / sign up / guest | `views/welcome.py`, `core/storage.py` (SQLite file `data/app.db`, PBKDF2-hashed passwords) |
| Browse + filters, Similar movies, My lists (favorites / watched / "for you") | `views/` |
| Chatbot (offline, rule-based, English only) | `chatbot/parser.py` (patterns) → `chatbot/engine.py` (replies) |
| Insights page (charts from the notebook) | `views/insights.py` |
| Recommenders | `recommender/` (`base.py` is the contract) |
| Data | `data/movies_clean.csv.gz` (made by `scripts/prepare_data.py`) |

Guests can browse, search, chat and get recommendations. Logged-in users can also favorite
movies and mark them as watched (buttons on every card, or by chatting:
*"add the second one to my favorites"*, *"mark Inception as watched"*).

## Plugging in the models (the notebook your teammates finish)
Your notebook already builds the fused embedding matrix `E`. Add ONE cell at the end:
```python
np.savez_compressed("models/hybrid_embeddings.npz",
                    embeddings=E, ids=app["id"].to_numpy())   # app = the app_meta table
```
Copy that file into this project's `models/` folder and restart. A new entry
**"Hybrid Embeddings"** appears in the sidebar model dropdown, and the Similar page has a
**"Compare all models"** switch to show baseline vs. hybrid side by side.
Rows are re-aligned by movie `id`, so row order and filters don't have to match exactly.

Any other model that gives one vector per movie (SBERT only, SVD, autoencoder...) works the same way:
save its matrix as `models/<name>.npz` and it shows up under that name.
For a model that is not an embedding matrix, subclass `BaseRecommender`
(2 methods: `similarity_to`, `similarity_to_text`) and register it in `recommender/registry.py` → `CUSTOM`.

## Chatbot examples
`scary movies from the 90s` · `something like Sherlock Jr. but newer, under 90 min` ·
`top 5 highly rated Japanese animation` · `movies directed by Christopher Nolan` ·
`movies with Tom Hanks` · `space heist` · `surprise me` · `tell me about The Godfather`.
It only ever returns movies from the dataset. Non-English input gets a "please type in English" reply.
If nothing matches all filters it relaxes them one by one and says which it dropped.

## Deploying
* **Local / demo laptop (recommended):** `data/app.db` is a normal file, so accounts and lists persist.
* **Streamlit Community Cloud:** works (the data file is 16 MB), but the container's disk is
  temporary: **accounts, favorites and watched lists are wiped on restart or redeploy.**
  For lasting data swap `core/storage.py` (the only file that touches the DB) for a hosted
  database such as Supabase or Turso.
