"""Chat engine: parse the message -> call the recommender -> write the reply.
The bot never invents movies: every title in a reply comes from the recommender / dataset."""
import random
from dataclasses import dataclass, field

from core import storage
from .lookup import Lookup
from .parser import parse

NON_ENGLISH = ("Sorry, I only understand English for now. 🌐 Please type your request in English, "
               "for example: *\"scary movies from the 90s\"* or *\"something like Sherlock Jr.\"*.")
HELP = (
    "I can recommend movies from our dataset. Try things like:\n\n"
    "- *scary movies from the 90s*\n"
    "- *something like Sherlock Jr. but newer, under 90 min*\n"
    "- *top 5 highly rated Japanese animation*\n"
    "- *movies directed by Christopher Nolan* or *movies with Tom Hanks*\n"
    "- *space heist*, *long war movies rated above 7.5*, *surprise me*\n"
    "- *tell me about The Godfather*\n\n"
    "If you're logged in you can also say *add the second one to my favorites*, "
    "*mark Inception as watched*, *show my favorites* or *recommend something based on my favorites*.")
LOGIN_NEEDED = "You need to be logged in for that. Use the sidebar to log in or sign up. 🔐"

RELAX = [("min_rating", "the rating filter"), ("max_runtime", "the runtime limit"),
         ("min_runtime", "the runtime limit"), ("year_range", "the year range"),
         ("language", "the language"), ("genres", "some genres")]


@dataclass
class Reply:
    text: str
    ids: list = field(default_factory=list)      # TMDB ids to show as cards, in order


def constraints(f):
    """Short text for the filters of a 'similar to' request, e.g. 'after 1925, under 90 min'."""
    bits = []
    if f.get("genres"):
        bits.append(" / ".join(f["genres"]))
    if f.get("language"):
        bits.append(f["language"])
    if f.get("year_range"):
        a, b = f["year_range"]
        bits.append(f"after {a - 1}" if b >= 2030 else f"before {b + 1}" if a <= 1880 else f"{a}–{b}")
    if f.get("max_runtime"):
        bits.append(f"under {f['max_runtime']} min")
    if f.get("min_runtime"):
        bits.append(f"over {f['min_runtime']} min")
    if f.get("min_rating"):
        bits.append("highly rated")
    if f.get("director"):
        bits.append(f"by {f['director']}")
    if f.get("actor"):
        bits.append(f"with {f['actor']}")
    return ", ".join(bits)


def describe(f, sort_by="wr"):
    bits = []
    if sort_by == "popularity":
        bits.append("popular")
    if f.get("min_rating"):
        bits.append("highly rated")
    if f.get("genres"):
        bits.append(" / ".join(f["genres"]))
    if f.get("language"):
        bits.append(f["language"]) if f.get("genres") is None else None
    out = " ".join(bits) or "top-rated"
    out += " movies"
    if f.get("language") and f.get("genres"):
        out = f["language"] + "-language " + out
    if f.get("director"):
        out += f" directed by {f['director']}"
    if f.get("actor"):
        out += f" with {f['actor']}"
    if f.get("year_range"):
        a, b = f["year_range"]
        out += f" from {a}–{b}" if b - a > 0 and b < 2030 and a > 1880 else \
            (f" released after {a}" if b >= 2030 else (f" released before {b}" if a <= 1880 else f" from {a}"))
    if f.get("max_runtime"):
        out += f", under {f['max_runtime']} min"
    if f.get("min_runtime"):
        out += f", over {f['min_runtime']} min"
    return out


class ChatEngine:
    def __init__(self, movies):
        self.movies = movies
        self.lk = Lookup(movies)

    # ------------------------------------------------------------------ helpers
    def _rows(self, ids):
        return [self.lk.id_to_row[i] for i in ids if i in self.lk.id_to_row]

    def _info(self, row):
        m = self.movies.iloc[row]
        cast = ", ".join(m["cast"].split(", ")[:5])
        bits = [f"**{m['title']}** ({m['release_year']})",
                f"⭐ {m['vote_average']:.1f} ({int(m['vote_count']):,} votes)"]
        if m["runtime"] == m["runtime"]:
            bits.append(f"{int(m['runtime'])} min")
        txt = " · ".join(bits) + f"\n\n*{m['genres']}*"
        if m["directors"]:
            txt += f"\n\n**Director:** {m['directors']}"
        if cast:
            txt += f"\n\n**Cast:** {cast}"
        if m["overview"]:
            o = m["overview"]
            txt += "\n\n" + (o if len(o) < 450 else o[:447].rsplit(" ", 1)[0] + "…")
        return txt

    def _recommend_with_relax(self, rec, f, text, n, sort_by):
        cur, txt, dropped = dict(f), text, []
        while True:
            res = rec.recommend_by_filters(**cur, text=txt or None, n=n, sort_by=sort_by)
            if len(res):
                return res, dropped
            if txt:
                dropped.append(f'the theme "{txt}"'); txt = ""; continue
            for key, label in RELAX:
                if key in cur:
                    if key == "genres" and len(cur["genres"]) > 1:
                        cur["genres"] = cur["genres"][:1]
                    else:
                        del cur[key]
                    dropped.append(label); break
            else:
                return res, dropped

    # ------------------------------------------------------------------ main entry
    def handle(self, message, rec, username=None, last_ids=None) -> Reply:
        last_ids = last_ids or []
        it = parse(message, self.lk)
        k = it.kind

        if k == "non_english":
            return Reply(NON_ENGLISH)
        if k == "greet":
            return Reply("Hi! 👋 Tell me what you feel like watching, or type **help** to see examples.")
        if k == "thanks":
            return Reply("You're welcome! Want more suggestions? 🍿")
        if k == "bye":
            return Reply("Bye! Enjoy the movie. 🎬")
        if k in ("help", "unknown"):
            pre = "" if k == "help" else "I didn't quite get that. "
            return Reply(pre + HELP)

        if k in ("show_favorites", "show_watched", "for_me", "add_favorite", "mark_watched") and not username:
            return Reply(LOGIN_NEEDED)

        if k == "show_favorites" or k == "show_watched":
            kind = "favorites" if k == "show_favorites" else "watched"
            ids = [i for i in storage.get_ids(username, kind) if i in self.lk.id_to_row]
            if not ids:
                return Reply(f"Your {kind} list is empty for now.")
            return Reply(f"Your {kind} list ({len(ids)}):", ids[:10])

        if k == "for_me":
            fav_rows = self._rows(storage.get_ids(username, "favorites"))[:8]
            if not fav_rows:
                return Reply("Add a few movies to your favorites first, then I can recommend from them. ❤️")
            res = rec.recommend_similar(fav_rows, n=it.n + 3)
            return Reply("Based on your favorites, you might like:", res["id"].tolist()[:max(it.n, 5)])

        if k in ("add_favorite", "mark_watched"):
            row = it.title_row
            if it.ordinal is not None:
                if not last_ids:
                    return Reply("Ask me for some recommendations first, then I can add one of them.")
                try:
                    row = self.lk.id_to_row[last_ids[it.ordinal]]
                except (IndexError, KeyError):
                    return Reply(f"I only showed {len(last_ids)} movies, so I can't find that one.")
            if row is None:
                return Reply(f'I couldn\'t find "{it.title_text}" in the dataset. Check the spelling?')
            m = self.movies.iloc[row]
            storage.add(username, "favorites" if k == "add_favorite" else "watched", int(m["id"]))
            what = "your favorites ❤️" if k == "add_favorite" else "your watched list ✔️"
            return Reply(f"Added **{m['title']}** ({m['release_year']}) to {what}.", [int(m["id"])])

        if k == "title_not_found":
            return Reply(f'I couldn\'t find "{it.title_text}" in the dataset. Check the spelling, '
                         "or try describing what you want (genre, year, mood).")

        if k == "info":
            m = self.movies.iloc[it.title_row]
            return Reply(self._info(it.title_row), [int(m["id"])])

        if k == "title_lookup":
            m = self.movies.iloc[it.title_row]
            sim = rec.recommend_similar([it.title_row], n=4)
            return Reply(self._info(it.title_row) + "\n\n**Similar movies:**",
                         [int(m["id"])] + sim["id"].tolist())

        if k == "similar":
            m = self.movies.iloc[it.title_row]
            res = rec.recommend_similar([it.title_row], n=it.n, **it.filters)
            if not len(res):
                res = rec.recommend_similar([it.title_row], n=it.n)
                return Reply(f"Nothing similar to **{m['title']}** matched all your filters, "
                             "so here are the closest ones without them:", res["id"].tolist())
            tail = f" — {constraints(it.filters)}" if it.filters else ""
            return Reply(f"Movies similar to **{m['title']}** ({m['release_year']}){tail}:", res["id"].tolist())

        # ---- recommend ----
        if it.surprise:
            pool = rec.recommend_by_filters(**it.filters, n=150, min_votes=300)
            if len(pool):
                pick = pool.sample(min(it.n, len(pool)))
                return Reply("Here's a surprise pick for you 🎲", pick["id"].tolist())
        res, dropped = self._recommend_with_relax(rec, it.filters, it.text, it.n, it.sort_by)
        if not len(res):
            return Reply("I couldn't find anything for that. Try fewer filters.")
        intro = f"Here are {len(res)} {describe(it.filters, it.sort_by)}"
        if it.text and not any("theme" in d for d in dropped):
            intro += f' about "{it.text}"'
        intro += ":"
        if dropped:
            intro = ("Nothing matched everything, so I relaxed " + ", ".join(dict.fromkeys(dropped)) +
                     ". " + intro)
        return Reply(intro, res["id"].tolist())
