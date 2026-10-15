"""Build output/mood.html: my library on a valence x arousal map, with sources kept apart.

Usage: python mood_report.py
"""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from load import ROOT
from mood import CACHE, build_moods
from report import NOT_GENRE

OUT = ROOT / "output"
SOURCE_COLORS = {"last.fm": "#2a7ab0", "LLM generated": "#e07b39"}
QUADRANTS = ["happy / energetic", "calm / content", "sad / melancholic", "tense / angry", "neutral"]


def weighted(df, col, w="plays"):
    w = df[w].clip(lower=1)
    return (df[col] * w).sum() / w.sum()


def top_genre(artist_tags):
    """artist -> most-voted Last.fm tag that is a genre, for grouping moods by genre."""
    out = {}
    for artist, tags in artist_tags.items():
        tags = {t: c for t, c in tags.items() if t not in NOT_GENRE}
        if tags:
            out[artist] = max(tags, key=tags.get)
    return out


def jitter(s, scale=0.025, seed=0):
    # Artist-level points share exact coordinates; a little spread makes the density visible
    return s + np.random.default_rng(seed).normal(0, scale, len(s))


def build():
    OUT.mkdir(exist_ok=True)
    m = build_moods()
    placed = m[m.source.notna()].copy()
    figs = []

    counts = m.groupby(["source", "level"], dropna=False).size()
    summary = pd.DataFrame({"tracks": counts, "share": (counts / len(m) * 100).round(1)})
    summary.index = [" / ".join(str(x) for x in i) if pd.notna(i[0]) else "not placed"
                     for i in summary.index]

    # --- the map
    placed["x"] = jitter(placed.valence)
    placed["y"] = jitter(placed.arousal, seed=1)
    placed["label"] = placed.source + " (" + placed.level + ")"
    placed["size"] = np.sqrt(placed.plays.clip(lower=1))
    f = px.scatter(
        placed, x="x", y="y", color="source", symbol="level", size="size", size_max=18,
        color_discrete_map=SOURCE_COLORS, opacity=0.6,
        hover_data={"name": True, "artist": True, "plays": True, "mood_words": True,
                    "label": True, "x": False, "y": False, "size": False,
                    "source": False, "level": False},
        title="Every placed track (size = plays). Squares/diamonds = artist-level, not song-specific",
        labels={"x": "valence  (sad → happy)", "y": "arousal  (calm → energetic)"},
    )
    f.add_hline(y=0, line_color="#bbb"); f.add_vline(x=0, line_color="#bbb")
    f.update_xaxes(range=[-1.05, 1.05]); f.update_yaxes(range=[-1.05, 1.05])
    for x, y, t in [(0.7, 0.95, "happy / energetic"), (0.7, -0.95, "calm / content"),
                    (-0.7, -0.95, "sad / melancholic"), (-0.7, 0.95, "tense / angry")]:
        f.add_annotation(x=x, y=y, text=t, showarrow=False, font=dict(color="#888"))
    f.update_layout(height=700)
    figs.append(f)

    # --- quadrants: owned vs played, per source
    q = []
    for src, g in placed.groupby("source"):
        for quad in QUADRANTS:
            sub = g[g.quadrant == quad]
            q.append({"source": src, "quadrant": quad,
                      "% of tracks": len(sub) / len(g) * 100,
                      "% of plays": sub.plays.sum() / max(g.plays.sum(), 1) * 100})
    q = pd.DataFrame(q).melt(id_vars=["source", "quadrant"], var_name="measure", value_name="pct")
    f = px.bar(q, x="pct", y="quadrant", color="measure", barmode="group", orientation="h",
               facet_col="source", title="Mood quadrants: what I own vs what I play, by source",
               labels={"pct": "%", "quadrant": ""},
               category_orders={"quadrant": QUADRANTS})
    f.update_layout(height=420)
    figs.append(f)

    # --- artists on the map, play-weighted
    a = (placed.groupby("artist")
         .apply(lambda g: pd.Series({
             "valence": weighted(g, "valence"), "arousal": weighted(g, "arousal"),
             "plays": g.plays.sum(), "tracks": len(g),
             "source": "LLM generated" if (g.source == "LLM generated").mean() > 0.5 else "last.fm"}))
         .sort_values("plays", ascending=False).head(40).reset_index())
    f = px.scatter(a, x="valence", y="arousal", text="artist", size="plays", color="source",
                   color_discrete_map=SOURCE_COLORS, size_max=40,
                   title="Top 40 artists by plays, at the play-weighted mood of their tracks")
    f.update_traces(textposition="top center", textfont_size=10)
    f.add_hline(y=0, line_color="#bbb"); f.add_vline(x=0, line_color="#bbb")
    f.update_layout(height=700)
    figs.append(f)

    # --- mood by genre
    genre = top_genre(json.loads((CACHE / "tags_lastfm.json").read_text()))
    placed["genre"] = placed.artist.map(genre)
    gg = (placed.dropna(subset=["genre"]).groupby("genre")
          .apply(lambda g: pd.Series({"valence": weighted(g, "valence"),
                                      "arousal": weighted(g, "arousal"),
                                      "plays": g.plays.sum(), "tracks": len(g)}))
          .query("tracks >= 15").reset_index())
    f = px.scatter(gg, x="valence", y="arousal", text="genre", size="plays", size_max=40,
                   title="Main genres (artist's top Last.fm tag, 15+ tracks) by average mood")
    f.update_traces(textposition="top center", marker_color="#777")
    f.add_hline(y=0, line_color="#bbb"); f.add_vline(x=0, line_color="#bbb")
    f.update_layout(height=600)
    figs.append(f)

    # --- tables
    overall = {src: (weighted(g, "valence"), weighted(g, "arousal"))
               for src, g in placed.groupby("source")}
    extremes = []
    for label, col, asc in [("Saddest", "valence", True), ("Happiest", "valence", False),
                            ("Calmest", "arousal", True), ("Most energetic", "arousal", False)]:
        top = placed[placed.level.isin(["track", "song"])].sort_values(col, ascending=asc)
        extremes.append(f"<h3>{label} (song-specific only)</h3>" + top.head(10)[
            ["name", "artist", "plays", "valence", "arousal", "source", "mood_words"]
        ].round(2).to_html(index=False))
    m.drop(columns=["size"], errors="ignore").round(3).to_csv(OUT / "moods.csv", index=False)

    html = ["<html><head><meta charset='utf-8'><title>Mood map</title>",
            "<style>body{font-family:-apple-system,sans-serif;max-width:1000px;margin:2em auto;"
            "padding:0 1em;color:#222}table{border-collapse:collapse;font-size:13px}"
            "td,th{padding:3px 10px;border-bottom:1px solid #ddd;text-align:left}"
            ".llm{color:#e07b39;font-weight:600}.lfm{color:#2a7ab0;font-weight:600}</style>"
            "</head><body><h1>Mood map of my library</h1>",
            "<p>Valence runs from sad (−1) to happy (+1), arousal from calm (−1) to energetic (+1). "
            "<span class='lfm'>last.fm</span> points come from crowd-sourced mood tags, scored with "
            "the Warriner et al. (2013) word norms plus a few hand ratings. "
            "<span class='llm'>LLM generated</span> points are guesses by Claude: "
            "<i>song</i> means it knew the song, <i>artist</i> means one guess for the artist, "
            "reused for every unknown song by them. Artist-level points of either source are "
            "not song-specific.</p>",
            "<h2>Where the numbers come from</h2>" + summary.to_html()]
    html.append("<p>Play-weighted average mood: " + "; ".join(
        f"{s}: valence {v:+.2f}, arousal {a:+.2f}" for s, (v, a) in overall.items()) + "</p>")
    for i, fig in enumerate(figs):
        fig.update_layout(template="simple_white", margin=dict(l=10, r=10, t=60, b=10))
        html.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    html += extremes
    html.append("<p>Full table: output/moods.csv</p></body></html>")
    (OUT / "mood.html").write_text("\n".join(html))
    print(summary)
    print(f"wrote {OUT / 'mood.html'}")


if __name__ == "__main__":
    build()
