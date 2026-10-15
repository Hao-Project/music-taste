"""Build output/mood.html: LLM track mood, LLM artist mood, and a check against Last.fm.

The maps use LLM ratings only, so every point is on the same scale from the same rater.
Last.fm appears only in the robustness check at the end.

Usage: python mood_report.py
"""
import json

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

from load import ROOT
from mood import CACHE, build_moods
from report import NOT_GENRE

OUT = ROOT / "output"
QUADRANTS = ["happy / energetic", "calm / content", "sad / melancholic", "tense / angry", "neutral"]
AXES = {"x": "valence  (sad → happy)", "y": "arousal  (calm → energetic)"}

# Songs whose Last.fm tags I saw while debugging, before rating them. Their LLM ratings
# aren't blind, so they are left out of the robustness check.
NOT_BLIND = {("Beach House", "Space Song"), ("Metric", "Lost Kitten"),
             ("The Marías", "Over the Moon"), ("The Marías", "All I Really Want Is You"),
             ("Caravan Palace", "Lone Digger (Mixed)")}


def weighted(df, col):
    w = df.plays.clip(lower=1)
    return (df[col] * w).sum() / w.sum()


def quadrant_axes(f):
    f.add_hline(y=0, line_color="#bbb")
    f.add_vline(x=0, line_color="#bbb")
    f.update_xaxes(range=[-1.05, 1.05])
    f.update_yaxes(range=[-1.05, 1.05])
    for x, y, t in [(0.7, 0.97, "happy / energetic"), (0.7, -0.97, "calm / content"),
                    (-0.7, -0.97, "sad / melancholic"), (-0.7, 0.97, "tense / angry")]:
        f.add_annotation(x=x, y=y, text=t, showarrow=False, font=dict(color="#999"))
    return f


def quadrant_bars(df, col, title):
    rows = []
    for q in QUADRANTS:
        sub = df[df[col] == q]
        rows += [{"quadrant": q, "measure": "% of tracks", "pct": len(sub) / len(df) * 100},
                 {"quadrant": q, "measure": "% of plays",
                  "pct": sub.plays.sum() / max(df.plays.sum(), 1) * 100}]
    f = px.bar(pd.DataFrame(rows), x="pct", y="quadrant", color="measure", barmode="group",
               orientation="h", title=title, labels={"pct": "%", "quadrant": ""},
               category_orders={"quadrant": QUADRANTS})
    f.update_layout(height=380)
    return f


def corr_ci(x, y, method="pearson", n_boot=2000, seed=0):
    """Correlation with a 95% bootstrap interval; small samples need the interval."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if method == "spearman":
        x, y = pd.Series(x).rank().values, pd.Series(y).rank().values
    r = np.corrcoef(x, y)[0, 1]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            boots.append(np.corrcoef(x[i], y[i])[0, 1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return r, lo, hi


def robustness(m):
    """Compare LLM track ratings with Last.fm tags on songs that have both."""
    both = m.dropna(subset=["track_valence", "lfm_valence"]).copy()
    both = both[[(a, n) not in NOT_BLIND for a, n in zip(both.artist, both.name)]]
    lfm_only = m.lfm_valence.notna().sum()

    stats = []
    for dim in ("valence", "arousal"):
        for pred, label in (("track", "LLM track rating"), ("artist", "LLM artist rating")):
            r, lo, hi = corr_ci(both[f"{pred}_{dim}"], both[f"lfm_{dim}"])
            rs, _, _ = corr_ci(both[f"{pred}_{dim}"], both[f"lfm_{dim}"], "spearman", n_boot=200)
            mae = (both[f"{pred}_{dim}"] - both[f"lfm_{dim}"]).abs().mean()
            stats.append({"dimension": dim, "compared with last.fm": label,
                          "Pearson r": round(r, 2), "95% CI": f"{lo:.2f} to {hi:.2f}",
                          "Spearman ρ": round(rs, 2), "mean abs. difference": round(mae, 2)})
    stats = pd.DataFrame(stats)
    agree = (both.track_quadrant == both.lfm_quadrant).mean()
    agree_artist = (both.artist_quadrant == both.lfm_quadrant).mean()

    f = make_subplots(rows=1, cols=2, subplot_titles=("Valence", "Arousal"))
    for i, dim in enumerate(("valence", "arousal"), 1):
        sc = px.scatter(both, x=f"lfm_{dim}", y=f"track_{dim}",
                        hover_data=["name", "artist", "lfm_words"]).data[0]
        sc.marker.color = "#2a7ab0"
        f.add_trace(sc, row=1, col=i)
        f.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1, line=dict(color="#bbb", dash="dot"),
                    row=1, col=i)
        f.update_xaxes(title_text="last.fm", range=[-1.05, 1.05], row=1, col=i)
        f.update_yaxes(title_text="LLM track rating", range=[-1.05, 1.05], row=1, col=i)
    f.update_layout(height=430, showlegend=False,
                    title="LLM track rating vs Last.fm, same songs (dotted line = perfect agreement)")

    both["gap"] = np.hypot(both.track_valence - both.lfm_valence,
                           both.track_arousal - both.lfm_arousal)
    worst = both.sort_values("gap", ascending=False).head(10)[
        ["name", "artist", "lfm_words", "lfm_valence", "lfm_arousal",
         "track_valence", "track_arousal"]].round(2)
    return both, lfm_only, stats, agree, agree_artist, f, worst


def build():
    OUT.mkdir(exist_ok=True)
    m = build_moods()
    figs = []

    # --- coverage
    cov = pd.DataFrame([
        {"rating": label, "tracks": int(m[col].notna().sum()),
         "% of tracks": round(m[col].notna().mean() * 100, 1),
         "% of plays": round(m.loc[m[col].notna(), "plays"].sum() / m.plays.sum() * 100, 1)}
        for label, col in [("LLM track mood", "track_valence"),
                           ("LLM artist mood", "artist_valence"),
                           ("last.fm track tags (check only)", "lfm_valence")]])

    # --- track map
    tr = m.dropna(subset=["track_valence"]).copy()
    tr["size"] = np.sqrt(tr.plays.clip(lower=1))
    f = px.scatter(tr, x="track_valence", y="track_arousal", size="size", size_max=18,
                   opacity=0.55, color_discrete_sequence=["#e07b39"],
                   hover_data={"name": True, "artist": True, "plays": True,
                               "track_valence": ":.2f", "track_arousal": ":.2f", "size": False},
                   title=f"Track mood: {len(tr):,} songs the LLM knows (size = plays)",
                   labels={"track_valence": AXES["x"], "track_arousal": AXES["y"]})
    f.update_layout(height=680)
    figs.append(("Track mood", quadrant_axes(f)))
    figs.append(("", quadrant_bars(tr, "track_quadrant",
                                   "Track mood quadrants: owned vs played (rated songs only)")))

    # --- artist map, one dot per artist
    a = (m.groupby("artist")
         .agg(valence=("artist_valence", "first"), arousal=("artist_arousal", "first"),
              plays=("plays", "sum"), tracks=("id", "size"))
         .dropna().sort_values("plays", ascending=False).reset_index())
    a["label"] = np.where(a.index < 30, a.artist, "")
    f = px.scatter(a, x="valence", y="arousal", size="plays", size_max=45, text="label",
                   opacity=0.6, color_discrete_sequence=["#7a5bb5"],
                   hover_data={"artist": True, "plays": True, "tracks": True, "label": False},
                   title=f"Artist mood: all {len(a)} artists, one dot each (size = plays, top 30 labelled)",
                   labels={"valence": AXES["x"], "arousal": AXES["y"]})
    f.update_traces(textposition="top center", textfont_size=10)
    f.update_layout(height=720)
    figs.append(("Artist mood", quadrant_axes(f)))
    figs.append(("", quadrant_bars(m.dropna(subset=["artist_quadrant"]), "artist_quadrant",
                                   "Artist mood quadrants: share of my tracks and plays by the artist's mood")))

    # --- genre, using artist mood so every artist counts
    raw = json.loads((CACHE / "tags_lastfm.json").read_text())
    genre = {}
    for artist, tags in raw.items():
        tags = {t: c for t, c in tags.items() if t not in NOT_GENRE}
        if tags:
            genre[artist] = max(tags, key=tags.get)
    m["genre"] = m.artist.map(genre)
    gg = (m.dropna(subset=["genre"]).groupby("genre")
          .apply(lambda g: pd.Series({"valence": weighted(g, "artist_valence"),
                                      "arousal": weighted(g, "artist_arousal"),
                                      "plays": g.plays.sum(), "tracks": len(g)}))
          .query("tracks >= 15").reset_index())
    f = px.scatter(gg, x="valence", y="arousal", text="genre", size="plays", size_max=40,
                   title="Main genres (artist's top Last.fm tag, 15+ tracks) at their artists' average mood",
                   labels={"valence": AXES["x"], "arousal": AXES["y"]})
    f.update_traces(textposition="top center", marker_color="#888")
    f.update_layout(height=560)
    figs.append(("Mood by genre", quadrant_axes(f)))

    # --- extremes, track mood only
    extremes = []
    for label, col, asc in [("Saddest", "track_valence", True), ("Happiest", "track_valence", False),
                            ("Calmest", "track_arousal", True), ("Most energetic", "track_arousal", False)]:
        extremes.append(f"<h3>{label}</h3>" + tr.sort_values(col, ascending=asc).head(10)[
            ["name", "artist", "plays", "track_valence", "track_arousal"]].round(2).to_html(index=False))

    # --- robustness
    both, lfm_only, stats, agree, agree_artist, rob_fig, worst = robustness(m)

    m.round(3).to_csv(OUT / "moods.csv", index=False)

    html = ["<html><head><meta charset='utf-8'><title>Mood map</title>",
            "<style>body{font-family:-apple-system,sans-serif;max-width:1000px;margin:2em auto;"
            "padding:0 1em;color:#222;line-height:1.45}table{border-collapse:collapse;font-size:13px}"
            "td,th{padding:3px 10px;border-bottom:1px solid #ddd;text-align:left}"
            ".note{background:#f6f6f2;padding:.8em 1em;border-radius:6px}</style>"
            "</head><body><h1>Mood map of my library</h1>",
            "<p>Valence runs from sad (−1) to happy (+1), arousal from calm (−1) to energetic (+1). "
            "Both maps use LLM (Claude) ratings only, so the scale is consistent. There are two "
            "separate ratings that are never mixed: <b>track mood</b> rates a song the LLM knows "
            "well, and songs it doesn't know are left out; <b>artist mood</b> rates each artist's "
            "general sound. Last.fm is used only in the robustness check at the end.</p>",
            "<h2>Coverage</h2>" + cov.to_html(index=False)]
    for i, (heading, fig) in enumerate(figs):
        fig.update_layout(template="simple_white", margin=dict(l=10, r=10, t=60, b=10))
        if heading:
            html.append(f"<h2>{heading}</h2>")
        html.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    html.append("<h2>Extremes (track mood)</h2>" + "".join(extremes))

    html.append("<h2>Robustness check: LLM vs Last.fm</h2>")
    html.append(
        f"<p>{lfm_only} songs have Last.fm mood tags; {len(both)} of them also have an LLM track "
        f"rating and were rated blind to their tags ({len(NOT_BLIND)} songs whose tags I saw "
        "beforehand are excluded). The LLM artist rating is shown as a baseline: if the song-level "
        "rating is worth having, it should agree with Last.fm better than the artist's general "
        "mood does.</p>")
    html.append(stats.to_html(index=False))
    html.append(f"<p>Same quadrant as Last.fm: LLM track rating {agree:.0%}, "
                f"LLM artist rating {agree_artist:.0%}.</p>")
    html.append(rob_fig.update_layout(template="simple_white").to_html(full_html=False,
                                                                        include_plotlyjs=False))
    html.append("<h3>Largest disagreements</h3>" + worst.to_html(index=False))
    html.append(
        "<div class='note'><b>How to read this.</b> Agreement with Last.fm is not the same as "
        "accuracy: Last.fm scores here often rest on one or two tags (\"ethereal\", \"cute\"), so "
        "they are noisy too. The overlap is also not a random sample. Songs with Last.fm mood "
        "tags are the more popular ones, which are also the songs the LLM knows best, so "
        "agreement here likely overstates how well the LLM does on the niche songs in the "
        "library.</div>")
    html.append("<p>Full table: output/moods.csv</p></body></html>")
    (OUT / "mood.html").write_text("\n".join(html))

    print(cov.to_string(index=False))
    print(f"\nrobustness: {len(both)} songs\n{stats.to_string(index=False)}")
    print(f"quadrant agreement: track {agree:.0%}, artist {agree_artist:.0%}")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    build()
