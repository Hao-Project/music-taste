"""Build output/mood.html: LLM track mood, LLM artist mood, and a check against Last.fm.

The maps use LLM ratings only, so every point is on the same scale from the same rater.
Last.fm appears only in the robustness check at the end.

Usage: python mood_report.py
"""
import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def zero_lines(f, **where):
    """Dashed lines at valence 0 and arousal 0, each running from -1 to 1."""
    style = dict(color="#444", dash="dash", width=2)
    f.add_shape(type="line", x0=-1, x1=1, y0=0, y1=0, line=style, **where)
    f.add_shape(type="line", x0=0, x1=0, y0=-1, y1=1, line=style, **where)
    return f


def quadrant_axes(f):
    zero_lines(f)
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


def top_artists_scatter(tr, max_n=20, default_n=5):
    """Rated songs from my N most-played artists, one colour per artist; a slider sets N."""
    rank = tr.groupby("artist").plays.sum().sort_values(ascending=False).head(max_n).index
    colours = px.colors.qualitative.Dark24
    biggest = np.sqrt(tr[tr.artist.isin(rank)].plays.max())
    # Many songs share exact ratings; a small jitter keeps them from hiding each other.
    # Hover shows the real values.
    rng = np.random.default_rng(0)
    f = go.Figure()
    for i, artist in enumerate(rank):
        g = tr[tr.artist == artist]
        f.add_trace(go.Scatter(
            x=g.track_valence + rng.normal(0, 0.02, len(g)),
            y=g.track_arousal + rng.normal(0, 0.02, len(g)),
            mode="markers", name=artist, visible=i < default_n,
            marker=dict(color=colours[i % len(colours)], opacity=0.8,
                        size=6 + 16 * np.sqrt(g.plays.clip(lower=1)) / biggest,
                        line=dict(width=0.5, color="white")),
            customdata=np.stack([g.name + " by " + g.artist, g.plays,
                                 g.track_valence, g.track_arousal], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} plays<br>"
                          "valence %{customdata[2]:.2f}, arousal %{customdata[3]:.2f}<extra></extra>"))

    def title(n):
        songs = int(tr.artist.isin(rank[:n]).sum())
        return f"Songs by my top {n} artist{'s' if n > 1 else ''} ({songs} rated songs; size = plays)"

    steps = [dict(method="update", label=str(n),
                  args=[{"visible": [i < n for i in range(len(rank))]}, {"title": title(n)}])
             for n in range(1, len(rank) + 1)]
    f.update_layout(
        title=title(default_n), height=720, xaxis_title=AXES["x"], yaxis_title=AXES["y"],
        sliders=[dict(active=default_n - 1, steps=steps, pad=dict(t=50),
                      currentvalue=dict(prefix="Top artists: "))])
    return quadrant_axes(f)


def write_standalone(page, path):
    """Same report with plotly.js inlined, so it opens with no network (e.g. inside WeChat,
    where the plotly CDN can be slow or blocked). Also adds a phone viewport and lets wide
    tables scroll sideways instead of overflowing the screen."""
    from plotly.offline import get_plotlyjs
    tag = re.search(r'<script[^>]*src="https://cdn\.plot\.ly/[^"]+"[^>]*></script>', page)
    if not tag:
        raise RuntimeError("plotly CDN script tag not found; can't inline plotly.js")
    page = page.replace(tag.group(0), f"<script>{get_plotlyjs()}</script>", 1)
    page = page.replace(
        "<meta charset='utf-8'>",
        "<meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<style>table{display:block;overflow-x:auto;max-width:100%}</style>", 1)
    path.write_text(page)


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
        zero_lines(f, row=1, col=i)
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

    # --- track heatmap: where my listening sits, over every rated song
    tr = m.dropna(subset=["track_valence"]).copy()
    # Ratings come in 0.05-0.1 steps, so bins are centred on multiples of 0.1
    bins = dict(start=-1.05, end=1.05, size=0.1)
    f = go.Figure(go.Histogram2d(
        x=tr.track_valence, y=tr.track_arousal, z=tr.plays, histfunc="sum",
        xbins=bins, ybins=bins, colorscale="YlOrRd", colorbar=dict(title="plays"),
        hovertemplate="valence %{x}<br>arousal %{y}<br>%{z} plays<extra></extra>"))
    f.update_layout(title=f"Track mood heatmap: plays across all {len(tr):,} rated songs (0.1 × 0.1 cells)",
                    xaxis_title=AXES["x"], yaxis_title=AXES["y"], height=640)
    figs.append(("Track mood", quadrant_axes(f)))

    # --- track scatter: rated songs from my top N artists, N picked with a slider
    figs.append(("", top_artists_scatter(tr)))
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
    page = "\n".join(html)
    (OUT / "mood.html").write_text(page)
    write_standalone(page, OUT / "mood_standalone.html")

    print(cov.to_string(index=False))
    print(f"\nrobustness: {len(both)} songs\n{stats.to_string(index=False)}")
    print(f"quadrant agreement: track {agree:.0%}, artist {agree_artist:.0%}")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    build()
