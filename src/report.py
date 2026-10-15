"""Build output/report.html: a self-contained look at my listening, plus CSVs for later steps.

Usage: python report.py [provider]    (provider = tag cache to use, default musicbrainz)
"""
import json
import sys

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from load import ROOT, load_tracks

OUT = ROOT / "output"
CACHE = ROOT / "cache"

# Tags that name a place, a language or a vocalist type rather than a sound
NOT_GENRE = {
    "american", "british", "english", "scottish", "canadian", "canada", "france", "french",
    "japanese", "japan", "korean", "uk", "usa", "united states", "new zealand", "australian",
    "german", "germany", "swedish", "sweden", "female vocals", "female vocalists",
    "male vocalists", "seen live", "the lost generation", "poggers", "00s", "10s", "80s", "90s",
}


def artist_tag_weights(provider):
    """artist -> {tag: share}, shares sum to 1 per artist so prolific taggers don't dominate."""
    path = CACHE / f"tags_{provider}.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out = {}
    for artist, tags in raw.items():
        tags = {t: c for t, c in tags.items() if t not in NOT_GENRE and c > 0}
        total = sum(tags.values())
        if total:
            out[artist] = {t: c / total for t, c in tags.items()}
    return out


def tag_profile(tracks, weights):
    """Per tag: share of what I listen to (plays) vs share of what I own (tracks)."""
    per_artist = tracks.groupby("artist").agg(plays=("plays", "sum"), n=("id", "size"))
    rows = []
    for artist, r in per_artist.iterrows():
        for tag, w in weights.get(artist, {}).items():
            rows.append((tag, w * r.plays, w * r.n))
    df = pd.DataFrame(rows, columns=["tag", "plays", "tracks"]).groupby("tag").sum()
    df["listen_share"] = df.plays / df.plays.sum() * 100
    df["library_share"] = df.tracks / df.tracks.sum() * 100
    return df.sort_values("plays", ascending=False)


def build(provider="musicbrainz"):
    OUT.mkdir(exist_ok=True)
    t = load_tracks()
    figs = []
    export = t.attrs["export_date"].date()

    # --- headline numbers
    hours = (t.plays * t.minutes).sum() / 60
    summary = (f"{len(t):,} tracks · {t.artist.nunique():,} artists · {t.plays.sum():,} plays "
               f"(~{hours:,.0f} hours) · {int(t.loved.sum())} loved · "
               f"{(t.plays == 0).mean():.0%} never played · library exported {export}")

    # --- Apple genre vs tag-based genre
    g = t.groupby("genre").agg(plays=("plays", "sum"), tracks=("id", "size")).sort_values("plays")
    g = g.tail(12)
    f = px.bar(g, x="plays", y=g.index, orientation="h", title="Apple's genre labels, by plays",
               labels={"y": ""})
    figs.append(f)

    weights = artist_tag_weights(provider)
    covered = t.artist.isin(weights).mean()
    prof = None
    if weights:
        prof = tag_profile(t, weights)
        top = prof.head(20).iloc[::-1]
        f = px.bar(top, x="listen_share", y=top.index, orientation="h",
                   title=f"Crowd-sourced tags, by share of plays ({covered:.0%} of tracks tagged)",
                   labels={"listen_share": "% of plays", "y": ""})
        figs.append(f)

        both = prof[prof.plays > prof.plays.sum() * 0.004].copy()
        both["tilt"] = both.listen_share - both.library_share
        both = both.sort_values("tilt")
        both = pd.concat([both.head(8), both.tail(8)])
        f = px.bar(both, x="tilt", y=both.index, orientation="h",
                   title="Tags I play more (right) or less (left) than I own, in % points",
                   labels={"tilt": "listen share − library share", "y": ""})
        figs.append(f)
        prof.to_csv(OUT / "tag_profile.csv")

    # --- artists
    a = (t.groupby("artist").agg(tracks=("id", "size"), plays=("plays", "sum"),
                                 loved=("loved", "sum"), avg_score=("score", "mean"),
                                 skip_ratio=("skip_ratio", "mean"))
         .sort_values("plays", ascending=False))
    a.round(2).to_csv(OUT / "artists.csv")
    top = a.head(20).iloc[::-1]
    figs.append(px.bar(top, x="plays", y=top.index, orientation="h", title="Top 20 artists by plays",
                       labels={"y": ""}))

    # Artists I have only a few tracks of but play heavily: strongest taste signal per song
    deep = a[a.tracks >= 3].sort_values("avg_score", ascending=False).head(15).iloc[::-1]
    figs.append(px.bar(deep, x="avg_score", y=deep.index, orientation="h",
                       title="Artists with the highest average track score (3+ tracks)",
                       labels={"avg_score": "average score (0-100)", "y": ""}))

    # --- time
    added = t.dropna(subset=["date_added"]).copy()
    added["added_year"] = added.date_added.dt.year
    f = px.bar(added.groupby("added_year").size().reset_index(name="tracks"),
               x="added_year", y="tracks", title="Tracks added per year")
    figs.append(f)

    t["decade"] = (t.year // 10 * 10).astype("Int64")
    d = t.dropna(subset=["decade"]).groupby("decade").agg(tracks=("id", "size"), plays=("plays", "sum"))
    d = d.reset_index()
    d["decade"] = d.decade.astype(str) + "s"
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_bar(x=d.decade, y=d.tracks, name="tracks owned")
    f.add_scatter(x=d.decade, y=d.plays, name="plays", secondary_y=True, mode="lines+markers")
    f.update_layout(title="Release decade: what I own vs what I play")
    figs.append(f)

    # --- tables
    loved = (t[t.loved].sort_values("score", ascending=False)
             [["name", "artist", "album", "plays", "skips", "score"]])
    loved.to_csv(OUT / "loved.csv", index=False)
    never = t[(t.plays == 0) & (t.years_in_library > 1)]
    top_tracks = (t.sort_values("score", ascending=False)
                  [["name", "artist", "plays", "skips", "loved", "years_in_library", "score"]]
                  .head(25).round(2))
    t.round(3).to_csv(OUT / "tracks.csv", index=False)

    html = ["<html><head><meta charset='utf-8'><title>My music taste</title>",
            "<style>body{font-family:-apple-system,sans-serif;max-width:900px;margin:2em auto;"
            "padding:0 1em;color:#222}table{border-collapse:collapse;font-size:13px}"
            "td,th{padding:3px 10px;border-bottom:1px solid #ddd;text-align:left}"
            "h2{margin-top:2em}</style></head><body>",
            f"<h1>My music taste</h1><p>{summary}</p>"]
    if prof is None:
        html.append("<p><b>No tag cache found yet</b>, so tag charts are skipped. "
                    "Run <code>python tags.py</code> first.</p>")
    for i, fig in enumerate(figs):
        fig.update_layout(template="simple_white", height=480, margin=dict(l=10, r=10, t=50, b=10),
                          showlegend=False if i != len(figs) - 1 else True)
        html.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    html.append("<h2>Top 25 tracks by taste score</h2>" + top_tracks.to_html(index=False))
    html.append(f"<h2>Loved ({len(loved)})</h2>" + loved.to_html(index=False))
    html.append(f"<h2>Added over a year ago and never played ({len(never)})</h2>"
                "<p>Candidates to prune, or tracks I added and forgot. Full list in output/tracks.csv.</p>")
    html.append("</body></html>")
    (OUT / "report.html").write_text("\n".join(html))
    print(f"wrote {OUT / 'report.html'}  (tag coverage {covered:.0%})")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "musicbrainz")
