"""Parse the Apple Music Library.xml export into a clean track table with a taste score."""
import plistlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LIBRARY_XML = ROOT / "data" / "Library.xml"


def load_env(path=ROOT / ".env"):
    """Read KEY=value lines from .env into os.environ, without overriding what's already set."""
    import os
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

KEEP = {
    "Persistent ID": "id",
    "Name": "name",
    "Artist": "artist",
    "Album Artist": "album_artist",
    "Album": "album",
    "Genre": "genre",
    "Year": "year",
    "Total Time": "duration_ms",
    "Date Added": "date_added",
    "Play Count": "plays",
    "Play Date UTC": "last_played",
    "Skip Count": "skips",
    "Loved": "loved",
    "Compilation": "compilation",
}


def load_tracks(path=LIBRARY_XML):
    with open(path, "rb") as f:
        lib = plistlib.load(f)
    export_date = pd.Timestamp(lib["Date"]).tz_localize(None)

    rows = [{new: t.get(old) for old, new in KEEP.items()} for t in lib["Tracks"].values()]
    df = pd.DataFrame(rows)

    for col in ("plays", "skips"):
        df[col] = df[col].fillna(0).astype(int)
    for col in ("loved", "compilation"):
        df[col] = df[col].eq(True)
    for col in ("date_added", "last_played"):
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)

    # Compilation albums carry the real artist on the track, so prefer Artist.
    df["artist"] = df["artist"].fillna(df["album_artist"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["minutes"] = df["duration_ms"] / 60000

    df.attrs["export_date"] = export_date
    return add_taste_score(df, export_date)


def add_taste_score(df, export_date):
    """Score 0-100 of how much I like each track.

    Raw play count favours old tracks, so plays are divided by years in the library
    (floored at 3 months so a track added last week isn't over-weighted). Skips pull the
    score down, and Loved adds a fixed bonus. The result is a percentile within the library.
    """
    years = ((export_date - df["date_added"]).dt.days / 365.25).clip(lower=0.25)
    years = years.fillna(years.median())  # ~1.5% of tracks have no Date Added
    df["years_in_library"] = years
    df["plays_per_year"] = df["plays"] / years
    df["skip_ratio"] = df["skips"] / (df["plays"] + df["skips"]).replace(0, np.nan)
    df["skip_ratio"] = df["skip_ratio"].fillna(0)

    raw = np.log1p(df["plays_per_year"]) * (1 - 0.5 * df["skip_ratio"]) + 1.0 * df["loved"]
    df["score"] = (raw.rank(pct=True) * 100).round(1)
    return df


if __name__ == "__main__":
    tracks = load_tracks()
    print(tracks.shape)
    print(tracks.head())
