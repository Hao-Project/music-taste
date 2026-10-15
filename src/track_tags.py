"""Fetch Last.fm tags for every track (track.getTopTags) and cache them by track id.

Artist tags (tags.py) are mostly genres. Mood words like "melancholic" or "upbeat" are
attached to individual songs, so the mood analysis needs this per-track fetch.

Usage: python track_tags.py [limit]    needs LASTFM_API_KEY in the environment or .env
Cache: cache/track_tags_lastfm.json  ({track id: {tag: count}}), re-runs only fetch new tracks.
"""
import json
import os
import re
import sys
import time

import requests

from load import ROOT, load_env, load_tracks
from tags import LASTFM_NOISE, UA

load_env()
CACHE = ROOT / "cache" / "track_tags_lastfm.json"

# "Song (feat. X)", "Song - 2011 Remaster", "Song (Remastered)": Last.fm often files these
# under the plain title, so retry with the suffix stripped when the full title has no tags.
SUFFIX = re.compile(r"\s*(\((feat|with|ft)\.?[^)]*\)|\[(feat|with|ft)\.?[^\]]*\]"
                    r"|-\s*[^-]*remaster[^-]*$|\([^)]*remaster[^)]*\))", re.I)


def clean_title(name):
    return SUFFIX.sub("", name).strip()


def fetch(artist, title, key):
    r = requests.get(
        "https://ws.audioscrobbler.com/2.0/",
        params={"method": "track.gettoptags", "artist": artist, "track": title,
                "api_key": key, "format": "json", "autocorrect": 1},
        headers=UA, timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    if "error" in data:  # error 6 = track not found, which just means no tags
        return {}
    tags = data.get("toptags", {}).get("tag", [])
    return {t["name"].lower(): int(t["count"]) for t in tags
            if int(t["count"]) >= 1 and t["name"].lower() not in LASTFM_NOISE}


def fetch_all(limit=None):
    key = os.environ.get("LASTFM_API_KEY")
    if not key:
        sys.exit("Set LASTFM_API_KEY in .env or the environment first")
    CACHE.parent.mkdir(exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    tracks = load_tracks().sort_values("plays", ascending=False)
    todo = tracks[~tracks.id.isin(cache) & tracks.artist.notna() & tracks.name.notna()]
    todo = todo.head(limit) if limit else todo
    print(f"{len(cache)} cached, {len(todo)} to fetch")

    for i, t in enumerate(todo.itertuples(), 1):
        try:
            tags = fetch(t.artist, t.name, key)
            short = clean_title(t.name)
            if not tags and short != t.name:
                time.sleep(0.2)
                tags = fetch(t.artist, short, key)
        except requests.RequestException as e:
            print(f"  skip {t.artist} - {t.name}: {e}")
            continue
        cache[t.id] = tags
        if i % 100 == 0:
            CACHE.write_text(json.dumps(cache, ensure_ascii=False))
            print(f"  {i}/{len(todo)}")
        time.sleep(0.2)  # Last.fm asks for no more than ~5 requests/second
    CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    tagged = sum(1 for v in cache.values() if v)
    print(f"done: {tagged}/{len(cache)} tracks have at least one tag")


if __name__ == "__main__":
    fetch_all(int(sys.argv[1]) if len(sys.argv) > 1 else None)
