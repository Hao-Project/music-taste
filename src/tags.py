"""Fetch crowd-sourced genre tags per artist and cache them on disk.

Providers:
  lastfm       artist.getTopTags, richer and more current. Needs LASTFM_API_KEY (free).
  musicbrainz  artist search, no key, but ~1 request/second and sparser tags.

Tags are cached in cache/tags_<provider>.json, so re-runs only fetch new artists.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

from load import ROOT, load_tracks

CACHE_DIR = ROOT / "cache"
UA = {"User-Agent": "music-taste/0.1 (personal project)"}

# Last.fm tags that describe the listener or the era, not the music
LASTFM_NOISE = {
    "seen live", "favorites", "favourite", "favorite", "albums i own", "love", "awesome",
    "female vocalists", "male vocalists", "beautiful", "good", "amazing", "best",
}


def fetch_lastfm(artist, key):
    r = requests.get(
        "https://ws.audioscrobbler.com/2.0/",
        params={"method": "artist.gettoptags", "artist": artist, "api_key": key,
                "format": "json", "autocorrect": 1},
        headers=UA, timeout=20,
    )
    r.raise_for_status()
    tags = r.json().get("toptags", {}).get("tag", [])
    return {t["name"].lower(): int(t["count"]) for t in tags
            if int(t["count"]) >= 5 and t["name"].lower() not in LASTFM_NOISE}


def fetch_musicbrainz(artist, _key=None):
    time.sleep(1.1)  # MusicBrainz asks for at most 1 request/second
    r = requests.get(
        "https://musicbrainz.org/ws/2/artist/",
        params={"query": f'artist:"{artist}"', "fmt": "json", "limit": 1},
        headers=UA, timeout=20,
    )
    r.raise_for_status()
    hits = r.json().get("artists", [])
    # Only trust a confident match, otherwise we tag the wrong band
    if not hits or hits[0].get("score", 0) < 90:
        return {}
    return {t["name"].lower(): int(t["count"]) for t in hits[0].get("tags", [])}


PROVIDERS = {"lastfm": fetch_lastfm, "musicbrainz": fetch_musicbrainz}


def fetch_all(provider="lastfm", limit=None):
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"tags_{provider}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    key = os.environ.get("LASTFM_API_KEY")
    if provider == "lastfm" and not key:
        sys.exit("Set LASTFM_API_KEY first (free key: https://www.last.fm/api/account/create)")

    # Most-played artists first, so a partial run is still useful
    artists = (load_tracks().groupby("artist")["plays"].sum()
               .sort_values(ascending=False).index.tolist())
    todo = [a for a in artists if a not in cache][:limit]
    print(f"{len(cache)} cached, {len(todo)} to fetch ({provider})")

    for i, artist in enumerate(todo, 1):
        try:
            cache[artist] = PROVIDERS[provider](artist, key)
        except requests.RequestException as e:
            print(f"  skip {artist!r}: {e}")
            continue
        if i % 25 == 0:
            path.write_text(json.dumps(cache, ensure_ascii=False))
            print(f"  {i}/{len(todo)}")
        if provider == "lastfm":
            time.sleep(0.2)
    path.write_text(json.dumps(cache, ensure_ascii=False))
    return cache


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "lastfm"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    fetch_all(provider, limit)
