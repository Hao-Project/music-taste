"""Place every track on a valence (sad <-> happy) x arousal (calm <-> energetic) map.

Sources, in order of preference:
  last.fm / track   the track's own Last.fm mood tags
  last.fm / artist  the artist's Last.fm mood tags, when the track has none
  LLM generated     a guess from cache/mood_llm.csv, when Last.fm has no mood tags at all

Mood tags are turned into numbers with the Warriner et al. (2013) word norms, rescaled to
-1..1. Words whose musical sense differs from everyday English ("dark", "lazy") use the hand
ratings in OVERRIDES instead. Only tags in MOOD_WORDS count, so genre tags like "dream pop"
are never scored as moods.

The Warriner file is CC BY-NC-ND, so it is downloaded into cache/ and never committed.
"""
import json

import numpy as np
import pandas as pd
import requests

from load import ROOT, load_tracks

CACHE = ROOT / "cache"
WARRINER_URL = "https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv"

MOOD_WORDS = """
happy sad melancholy mellow chill relaxing calm peaceful dreamy ethereal atmospheric dark
depressing upbeat energetic aggressive angry uplifting euphoric romantic sexy sensual nostalgic
bittersweet haunting eerie creepy fun playful cheerful joyful groovy quirky epic intense
soothing gentle soft lonely heartbreak longing hopeful triumphant anxious moody introspective
contemplative hypnotic lush warm cold angsty brooding gloomy somber wistful sentimental tender
sweet cute sunny party lazy dramatic emotional passionate powerful mysterious spooky serene
""".split()

# Tag spellings that mean the same mood
ALIASES = {
    "melancholic": "melancholy", "chillout": "chill", "chill out": "chill", "chilled": "chill",
    "relax": "relaxing", "relaxed": "relaxing", "sombre": "somber", "sad songs": "sad",
    "happy songs": "happy", "feel good": "feel good", "feelgood": "feel good",
    "heartbreaking": "heartbreak", "heartbroken": "heartbreak", "energy": "energetic",
}

# Hand ratings (valence, arousal) on the -1..1 scale, for words missing from Warriner or whose
# everyday rating doesn't fit music. These are my judgements, not published norms.
OVERRIDES = {
    "dark": (-0.45, 0.0),        # Warriner: neutral (night-time sense)
    "dreamy": (0.35, -0.4),      # Warriner: fairly aroused
    "relaxing": (0.5, -0.7),     # Warriner: mid arousal
    "peaceful": (0.6, -0.7),
    "lazy": (0.2, -0.6),         # Warriner: negative (character flaw sense)
    "chill": (0.25, -0.5),
    "ethereal": (0.3, -0.45),    # missing from Warriner
    "haunting": (-0.35, 0.05),
    "longing": (-0.35, -0.1),
    "introspective": (-0.1, -0.35),
    "contemplative": (-0.05, -0.4),
    "angsty": (-0.5, 0.5),
    "somber": (-0.55, -0.35),
    "wistful": (-0.25, -0.35),
    "feel good": (0.75, 0.4),
    "rainy day": (-0.3, -0.5),
    "late night": (-0.1, -0.4),
}
# Warriner centres and spreads, used to rescale its 1-9 ratings to roughly -1..1
V_MID, V_SPAN = 5.0, 4.0
A_MID, A_SPAN = 4.2, 3.1


def warriner():
    path = CACHE / "warriner.csv"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        path.write_bytes(requests.get(WARRINER_URL, timeout=60).content)
    w = pd.read_csv(path, index_col=0).set_index("Word")
    return w["V.Mean.Sum"], w["A.Mean.Sum"]


def mood_lexicon():
    """tag -> (valence, arousal, source of the rating)"""
    v, a = warriner()
    lex = {}
    for word in MOOD_WORDS + list(OVERRIDES):
        if word in OVERRIDES:
            lex[word] = (*OVERRIDES[word], "hand")
        elif word in v.index:
            lex[word] = (float(np.clip((v[word] - V_MID) / V_SPAN, -1, 1)),
                         float(np.clip((a[word] - A_MID) / A_SPAN, -1, 1)), "warriner")
    return lex


def score_tags(tags, lex):
    """Weighted mean of the mood tags in {tag: count}. Returns (v, a, n_words, words) or None."""
    hits = {}
    for tag, count in tags.items():
        word = ALIASES.get(tag, tag)
        if word in lex and count > 0:
            hits[word] = hits.get(word, 0) + count
    if not hits:
        return None
    total = sum(hits.values())
    v = sum(lex[w][0] * c for w, c in hits.items()) / total
    a = sum(lex[w][1] * c for w, c in hits.items()) / total
    words = ", ".join(sorted(hits, key=hits.get, reverse=True))
    return v, a, len(hits), words


def quadrant(v, a):
    if abs(v) < 0.1 and abs(a) < 0.1:
        return "neutral"
    if v >= 0:
        return "happy / energetic" if a >= 0 else "calm / content"
    return "tense / angry" if a >= 0 else "sad / melancholic"


def load_llm():
    """cache/mood_llm.json: {artist: {"artist": [v, a], "songs": {title: [v, a]}}}

    "songs" holds ratings for songs the LLM actually knows. Other songs by that artist get
    the "artist" guess, labelled basis="artist" so it's clear it isn't song-specific.
    """
    path = CACHE / "mood_llm.json"
    return json.loads(path.read_text()) if path.exists() else {}


def llm_guess(llm, artist, title):
    entry = llm.get(artist)
    if not entry:
        return None
    if title in entry.get("songs", {}):
        return (*entry["songs"][title], "song")
    if entry.get("artist"):
        return (*entry["artist"], "artist")
    return None


def build_moods():
    tracks = load_tracks()
    lex = mood_lexicon()
    track_tags = json.loads((CACHE / "track_tags_lastfm.json").read_text())
    artist_tags = json.loads((CACHE / "tags_lastfm.json").read_text())
    llm = load_llm()

    rows = []
    for t in tracks.itertuples():
        row = {"id": t.id, "source": None, "level": None, "valence": np.nan, "arousal": np.nan,
               "mood_words": ""}
        hit = score_tags(track_tags.get(t.id, {}), lex)
        level = "track"
        if hit is None:
            hit = score_tags(artist_tags.get(t.artist, {}), lex)
            level = "artist"
        if hit is not None:
            row.update(source="last.fm", level=level, valence=hit[0], arousal=hit[1],
                       mood_words=hit[3])
        elif (g := llm_guess(llm, t.artist, t.name)) is not None:
            row.update(source="LLM generated", level=g[2], valence=g[0], arousal=g[1])
        rows.append(row)

    m = tracks.merge(pd.DataFrame(rows), on="id")
    m["quadrant"] = [quadrant(v, a) if pd.notna(v) else None for v, a in zip(m.valence, m.arousal)]
    return m


def gaps():
    """Tracks Last.fm can't place, for the LLM to fill."""
    m = build_moods()
    return m[m.source.isna()]


if __name__ == "__main__":
    m = build_moods()
    print(m.groupby(["source", "level"], dropna=False).size())
