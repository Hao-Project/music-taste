# music-taste

Looking at my own Apple Music library to understand what I listen to, and later to get new recommendations.

Part 1 (this stage): describe the taste. Playlists, audio features and recommendations come later.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install pandas plotly requests
```

Put the Apple Music export at `data/Library.xml` (Music app > File > Library > Export Library). The `data/`, `output/` and `cache/` folders are gitignored because they hold my listening history.

## Run

```
cd src
../.venv/bin/python tags.py musicbrainz     # fetch artist tags, no key needed, ~15 minutes
../.venv/bin/python report.py               # writes output/report.html and CSVs
```

For better tags, get a free Last.fm key (https://www.last.fm/api/account/create), then:

```
export LASTFM_API_KEY=...
../.venv/bin/python tags.py lastfm          # a few minutes
../.venv/bin/python report.py lastfm
```

Tags are cached, so re-running only fetches new artists. The key can also live in a `.env` file (`LASTFM_API_KEY=...`), which is gitignored.

### Mood map

```
../.venv/bin/python track_tags.py           # Last.fm tags per track, ~30-40 minutes
../.venv/bin/python mood_report.py          # writes output/mood.html and moods.csv
```

Tracks and artists are placed on a valence (sad to happy) by arousal (calm to energetic) map, with two separate ratings that are never mixed:

- **Track mood:** an LLM rating of each song it knows well. Songs it doesn't know are left missing, not filled in from the artist.
- **Artist mood:** an LLM rating of each artist's general sound, one dot per artist.

Both come from `cache/mood_llm.json`, in the form `{"tracks": {artist: {title: [v, a]}}, "artists": {artist: [v, a]}}`. The maps use these LLM ratings only, so the scale is consistent.

Last.fm mood tags are used only for a robustness check, where the report compares them with the LLM track ratings on songs that have both. Tags such as "melancholic" or "upbeat" are scored with the word norms from Warriner, Kuperman & Brysbaert (2013), "Norms of valence, arousal, and dominance for 13,915 English lemmas", *Behavior Research Methods* 45. A few words whose musical sense differs from everyday English use hand ratings instead.

The Warriner ratings are licensed CC BY-NC-ND, so they are downloaded into `cache/` at runtime and not included in this repo.

## What is in `src/`

- `load.py` parses the XML into a track table and adds a taste score (0-100): plays per year in the library, reduced by skips, boosted by Loved.
- `tags.py` fetches crowd-sourced genre tags per artist (Last.fm or MusicBrainz).
- `report.py` builds the HTML report plus `tag_profile.csv`, `artists.csv`, `loved.csv` and `tracks.csv` in `output/`.
- `track_tags.py` fetches Last.fm tags for each track, which is where mood words live.
- `mood.py` turns mood tags (or LLM guesses) into valence and arousal for each track.
- `mood_report.py` builds the mood map report.
