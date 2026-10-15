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

Tags are cached, so re-running only fetches new artists.

## What is in `src/`

- `load.py` parses the XML into a track table and adds a taste score (0-100): plays per year in the library, reduced by skips, boosted by Loved.
- `tags.py` fetches crowd-sourced genre tags per artist (Last.fm or MusicBrainz).
- `report.py` builds the HTML report plus `tag_profile.csv`, `artists.csv`, `loved.csv` and `tracks.csv` in `output/`.
