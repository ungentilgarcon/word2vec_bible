#!/usr/bin/env python3
"""Fetch lyrics for artist/title pairs.

Input: CSV with columns 'artist','title' and optional 'url'.
Output: CSV with columns artist,title,lyrics

Strategy:
- Try lyrics.ovh API first: https://api.lyrics.ovh/v1/{artist}/{title}
- If not found and `url` column present, attempt to scrape the page (best-effort)

This script is intentionally conservative (API first). If you want extensive scraping
for many songs, check the target site's robots.txt and Terms of Service.
"""
import argparse
import csv
import os
import re
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import random
import time
import urllib.robotparser
from urllib.parse import urlparse
import difflib

# try to import rapidfuzz for fuzzy matching; fallback to difflib if unavailable
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


API_URL = "https://api.lyrics.ovh/v1/{artist}/{title}"
HEADERS = {"User-Agent": "word2vec-lyrics-bot/0.1 (+https://example.local)"}


def fetch_via_api(artist: str, title: str):
    url = API_URL.format(artist=quote(artist, safe=""), title=quote(title, safe=""))
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            j = r.json()
            if "lyrics" in j and j["lyrics"]:
                return j["lyrics"].strip()
    except Exception:
        return None
    return None


def scrape_url(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        # Decode using detected or declared encoding to avoid mojibake
        enc = r.apparent_encoding or r.encoding or 'utf-8'
        try:
            text_html = r.content.decode(enc, errors='replace')
        except Exception:
            text_html = r.text
        # normalize unicode to NFC
        try:
            import unicodedata
            text_html = unicodedata.normalize('NFC', text_html)
        except Exception:
            pass
        soup = BeautifulSoup(text_html, "lxml")
        # Best-effort heuristics for lyrics content
        selectors = [
            "div.lyrics",
            "div#lyrics",
            "div[class*='lyric']",
            "div[class*='lyrics']",
            "article.lyrics",
            "div[itemprop='text']",
            "div#content",
        ]
        text = None
        for sel in selectors:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                text = el.get_text("\n", strip=True)
                break
        if not text:
            # fallback: select many <p> and join
            ps = soup.find_all("p")
            if ps:
                texts = [p.get_text(strip=True) for p in ps if p.get_text(strip=True)]
                text = "\n".join(texts)
        if text:
            # Clean some common boilerplate
            text = re.sub(r"\[.*?\]", "", text)
            try:
                import unicodedata
                text = unicodedata.normalize('NFC', text)
            except Exception:
                pass
            return text.strip()
    except Exception:
        return None
    return None


def scrape_genius_song(url: str):
    """Scrape lyrics text from a Genius song page.

    Modern Genius pages place lyrics inside multiple <div data-lyrics-container="true"> blocks
    or React components with classes like Lyrics__Container. We try several selectors and
    join the contained text while preserving line breaks.
    Returns None when no sensible lyrics are found (or only placeholder/help text).
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        enc = r.apparent_encoding or r.encoding or 'utf-8'
        try:
            html = r.content.decode(enc, errors='replace')
        except Exception:
            html = r.text
        try:
            import unicodedata
            html = unicodedata.normalize('NFC', html)
        except Exception:
            pass
        soup = BeautifulSoup(html, 'lxml')

        # candidate selectors ordered by preference
        selectors = [
            "div[data-lyrics-container='true']",
            "div[class^='Lyrics__Container']",
            "div.lyrics",
            "div[class*='lyrics']",
        ]

        parts = []
        for sel in selectors:
            els = soup.select(sel)
            if not els:
                continue
            for el in els:
                # Replace <br> with newline to preserve line breaks
                for br in el.find_all('br'):
                    br.replace_with('\n')
                text = el.get_text('\n', strip=True)
                if text:
                    parts.append(text)
            if parts:
                break

        if not parts:
            # fallback: try collecting <p> elements under main content
            ps = soup.find_all('p')
            texts = [p.get_text('\n', strip=True) for p in ps if p.get_text(strip=True)]
            if texts:
                parts = texts

        if not parts:
            return None

        joined = '\n'.join(parts).strip()

        # Detect obvious Genius placeholder/help text and reject it
        placeholder_markers = [
            'how to format lyrics',
            'transcription guide',
            'visit our transcribers forum',
        ]
        low = joined.lower()
        if any(m in low for m in placeholder_markers):
            return None

        # Basic cleanup
        joined = re.sub(r"\[.*?\]", '', joined)
        try:
            import unicodedata
            joined = unicodedata.normalize('NFC', joined)
        except Exception:
            pass
        return joined
    except Exception:
        return None


def fetch_via_genius(artist: str, title: str, token: str = None, max_retries: int = 3, backoff_factor: float = 1.0, fuzz_threshold: float = 60.0):
    """Search Genius API for artist+title and scrape the song URL for lyrics.

    Returns lyrics text or None.
    Note: Genius API does not return full lyrics; we use the returned song URL and
    scrape the page (best-effort). Provide a Genius API client access token.

    Parameters:
    - max_retries: number of retries for transient errors / rate-limits
    - backoff_factor: base seconds for exponential backoff
    - fuzz_threshold: minimum combined fuzzy match score (0-100) to accept a hit
    """
    if not token:
        return None

    headers = {'Authorization': f'Bearer {token}'}
    q = f"{artist} {title}" if artist and title else (artist or title)

    attempt = 0
    while attempt <= max_retries:
        try:
            r = requests.get('https://api.genius.com/search', params={'q': q}, headers=headers, timeout=10)
        except Exception:
            # network issue; backoff and retry
            attempt += 1
            time.sleep(backoff_factor * (2 ** (attempt - 1)))
            continue

        # Handle rate limiting explicitly
        if r.status_code == 429:
            retry_after = r.headers.get('Retry-After')
            try:
                wait = float(retry_after) if retry_after else backoff_factor * (2 ** attempt)
            except Exception:
                wait = backoff_factor * (2 ** attempt)
            time.sleep(wait)
            attempt += 1
            continue

        if r.status_code != 200:
            # Non-recoverable error
            return None

        try:
            data = r.json()
        except Exception:
            return None

        hits = data.get('response', {}).get('hits', [])
        if not hits:
            return None

        # Score hits using fuzzy matching on title and artist. Use rapidfuzz if available.
        best_score = -1.0
        best_result = None
        target_title = (title or '').lower()
        target_artist = (artist or '').lower()

        for h in hits:
            res = h.get('result')
            if not res:
                continue
            res_title = (res.get('title') or '').lower()
            res_artist = (res.get('primary_artist', {}).get('name') or '').lower()

            if fuzz:
                title_score = fuzz.token_sort_ratio(target_title, res_title)
                artist_score = fuzz.token_sort_ratio(target_artist, res_artist)
            else:
                title_score = difflib.SequenceMatcher(None, target_title, res_title).ratio() * 100
                artist_score = difflib.SequenceMatcher(None, target_artist, res_artist).ratio() * 100

            score = (0.65 * title_score) + (0.35 * artist_score)
            if score > best_score:
                best_score = score
                best_result = res

        if best_score < fuzz_threshold:
            # if no sufficiently good match, fall back to first hit
            best_result = hits[0].get('result')

        if not best_result:
            return None

        song_url = best_result.get('url')
        if not song_url:
            return None

        # Prefer a Genius-specific scraper to avoid placeholder/help pages
        try:
            lyrics = scrape_genius_song(song_url)
        except Exception:
            lyrics = None
        if not lyrics:
            lyrics = scrape_url(song_url)
        return lyrics

    return None


def _get_robots_crawl_delay(url: str, user_agent: str = HEADERS.get('User-Agent')):
    """Return crawl-delay from robots.txt for the given URL if present, else None.

    This is used to be polite; it is NOT an evasion technique. If robots.txt disallows
    crawling entirely the caller should respect that (we do not automatically bypass it).
    """
    try:
        p = urlparse(url)
        robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        delay = rp.crawl_delay(user_agent)
        return delay
    except Exception:
        return None


def _polite_sleep(min_delay: float = 0.5, max_delay: float = None, robots_delay: float = None):
    """Sleep a polite amount of time between requests.

    - If robots_delay is provided, use it.
    - Otherwise use a random value between min_delay and max_delay (or min_delay if max not set).
    """
    if robots_delay:
        to_sleep = float(robots_delay)
    else:
        if max_delay is None or max_delay <= min_delay:
            to_sleep = float(min_delay)
        else:
            to_sleep = random.uniform(min_delay, max_delay)
    # small jitter to avoid bursts when running many requests in a loop
    jitter = min(0.1, to_sleep * 0.1)
    to_sleep = max(0.0, to_sleep + random.uniform(-jitter, jitter))
    time.sleep(to_sleep)


def _slug_for_az(s: str) -> str:
    # create a simple slug: lowercase, keep a-z0-9 only
    if not s:
        return ''
    s = s.lower()
    # replace accented chars by ascii equivalents
    s = re.sub(r"[àáâãäå]", 'a', s)
    s = re.sub(r"[èéêë]", 'e', s)
    s = re.sub(r"[ìíîï]", 'i', s)
    s = re.sub(r"[òóôõö]", 'o', s)
    s = re.sub(r"[ùúûü]", 'u', s)
    s = re.sub(r"[ç]", 'c', s)
    s = re.sub(r"[^a-z0-9]", '', s)
    return s


def candidate_azlyrics_urls(artist: str, title: str):
    """Yield candidate azlyrics URLs using common slug patterns.

    Patterns tried:
    - join artist tokens as-is and reversed (handles 'Gainsbourg Serge' vs 'SergeGainsbourg')
    - title slug
    """
    base = 'https://www.azlyrics.com/lyrics/{artist}/{title}.html'
    a = (artist or '').strip()
    t = (title or '').strip()
    if not a or not t:
        return []
    # tokens
    parts = re.split(r"[\s\-_,]+", a)
    variants = []
    joined = ''.join(parts)
    variants.append(joined)
    # reversed order
    if len(parts) > 1:
        variants.append(''.join(reversed(parts)))
    # try removing leading 'the'
    if parts and parts[0].lower() == 'the':
        variants.append(''.join(parts[1:]))

    title_slug = _slug_for_az(t)
    urls = []
    for av in variants:
        artist_slug = _slug_for_az(av)
        if not artist_slug or not title_slug:
            continue
        urls.append(base.format(artist=artist_slug, title=title_slug))
    # unique preserve order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def scrape_azlyrics(artist: str, title: str):
    """Try to fetch lyrics from azlyrics by trying candidate URLs and selecting the best text block."""
    for url in candidate_azlyrics_urls(artist, title):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                continue
            enc = r.apparent_encoding or r.encoding or 'utf-8'
            try:
                html = r.content.decode(enc, errors='replace')
            except Exception:
                html = r.text
            try:
                import unicodedata
                html = unicodedata.normalize('NFC', html)
            except Exception:
                pass
            soup = BeautifulSoup(html, 'lxml')
            # Heuristic: pick the largest <div> with no attributes (azlyrics places lyrics in a plain div)
            divs = [d for d in soup.find_all('div') if not d.attrs]
            if not divs:
                continue
            # pick the div with largest text length
            best = max(divs, key=lambda d: len(d.get_text(strip=True) or ''))
            text = best.get_text('\n', strip=True)
            if text and len(text) > 20:
                # clean common boilerplate
                text = re.sub(r"\[.*?\]", '', text)
                return text.strip()
        except Exception:
            continue
    return None


def normalize_row(row):
    return {k.strip(): v.strip() for k, v in row.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV input with artist,title[,url]")
    p.add_argument("--output", required=True, help="CSV output with artist,title,lyrics")
    p.add_argument("--delay", type=float, default=0.5, help="Minimum delay between requests (s)")
    p.add_argument("--max-delay", type=float, default=None, help="Maximum delay between requests (s). If set, sleeps uniformly between --delay and --max-delay.")
    p.add_argument("--respect-robots", action='store_true', help="Respect robots.txt crawl-delay when available")
    p.add_argument("--use-genius", action='store_true', help="Use Genius API as an additional fallback (requires token)")
    p.add_argument("--genius-token", default=None, help="Genius API token (or set GENIUS_TOKEN env var)")
    p.add_argument("--genius-max-retries", type=int, default=3, help="Max retries for Genius API on rate-limit or transient errors")
    p.add_argument("--genius-backoff-factor", type=float, default=1.0, help="Base backoff seconds for Genius retries (exponential)")
    p.add_argument("--genius-fuzz-threshold", type=float, default=60.0, help="Minimum combined fuzzy score (0-100) to accept a Genius hit")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV must have a header with at least 'artist' and 'title' columns")
        for r in reader:
            rows.append(normalize_row(r))

    out_rows = []
    for r in tqdm(rows, desc="fetching"):
        artist = r.get("artist") or r.get("Artist")
        title = r.get("title") or r.get("Title")
        url = r.get("url") or r.get("URL")
        if not artist or not title:
            continue
        # polite delay: consult robots.txt optionally
        robots_delay = None
        if args.respect_robots:
            # check the source URL preferencing the site we'll hit: azlyrics if used, otherwise lyrics.ovh
            # here we inspect azlyrics site (common fallback) as a courtesy
            robots_delay = _get_robots_crawl_delay('https://www.azlyrics.com')

        lyrics = fetch_via_api(artist, title)
        # if API returned nothing, try URL column first, then azlyrics fallback
        if not lyrics:
            if url:
                _polite_sleep(args.delay, args.max_delay, robots_delay)
                lyrics = scrape_url(url)
            if not lyrics:
                _polite_sleep(args.delay, args.max_delay, robots_delay)
                # try azlyrics domain heuristics
                lyrics = scrape_azlyrics(artist, title)
        # Attempt Genius as a final fallback if enabled
        if not lyrics and args.use_genius:
            token = args.genius_token or os.environ.get('GENIUS_TOKEN')
            if token:
                _polite_sleep(args.delay, args.max_delay, robots_delay)
                try:
                    g_lyrics = fetch_via_genius(artist, title, token, max_retries=args.genius_max_retries, backoff_factor=args.genius_backoff_factor, fuzz_threshold=args.genius_fuzz_threshold)
                    if g_lyrics:
                        lyrics = g_lyrics
                except Exception:
                    # ignore and continue
                    pass
        # always sleep a little after each loop iteration to be polite
        _polite_sleep(args.delay, args.max_delay, robots_delay)
        out_rows.append({"artist": artist, "title": title, "lyrics": lyrics or ""})
        

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["artist", "title", "lyrics"])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)


if __name__ == "__main__":
    main()
