#!/usr/bin/env python3
"""Fetch all songs for an author from AZLyrics author page.

Usage:
  python scripts/fetch_author_azlyrics.py --artist "Serge Gainsbourg" --out data/lyrics_gainsbourg_all.csv

Strategy:
- Build author page URL: https://www.azlyrics.com/{first_letter}/{author_slug}.html
  where author_slug is artist name lowercased with non-alnum removed (e.g. 'sergegainsbourg')
- Parse the author page to find album headers with years and the following song links
- For each song link, fetch lyrics using existing `scrape_url` function from fetch_lyrics.py (best-effort)

Output CSV columns: artist,title,lyrics,year,album

Note: This is best-effort scraping for AZLyrics; respect robots.txt and TOS when crawling large catalogs.
"""
import argparse
import csv
import os
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import logging

# reuse helper from fetch_lyrics if available in same folder
from fetch_lyrics import HEADERS, scrape_url, scrape_genius_song, _get_robots_crawl_delay, _polite_sleep

# Genius helper: optional fallback when AZLyrics is blocked
GENIUS_API_BASE = 'https://api.genius.com'

BASE = 'https://www.azlyrics.com'


def _slugify(s: str) -> str:
    """Create a filesystem-safe slug from a string."""
    if not s:
        return 'unknown'
    s = s.strip()
    # replace spaces with underscore
    s = re.sub(r"\s+", '_', s)
    # remove characters that are not letters, numbers, underscore, hyphen or dot
    s = re.sub(r"[^\w\-\.]+", '', s, flags=re.UNICODE)
    # truncate to reasonable length
    return s[:200]


def _author_slug(artist: str) -> str:
    s = (artist or '').lower()
    s = re.sub(r"[^a-z0-9]", '', s)
    return s


def author_page_url(artist: str) -> str:
    slug = _author_slug(artist)
    if not slug:
        raise ValueError('empty artist')
    first = slug[0]
    return f"{BASE}/{first}/{slug}.html"


def parse_author_page(html: str):
    """Parse author page HTML and return list of (album, year, [(song_title, song_href), ...]).

    Prefer AZLyrics-specific structure: album headers are in divs with class containing
    'album' and songs listed in divs with class 'listalbum-item'. Walk the document in
    order to preserve album -> song grouping. Fall back to the older heuristic if this
    targeted strategy yields nothing.
    """
    soup = BeautifulSoup(html, 'lxml')
    results = []

    # AZLyrics often structures author pages with alternating <div class="album"> and
    # <div class="listalbum-item"><a href="...">Song Title</a></div>
    album_found = False
    current_album = None
    current_year = None
    current_songs = []

    for div in soup.find_all('div'):
        classes = ' '.join(div.get('class') or [])
        text = div.get_text(' ', strip=True)
        # album header
        if re.search(r'album', classes, re.I) or text.lower().startswith('album'):
            # if we were collecting songs for a previous album, save them
            if current_songs:
                results.append((current_album, current_year, current_songs))
                current_songs = []
            album_found = True
            # try to extract a year in the album header
            m = re.search(r"\(?\s*(\d{4})\s*\)?", text)
            current_year = int(m.group(1)) if m else None
            # remove any parenthesized year to get a cleaner album name
            current_album = re.sub(r"\(?\s*\d{4}\s*\)?", '', text).strip()
            continue

        # song item
        if 'listalbum-item' in classes:
            a = div.find('a', href=True)
            if a and a.get_text(strip=True):
                href = a['href']
                title = a.get_text(' ', strip=True)
                current_songs.append((title, href))

    # append the last collected block
    if current_songs:
        results.append((current_album, current_year, current_songs))

    # fallback to previous generic heuristic if nothing found
    if not results:
        # Strategy: find all elements that look like album headers (contain a year in parentheses)
        headers = []
        for tag in soup.find_all(['div', 'b', 'strong']):
            t = tag.get_text(strip=True) if tag else ''
            if re.search(r"\(\s*\d{4}\s*\)", t):
                headers.append(tag)

        if not headers:
            candidates = []
            for tag in soup.find_all():
                try:
                    txt = tag.get_text(' ', strip=True).lower()
                except Exception:
                    txt = ''
                if 'album:' in txt or txt.startswith('album'):
                    candidates.append(tag)
            if candidates:
                headers = candidates
            else:
                headers = soup.find_all(class_=re.compile(r'album', re.I))

        seen_headers = set()
        for h in headers:
            if h in seen_headers:
                continue
            seen_headers.add(h)
            hdr_text = h.get_text(' ', strip=True)
            m = re.search(r"\(\s*(\d{4})\s*\)", hdr_text)
            year = int(m.group(1)) if m else None
            album = re.sub(r"\(\s*\d{4}\s*\)", '', hdr_text).strip()
            songs = []
            node = h.next_sibling
            while node:
                txt = ''
                try:
                    txt = node.get_text(strip=True) if hasattr(node, 'get_text') else str(node)
                except Exception:
                    txt = str(node)
                if re.search(r"\(\s*\d{4}\s*\)", txt):
                    break
                if hasattr(node, 'find_all'):
                    for a in node.find_all('a', href=True):
                        href = a['href']
                        title = a.get_text(' ', strip=True)
                        if 'lyrics' in href or href.startswith('/lyrics') or 'azlyrics' in href:
                            songs.append((title, href))
                if getattr(node, 'name', None) == 'a' and node.get('href'):
                    href = node['href']
                    title = node.get_text(' ', strip=True)
                    if 'lyrics' in href or href.startswith('/lyrics') or 'azlyrics' in href:
                        songs.append((title, href))
                node = node.next_sibling
            if songs:
                results.append((album, year, songs))

        if not results:
            songs = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if 'lyrics' in href and a.get_text(strip=True):
                    songs.append((a.get_text(strip=True), href))
            if songs:
                results.append((None, None, songs))

    return results


def _absolute_href(href: str, page_url: str):
    if href.startswith('http'):
        return href
    # relative like ../lyrics/artist/song.html -> join with base
    return urljoin(page_url, href)


def fetch_author(artist: str, out_csv: str, delay: float = 0.3, max_delay: float = None, respect_robots: bool = False):
    logger = logging.getLogger(__name__)
    url = author_page_url(artist)
    logger.info('Fetching AZLyrics author page for %s -> %s', artist, url)
    # Try once with the default bot UA; if blocked (403) retry with a common browser UA.
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code == 403:
        # retry with a browser-like user agent to work around simple UA blocking
        browser_headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
        }
        r = requests.get(url, headers=browser_headers, timeout=10)
        logger.info('Retry with browser UA returned status %s', r.status_code)
    # If still not 200, return None so caller can decide to use Genius fallback
    if r.status_code != 200:
        logger.warning('Could not fetch AZLyrics author page: %s (status %s)', url, r.status_code)
        return None
    enc = r.apparent_encoding or r.encoding or 'utf-8'
    try:
        html = r.content.decode(enc, errors='replace')
    except Exception:
        html = r.text

    parsed = parse_author_page(html)
    logger.info('Parsed author page: found %d album blocks', len(parsed))
    out_rows = []
    # blacklist of obvious site navigation / noise titles
    noise_titles = set([
        'contact us', 'advertise here', 'cookie policy', 'privacy policy', 'dmca policy',
        'soundtracks', 'links', 'link', 'search', 'cookie', 'contact', 'privacy'
    ])
    robots_delay = None
    if respect_robots:
        robots_delay = _get_robots_crawl_delay(url)

    for album, year, songs in parsed:
        logger.info('Album: %s (%s) -> %d songs', (album or '').strip(), year or '', len(songs))
        for title, href in songs:
            title_norm = (title or '').strip().lower()
            # skip obvious noise titles
            if any(nt == title_norm for nt in noise_titles) or any(nt in title_norm for nt in noise_titles):
                continue
            song_url = _absolute_href(href, url)
            logger.debug('Fetching song: %s -> %s', title, song_url)
            _polite_sleep(delay, max_delay, robots_delay)
            lyrics = scrape_url(song_url)

            # detect navigation-like lyric pages: contain 'LINKS:' or long lists of album/song text
            ly = (lyrics or '').strip()
            ly_low = ly.lower()
            # don't drop legitimately empty lyrics (user may fill later); only drop obvious nav dumps
            if 'links:' in ly_low or 'list of artists' in ly_low or ('http' in ly_low and len(ly) < 100):
                continue
            # extremely long blocks with many "album:" markers are probably nav dumps
            if ly.count('album:') > 2 and len(ly) > 200:
                continue

            out_rows.append({
                'artist': artist,
                'title': title,
                'lyrics': ly,
                'year': year or '' ,
                'album': album or ''
            })
            logger.info('Fetched lyrics for %s (%d chars)', title, len(ly))
    # write CSV
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['artist', 'title', 'lyrics', 'year', 'album']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)
    logger.info('Wrote %d rows to %s', len(out_rows), out_csv)
    return out_csv


def genius_find_artist_id(artist: str, token: str):
    """Return Genius artist id for a given artist name (best-effort)."""
    headers = {'Authorization': f'Bearer {token}'}
    logger = logging.getLogger(__name__)
    try:
        r = requests.get(f"{GENIUS_API_BASE}/search", params={'q': artist}, headers=headers, timeout=10)
        logger.info('Genius search for %s returned %s', artist, r.status_code)
        if r.status_code != 200:
            return None
        data = r.json()
        hits = data.get('response', {}).get('hits', [])
        if not hits:
            return None
        # Prefer a hit where primary_artist name contains the query artist
        for h in hits:
            res = h.get('result')
            if not res:
                continue
            pa = (res.get('primary_artist', {}).get('name') or '').lower()
            if artist.lower() in pa:
                logger.info('Matched artist id %s for %s', res.get('primary_artist', {}).get('id'), artist)
                return res.get('primary_artist', {}).get('id')
        # otherwise return the first primary artist id
        first = hits[0].get('result')
        return first.get('primary_artist', {}).get('id') if first else None
    except Exception as e:
        logger.exception('Genius artist search failed: %s', e)
        return None


def genius_list_artist_songs(artist_id: int, token: str, per_page: int = 50, max_pages: int = 5):
    """Yield (title, url) tuples for songs from Genius artist endpoint."""
    headers = {'Authorization': f'Bearer {token}'}
    songs = []
    logger = logging.getLogger(__name__)
    for page in range(1, max_pages + 1):
        try:
            r = requests.get(f"{GENIUS_API_BASE}/artists/{artist_id}/songs", params={'per_page': per_page, 'page': page}, headers=headers, timeout=10)
            logger.info('Genius artist songs page %d -> %s', page, r.status_code)
            if r.status_code != 200:
                break
            data = r.json()
            items = data.get('response', {}).get('songs', [])
            if not items:
                break
            for s in items:
                title = s.get('title')
                url = s.get('url')
                if title and url:
                    songs.append((title, url))
        except Exception:
            logger.exception('Failed to fetch artist songs page %d', page)
            break
    return songs


def fetch_author_via_genius(artist: str, out_csv: str, token: str, delay: float = 0.3, max_delay: float = None, max_pages: int = 5):
    """Fetch songs for artist by querying Genius artist songs endpoint and scraping each song page."""
    artist_id = genius_find_artist_id(artist, token)
    if not artist_id:
        return None
    song_list = genius_list_artist_songs(artist_id, token, per_page=50, max_pages=max_pages)
    if not song_list:
        return None

    out_rows = []
    robots_delay = None
    # prepare artist folder based on out_csv parent directory
    base_dir = os.path.dirname(out_csv) or '.'
    artist_folder = os.path.join(base_dir, _slugify(artist))
    os.makedirs(artist_folder, exist_ok=True)
    # We don't have a canonical site for robots here; rely on polite sleeps passed in
    logger = logging.getLogger(__name__)
    total_songs = len(song_list)
    idx = 0
    for title, url in song_list:
        idx += 1
        logger.info('Fetching song %d/%d: %s', idx, total_songs, title)
        _polite_sleep(delay, max_delay, robots_delay)
        # Use Genius-specific scraper to avoid placeholder/help content
        lyrics = scrape_genius_song(url) or scrape_url(url)
        ly = (lyrics or '').strip()
        # quick nav heuristics similar to AZLyrics parser
        ly_low = ly.lower()
        if 'links:' in ly_low or 'list of artists' in ly_low or ('http' in ly_low and len(ly) < 100):
            logger.debug('Skipping likely navigation page for %s', title)
            continue
        # write per-song CSV file
        safe_title = _slugify(title) or f'song_{idx}'
        filename = f"{safe_title}.csv"
        path = os.path.join(artist_folder, filename)
        # avoid collisions
        k = 1
        base_path = path
        while os.path.exists(path):
            path = os.path.join(artist_folder, f"{safe_title}_{k}.csv")
            k += 1

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['artist', 'title', 'lyrics', 'year', 'album'])
            writer.writeheader()
            writer.writerow({'artist': artist, 'title': title, 'lyrics': ly, 'year': '', 'album': ''})

        logger.info('Wrote song file %s (%d chars)', path, len(ly))

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['artist', 'title', 'lyrics', 'year', 'album']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)
    return out_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artist', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--delay', type=float, default=0.3)
    p.add_argument('--max-delay', type=float, default=None)
    p.add_argument('--respect-robots', action='store_true')
    p.add_argument('--use-genius', action='store_true', help='Use Genius API to list artist songs when AZLyrics is unreachable')
    p.add_argument('--genius-token', default=None, help='Genius API token (or set GENIUS_TOKEN env var)')
    p.add_argument('--genius-max-pages', type=int, default=5, help='Max pages to request from /artists/:id/songs')
    args = p.parse_args()

    # Configure basic logging if not already configured by the caller
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    # If configured, use Genius-only mode (skip AZLyrics entirely)
    if args.use_genius:
        token = args.genius_token or os.environ.get('GENIUS_TOKEN')
        if not token:
            raise SystemExit('Genius token required when --use-genius is used')
        out = fetch_author_via_genius(args.artist, args.out, token, delay=args.delay, max_delay=args.max_delay, max_pages=args.genius_max_pages)
        if out:
            print(f'Wrote {out}')
            return out
        raise SystemExit('Could not fetch songs via Genius')

    # Default: attempt AZLyrics author fetch
    res = fetch_author(args.artist, args.out, delay=args.delay, max_delay=args.max_delay, respect_robots=bool(args.respect_robots))
    return res


if __name__ == '__main__':
    main()
