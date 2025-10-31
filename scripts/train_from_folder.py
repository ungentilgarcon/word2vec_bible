#!/usr/bin/env python3
"""Bible-tailored copy of train_from_folder.py

This is a near-copy of the project's `scripts/train_from_folder.py` with
Bible-friendly defaults (min token length, extra stopwords, and light-edges on by default).
It keeps the same CLI and behavior otherwise so you can run the same commands as the
main project but inside the fork without cross-file references.
"""
import argparse
import csv
import sys

# CSV fields may be very large for long lyric fields; raise the field size limit if possible
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # fall back to a large value
    csv.field_size_limit(10**7)
import os
import re
import time
from collections import defaultdict
import unicodedata

import requests
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from rapidfuzz import fuzz



def _read_song_file(path):
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        rows = list(r)
        if not rows:
            return None
        row = rows[0]
        return {
            'artist': row.get('artist','').strip(),
            'title': row.get('title','').strip(),
            'lyrics': row.get('lyrics','').strip(),
            'year': row.get('year','').strip() if 'year' in row else '',
            'album': row.get('album','').strip() if 'album' in row else ''
        }



def _normalize_title(t: str) -> str:
    return re.sub(r"[^\w]+", ' ', (t or '').lower()).strip()


def genius_search_year(artist: str, title: str, token: str):
    headers = {'Authorization': f'Bearer {token}'}
    try:
        q = f"{artist} {title}" if artist else title
        r = requests.get('https://api.genius.com/search', params={'q': q}, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        hits = data.get('response', {}).get('hits', [])
        if not hits:
            return None
        song = hits[0].get('result')
        sid = song.get('id')
        if not sid:
            return None
        r2 = requests.get(f'https://api.genius.com/songs/{sid}', headers=headers, timeout=10)
        if r2.status_code != 200:
            return None
        info = r2.json().get('response', {}).get('song', {})
        rd = info.get('release_date_for_display') or info.get('release_date')
        if not rd:
            comps = info.get('release_date_components')
            if comps and isinstance(comps, dict):
                y = comps.get('year')
                return int(y) if y else None
            return None
        m = re.search(r"(\d{4})", rd)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None


def tokenize(text: str):
    toks = re.findall(r"[\wÀ-ÖØ-öø-ÿ']+", text.lower(), flags=re.UNICODE)
    return toks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--genius-token', default=None)
    p.add_argument('--dedupe-threshold', type=float, default=88.0, help='Fuzzy score threshold to consider titles duplicates')
    p.add_argument('--min-count', type=int, default=1)
    p.add_argument('--manifest', default=None, help='Optional path to write a manifest CSV for dedupe groups')
    # Bible-friendly defaults: shorter tokens are often meaningful (e.g. 'god', 'man') so keep 3 by default
    p.add_argument('--min-token-length', type=int, default=3, help='Minimum token length to keep as a node')
    # Provide sensible biblical extra stopwords by default (common liturgical words that bloat graphs)
    default_extra = 'the,and,of,unto,unto,shall,ye,thou,thy,thee,unto,unto,unto'
    p.add_argument('--extra-stopwords', default=default_extra, help='Comma-separated extra stopwords to remove')
    # Make light-edges the default; allow disabling with --no-light-edges
    p.add_argument('--light-edges', dest='light_edges', action='store_true', default=True, help='Write minimal edge metadata by default')
    p.add_argument('--no-light-edges', dest='light_edges', action='store_false', help='Disable light edges and include textual edge fields')
    p.add_argument('--no-per-song-graphs', action='store_true', help='If set, do NOT produce per-song graph CSVs (by default per-song graphs are generated)')
    p.add_argument('--embedding', choices=['word2vec','fasttext','glove'], default='word2vec', help='Embedding backend to use: word2vec (default), fasttext (train FastText), or glove (load pre-trained GloVe)')
    p.add_argument('--glove-path', default=None, help='Path to pre-trained GloVe txt file (required when --embedding glove)')
    args = p.parse_args()

    files = [os.path.join(args.folder, f) for f in os.listdir(args.folder) if f.lower().endswith('.csv')]
    songs = []
    for f in files:
        s = _read_song_file(f)
        if s and s.get('lyrics'):
            s['path'] = f
            songs.append(s)

    groups = []
    for s in songs:
        norm = _normalize_title(s['title'])
        placed = False
        for g in groups:
            g0 = g[0]
            score = fuzz.token_sort_ratio(norm, _normalize_title(g0['title']))
            if score >= args.dedupe_threshold:
                g.append(s)
                placed = True
                break
        if not placed:
            groups.append([s])

    print(f'Found {len(songs)} songs, grouped into {len(groups)} deduplicated entries')

    rep_songs = []
    manifest_rows = []
    for gid, g in enumerate(groups, start=1):
        rep = max(g, key=lambda x: len(x.get('lyrics','') or ''))
        years = set()
        years_in_files = set()
        member_paths = []
        for member in g:
            member_paths.append(member.get('path') or '')
            y = member.get('year')
            if y and y.isdigit():
                years.add(int(y))
                years_in_files.add(int(y))

        genius_year = None
        if not years and args.genius_token:
            y = genius_search_year(rep.get('artist',''), rep.get('title',''), args.genius_token)
            if y:
                years.add(y)
                genius_year = y

        rep['years'] = sorted(list(years))
        rep_songs.append(rep)

        manifest_rows.append({
            'group_id': gid,
            'title': rep.get('title',''),
            'artist': rep.get('artist',''),
            'members': ';'.join(member_paths),
            'representative': rep.get('path',''),
            'years_in_files': ';'.join(str(y) for y in sorted(years_in_files)) if years_in_files else '',
            'genius_year': str(genius_year) if genius_year else '',
            'chosen_years': ';'.join(str(y) for y in rep.get('years', []))
        })

    built_in_stopwords = {
        'le','la','les','de','des','du','un','une','et','à','a','au','aux','ce','ces','dans','pas','que','qui','se','ne','en','du','pour','par','sur','mon','ma','mes','ton','ta','tes',
        'the','a','an','in','on','of','to','for','and','is','are','it','you','i','we','they','he','she','me','my','your','our',
    }
    extra_sw = set([s.strip().lower() for s in args.extra_stopwords.split(',') if s.strip()])
    stopwords_set = built_in_stopwords.union(extra_sw)

    docs = []
    token_years = defaultdict(set)
    for s in rep_songs:
        toks = tokenize(s['lyrics'])
        toks = [t for t in toks if len(t) >= args.min_token_length and re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]', t) and t not in stopwords_set]
        docs.append(toks)
        for t in set(toks):
            for y in s.get('years', []):
                token_years[t].add(y)

    if not docs:
        print('No documents to train on')
        return

    size = 100
    window = 5
    min_count = max(1, args.min_count)
    # Choose embedding backend: word2vec (default), fasttext (gensim FastText), or glove (pretrained file)
    emb = (args.embedding or 'word2vec').lower()
    model = None
    if emb == 'fasttext':
        # Train FastText (subword-aware)
        model = FastText(sentences=docs, vector_size=size, window=window, min_count=min_count, workers=2, epochs=10)
    elif emb == 'glove':
        # Expect a pre-trained GloVe text file path to be provided
        if not args.glove_path:
            print('Embedding "glove" selected but --glove-path was not provided; aborting')
            return
        # Convert GloVe to word2vec format in a temp file and load KeyedVectors
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpf:
            tmp_path = tmpf.name
        try:
            glove2word2vec(args.glove_path, tmp_path)
            model = KeyedVectors.load_word2vec_format(tmp_path, binary=False)
        except Exception as e:
            print('Failed to load GloVe vectors from', args.glove_path, e)
            return
    else:
        model = Word2Vec(sentences=docs, vector_size=size, window=window, min_count=min_count, workers=2, epochs=10)

    # Support models that are KeyedVectors (glove) or full models with .wv
    wv = model.wv if hasattr(model, 'wv') else model
    raw_vocab = list(wv.index_to_key)
    vocab = [t for t in raw_vocab if len(t) >= args.min_token_length and re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]', t) and t not in stopwords_set]
    id_map = {token: idx + 1 for idx, token in enumerate(vocab)}

    out_file = args.out if args.out.lower().endswith('.csv') else args.out + '.csv'
    header = [
        'id','name','label','description','color','fillColor','weight','rawWeight','lat','lng',
        'start','end','time','date','source','target','edgeLabel','edgeColor','edgeWeight',
        'relationship','enlightement','emoji','extra'
    ]

    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, token in enumerate(vocab, start=1):
            years = sorted(token_years.get(token, []))
            start = years[0] if years else ''
            end = years[-1] if years else ''
            try:
                # Not all backends expose token counts (e.g., preloaded KeyedVectors). Fall back to corpus counts.
                count = wv.get_vecattr(token, 'count')
            except Exception:
                count = sum(1 for doc in docs if token in doc)
            name = token
            label = token
            description = ''
            color = ''
            fillColor = ''
            weight = count
            rawWeight = count
            lat = ''
            lng = ''
            time = ''
            date = ''
            source = ''
            target = ''
            edgeLabel = ''
            edgeColor = ''
            edgeWeight = ''
            relationship = ''
            enlightement = ''
            emoji = ''
            extra = ''
            row = [i,name,label,description,color,fillColor,weight,rawWeight,lat,lng,start,end,time,date,source,target,edgeLabel,edgeColor,edgeWeight,relationship,enlightement,emoji,extra]
            writer.writerow(row)

        for token in vocab:
            try:
                sims = wv.most_similar(token, topn=10)
            except Exception:
                continue
            src = id_map[token]
            for tgt, score in sims:
                if tgt not in id_map:
                    continue
                tgt_id = id_map[tgt]
                edgeLabel = ''
                edgeColor = ''
                edgeWeight = float(score)
                relationship = '' if args.light_edges else 'similarity'
                row = [''] * len(header)
                row[14] = str(src)
                row[15] = str(tgt_id)
                row[16] = edgeLabel
                row[17] = edgeColor
                row[18] = edgeWeight
                row[19] = relationship
                if args.light_edges:
                    if len(row) > 20:
                        row[20] = ''
                    if len(row) > 21:
                        row[21] = ''
                writer.writerow(row)

    print(f'Wrote combined topogram CSV to {out_file}')

    if args.manifest:
        try:
            mpath = args.manifest
            with open(mpath, 'w', newline='', encoding='utf-8') as mf:
                fieldnames = ['group_id', 'title', 'artist', 'members', 'representative', 'years_in_files', 'genius_year', 'chosen_years']
                mw = csv.DictWriter(mf, fieldnames=fieldnames)
                mw.writeheader()
                for row in manifest_rows:
                    mw.writerow(row)
            print(f'Wrote manifest CSV to {mpath}')
        except Exception as e:
            print(f'Failed to write manifest {args.manifest}: {e}')

    if not args.no_per_song_graphs:
        try:
            per_dir = os.path.join(os.path.dirname(out_file) or '.', 'per_song_graphs')
            os.makedirs(per_dir, exist_ok=True)

            def _slugify(s: str) -> str:
                s = (s or '').strip()
                s = unicodedata.normalize('NFKD', s)
                s = re.sub(r"[^\w\- ]+", '', s)
                s = re.sub(r"[\s]+", '_', s)
                return s[:120]

            for s in songs:
                artist = s.get('artist') or 'Unknown'
                title = s.get('title') or 'untitled'
                year = s.get('year') or ''
                if not year:
                    nt = _normalize_title(title)
                    for r in rep_songs:
                        if _normalize_title(r.get('title','')) == nt and r.get('years'):
                            year = str(r.get('years')[0])
                            break

                toks = tokenize(s.get('lyrics',''))
                toks = [t for t in toks if len(t) >= args.min_token_length and re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]', t) and t not in stopwords_set]
                song_nodes = [t for t in toks if t in (wv.key_to_index if hasattr(wv, 'key_to_index') else {})]
                if not song_nodes:
                    continue

                song_id_map = {t: idx+1 for idx, t in enumerate(sorted(set(song_nodes)))}

                artist_slug = _slugify(artist)
                title_slug = _slugify(title)
                year_part = year if year else 'unknown'
                fname = f"{artist_slug}_{year_part}_{title_slug}.csv"
                fpath = os.path.join(per_dir, fname)

                with open(fpath, 'w', newline='', encoding='utf-8') as sf:
                    w = csv.writer(sf)
                    w.writerow(header)
                    for token, nid in song_id_map.items():
                        years = sorted(token_years.get(token, []))
                        start = years[0] if years else year
                        end = years[-1] if years else year
                        try:
                            count = model.wv.get_vecattr(token, 'count')
                        except Exception:
                            count = sum(1 for doc in docs if token in doc)
                        row = [nid, token, token, '', '', '', count, count, '', '', start, end, '', '', '', '', '', '', '', '', '', '']
                        w.writerow(row)

                    for token in song_id_map:
                        try:
                            sims = wv.most_similar(token, topn=10)
                        except Exception:
                            continue
                        src = song_id_map[token]
                        for tgt, score in sims:
                            if tgt not in song_id_map:
                                continue
                            tgt_id = song_id_map[tgt]
                            row = [''] * len(header)
                            row[14] = str(src)
                            row[15] = str(tgt_id)
                            row[16] = ''
                            row[17] = ''
                            row[18] = float(score)
                            row[19] = '' if args.light_edges else 'similarity'
                            if args.light_edges:
                                if len(row) > 20:
                                    row[20] = ''
                                if len(row) > 21:
                                    row[21] = ''
                            w.writerow(row)

                print(f'Wrote per-song graph {fpath}')
        except Exception as e:
            print(f'Failed to produce per-song graphs: {e}')


if __name__ == '__main__':
    main()
