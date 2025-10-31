#!/usr/bin/env python3
"""Export Topogram CSV(s) from a folder of per-file CSVs.

This script can either train a Word2Vec model from the given CSV folder or load
an existing saved gensim model and then export:
- a combined Topogram CSV (nodes + edges) for the whole folder
- per-file Topogram CSVs (one file per input CSV) under <out_dir>/per_song_graphs/

Each input CSV must have headers at least: artist,title,lyrics[,year,...]

Usage examples:

# Train then export
python bible_word2vec/scripts/export_topogram.py --folder data/Bible_CSV --out data/Bible_CSV/bible_topogram.csv

# Load an existing model and export
python bible_word2vec/scripts/export_topogram.py --folder data/Bible_CSV --model models/bible.model --out data/bible_topogram.csv

"""
import argparse
import csv
import os
import re
import unicodedata
from collections import defaultdict

from gensim.models import Word2Vec, KeyedVectors
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
            'path': path,
        }


def _normalize_title(t: str) -> str:
    return re.sub(r"[^\w]+", ' ', (t or '').lower()).strip()


def tokenize(text: str):
    toks = re.findall(r"[\wÀ-ÖØ-öø-ÿ']+", text.lower(), flags=re.UNICODE)
    return toks


def _slugify(s: str) -> str:
    if not s:
        return ''
    s = (s or '').strip()
    s = unicodedata.normalize('NFKD', s)
    s = re.sub(r"[^\w\- ]+", '', s)
    s = re.sub(r"[\s]+", '_', s)
    return s[:120]


def build_corpus_and_token_years(files, min_token_length, stopwords_set):
    songs = []
    for f in files:
        s = _read_song_file(f)
        if s and s.get('lyrics'):
            songs.append(s)

    # dedupe by title (same logic used in train script)
    groups = []
    for s in songs:
        norm = _normalize_title(s['title'])
        placed = False
        for g in groups:
            g0 = g[0]
            score = fuzz.token_sort_ratio(norm, _normalize_title(g0['title']))
            if score >= 88.0:
                g.append(s)
                placed = True
                break
        if not placed:
            groups.append([s])

    # pick representative per group and collect years from file metadata only
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
        rep['years'] = sorted(list(years))
        rep_songs.append(rep)
        manifest_rows.append({
            'group_id': gid,
            'title': rep.get('title',''),
            'artist': rep.get('artist',''),
            'members': ';'.join(member_paths),
            'representative': rep.get('path',''),
            'years_in_files': ';'.join(str(y) for y in sorted(years_in_files)) if years_in_files else '',
            'chosen_years': ';'.join(str(y) for y in rep.get('years', []))
        })

    docs = []
    token_years = defaultdict(set)
    for s in rep_songs:
        toks = tokenize(s['lyrics'])
        toks = [t for t in toks if len(t) >= min_token_length and re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]', t) and t not in stopwords_set]
        docs.append(toks)
        for t in set(toks):
            for y in s.get('years', []):
                token_years[t].add(y)

    return docs, token_years, songs, rep_songs, manifest_rows


def export_topogram(out_path, model, vocab, token_years, docs, topn=10, light_edges=True):
    header = [
        'id','name','label','description','color','fillColor','weight','rawWeight','lat','lng',
        'start','end','time','date','source','target','edgeLabel','edgeColor','edgeWeight',
        'relationship','enlightement','emoji','extra'
    ]

    id_map = {token: idx+1 for idx, token in enumerate(vocab)}

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)

        # nodes
        for token, nid in id_map.items():
            years = sorted(token_years.get(token, []))
            start = years[0] if years else ''
            end = years[-1] if years else ''
            try:
                count = model.wv.get_vecattr(token, 'count')
            except Exception:
                count = sum(1 for doc in docs if token in doc)
            row = [nid, token, token, '', '', '', count, count, '', '', start, end, '', '', '', '', '', '', '', '', '', '', '']
            w.writerow(row)

        # edges
        for token in vocab:
            try:
                sims = model.wv.most_similar(token, topn=topn)
            except Exception:
                continue
            src = id_map[token]
            for tgt, score in sims:
                if tgt not in id_map:
                    continue
                tgt_id = id_map[tgt]
                row = [''] * len(header)
                row[14] = str(src)
                row[15] = str(tgt_id)
                row[16] = ''
                row[17] = ''
                row[18] = float(score)
                row[19] = '' if light_edges else 'similarity'
                if light_edges:
                    if len(row) > 20:
                        row[20] = ''
                    if len(row) > 21:
                        row[21] = ''
                w.writerow(row)

    print(f'Wrote combined topogram CSV to {out_path}')


def export_per_file_graphs(per_dir, model, docs, songs, min_token_length, stopwords_set, token_years, light_edges=True, topn=10):
    os.makedirs(per_dir, exist_ok=True)

    # build a global index->id mapping when writing per-file graphs to start IDs from 1 per file
    def _slugify_local(s: str) -> str:
        return _slugify(s)

    header = [
        'id','name','label','description','color','fillColor','weight','rawWeight','lat','lng',
        'start','end','time','date','source','target','edgeLabel','edgeColor','edgeWeight',
        'relationship','enlightement','emoji','extra'
    ]

    for s in songs:
        title = s.get('title') or 'untitled'
        artist = s.get('artist') or 'Unknown'
        year = s.get('year') or ''
        toks = tokenize(s.get('lyrics',''))
        toks = [t for t in toks if len(t) >= min_token_length and re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]', t) and t not in stopwords_set]
        song_nodes = [t for t in toks if t in model.wv.key_to_index]
        if not song_nodes:
            continue
        song_id_map = {t: idx+1 for idx, t in enumerate(sorted(set(song_nodes)))}
        artist_slug = _slugify_local(artist)
        title_slug = _slugify_local(title)
        year_part = year if year else 'unknown'
        fname = f"{artist_slug}_{year_part}_{title_slug}.csv"
        fpath = os.path.join(per_dir, fname)
        with open(fpath, 'w', newline='', encoding='utf-8') as sf:
            w = csv.writer(sf)
            w.writerow(header)
            # nodes
            for token, nid in song_id_map.items():
                years = sorted(token_years.get(token, []))
                start = years[0] if years else year
                end = years[-1] if years else year
                try:
                    count = model.wv.get_vecattr(token, 'count')
                except Exception:
                    count = sum(1 for doc in docs if token in doc)
                row = [nid, token, token, '', '', '', count, count, '', '', start, end, '', '', '', '', '', '', '', '', '', '', '']
                w.writerow(row)

            for token in song_id_map:
                try:
                    sims = model.wv.most_similar(token, topn=topn)
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
                    row[19] = '' if light_edges else 'similarity'
                    if light_edges:
                        if len(row) > 20:
                            row[20] = ''
                        if len(row) > 21:
                            row[21] = ''
                    w.writerow(row)
        print(f'Wrote per-song graph {fpath}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True, help='Folder of per-file CSVs (artist,title,lyrics[,year])')
    p.add_argument('--out', required=True, help='Output Topogram CSV path (single-file nodes+edges)')
    p.add_argument('--model', default=None, help='Optional path to a saved gensim model to load (if omitted script trains one)')
    p.add_argument('--min-token-length', type=int, default=3)
    p.add_argument('--extra-stopwords', default='')
    p.add_argument('--min-count', type=int, default=1)
    p.add_argument('--topn', type=int, default=10, help='Top-N similar tokens per node for edges')
    p.add_argument('--light-edges', action='store_true', default=True)
    p.add_argument('--no-light-edges', dest='light_edges', action='store_false')
    p.add_argument('--manifest', default=None, help='Optional manifest CSV path')
    args = p.parse_args()

    files = [os.path.join(args.folder, f) for f in os.listdir(args.folder) if f.lower().endswith('.csv')]
    if not files:
        raise SystemExit('No CSV files found in folder: ' + args.folder)

    built_in_stopwords = {
        'le','la','les','de','des','du','un','une','et','à','a','au','aux','ce','ces','dans','pas','que','qui','se','ne','en','du','pour','par','sur','mon','ma','mes','ton','ta','tes',
        'the','a','an','in','on','of','to','for','and','is','are','it','you','i','we','they','he','she','me','my','your','our',
    }
    extra_sw = set([s.strip().lower() for s in args.extra_stopwords.split(',') if s.strip()])
    stopwords_set = built_in_stopwords.union(extra_sw)

    docs, token_years, songs, rep_songs, manifest_rows = build_corpus_and_token_years(files, args.min_token_length, stopwords_set)

    model = None
    if args.model:
        print('Loading model', args.model)
        model = Word2Vec.load(args.model) if args.model.endswith('.model') or args.model.endswith('.bin') else KeyedVectors.load(args.model)
    else:
        print('Training Word2Vec model on corpus...')
        model = Word2Vec(sentences=docs, vector_size=100, window=5, min_count=max(1, args.min_count), workers=2, epochs=10)
        # save the model next to out file
        try:
            model_dir = os.path.dirname(args.out) or '.'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'model.model')
            model.save(model_path)
            print('Saved model to', model_path)
        except Exception:
            pass

    raw_vocab = list(model.wv.index_to_key)
    vocab = [t for t in raw_vocab if len(t) >= args.min_token_length and re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]', t) and t not in stopwords_set]

    # export combined
    export_topogram(args.out, model, vocab, token_years, docs, topn=args.topn, light_edges=args.light_edges)

    # optional manifest
    if args.manifest:
        try:
            with open(args.manifest, 'w', newline='', encoding='utf-8') as mf:
                fieldnames = ['group_id', 'title', 'artist', 'members', 'representative', 'years_in_files', 'chosen_years']
                mw = csv.DictWriter(mf, fieldnames=fieldnames)
                mw.writeheader()
                for row in manifest_rows:
                    mw.writerow(row)
            print('Wrote manifest to', args.manifest)
        except Exception as e:
            print('Failed to write manifest', e)

    # per-file graphs
    per_dir = os.path.join(os.path.dirname(args.out) or '.', 'per_song_graphs')
    export_per_file_graphs(per_dir, model, docs, songs, args.min_token_length, stopwords_set, token_years, light_edges=args.light_edges, topn=args.topn)


if __name__ == '__main__':
    main()
