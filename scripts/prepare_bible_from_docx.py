#!/usr/bin/env python3
"""Prepare Bible text DOCX files for Word2Vec training.

This small utility converts each `.docx` file in an input folder into a single-row
CSV file compatible with the existing training script `train_from_folder.py`.

Output per-file CSV format (single row):
- artist,title,lyrics

Usage example:

    python bible_word2vec/scripts/prepare_bible_from_docx.py --input-folder /path/to/docx --out-folder data/Bible

Notes:
- Requires `python-docx` (install: `pip install python-docx`).
- Each DOCX file becomes one CSV named <Artist>_<filename>.csv (slugified).
- The script uses Artist="Bible" by default; you can change the artist via `--artist`.
"""
import argparse
import csv
import os
import re
import unicodedata
import json

try:
    from docx import Document
except Exception:
    Document = None
    # we'll handle missing python-docx at runtime for .docx files with a helpful message


def _slugify(s: str) -> str:
    if not s:
        return ''
    s = s.strip()
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r"[^\w\- ]+", '', s)
    s = re.sub(r"[\s]+", '_', s)
    return s[:120]


def extract_text_from_word(path: str) -> str:
    """Extract text from a Word file.

    Supports .docx via python-docx, and .doc via the `antiword` or `catdoc` command if available.
    Returns the extracted text or raises RuntimeError with an explanatory message.
    """
    lower = path.lower()
    if lower.endswith('.docx'):
        if Document is None:
            raise RuntimeError('python-docx is not installed; install with `pip install python-docx` to read .docx files')
        doc = Document(path)
        parts = []
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                parts.append(text)
        return '\n'.join(parts).strip()

    if lower.endswith('.doc'):
        # try antiword first
        import subprocess
        for cmd in (['antiword', '-m', 'UTF-8.txt', path], ['antiword', path], ['catdoc', path]):
            try:
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
                out = p.stdout.decode('utf-8', errors='replace')
                out = out.strip()
                if out:
                    return out
            except FileNotFoundError:
                # command not installed; try next
                continue
            except subprocess.CalledProcessError:
                continue
        # if we reach here, no tool succeeded
        raise RuntimeError('Could not extract text from .doc file: please install `antiword` or `catdoc`, or convert files to .docx')

    raise RuntimeError('Unsupported file extension for: ' + path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-folder', required=True, help='Folder containing .doc or .docx files')
    p.add_argument('--out-folder', required=True, help='Folder where per-file CSVs will be written')
    p.add_argument('--artist', default='Bible', help='Artist field to write into CSV (default: "Bible")')
    p.add_argument('--pattern', default='*.doc*', help='Filename glob pattern to search for')
    p.add_argument('--split-by-headings', action='store_true', help='Split large files into multiple CSVs when headings are detected')
    p.add_argument('--config', default=None, help='Optional JSON config file to tune stopwords, tokenization, or book-name headings')
    args = p.parse_args()
    # load optional config
    config = {}
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as cf:
                config = json.load(cf)
        except Exception as e:
            print('Failed to load config', args.config, e)
            config = {}

    os.makedirs(args.out_folder, exist_ok=True)

    files = [f for f in os.listdir(args.input_folder) if f.lower().endswith('.doc') or f.lower().endswith('.docx')]
    if not files:
        print('No .doc/.docx files found in', args.input_folder)
        return

    for fname in files:
        path = os.path.join(args.input_folder, fname)
        try:
            text = extract_text_from_word(path)
        except Exception as e:
            print('Failed to read', path, e)
            continue
        if not text:
            print('Skipping empty document', path)
            continue

        base_title = os.path.splitext(fname)[0]

        # If splitting by headings is requested, split the text into segments
        if args.split_by_headings:
            # Simple heading heuristics: lines that are ALL CAPS, lines starting with CHAPTER, or lines matching book names from config
            book_names = set([bn.lower() for bn in (config.get('book_names') or [])])
            lines = [l.strip() for l in text.splitlines()]
            segments = []  # list of (heading, content_lines)
            current_heading = None
            current_content = []

            def is_heading(line: str) -> bool:
                if not line:
                    return False
                # CHAPTER lines
                if re.match(r'^(chapter|CHAPTER)\b', line):
                    return True
                # lines in all caps with some letters and length >= 3
                alpha = re.sub(r'[^A-Z]+', '', line)
                if alpha and line.upper() == line and len(alpha) >= 3 and len(line) <= 200:
                    return True
                # book name match (contains or equals)
                low = line.lower()
                for bn in book_names:
                    if bn and (low == bn or bn in low):
                        return True
                return False

            for ln in lines:
                if is_heading(ln):
                    # start a new segment
                    if current_heading or current_content:
                        segments.append((current_heading or base_title, '\n'.join(current_content).strip()))
                    current_heading = ln
                    current_content = []
                else:
                    current_content.append(ln)

            # append last
            if current_heading or current_content:
                segments.append((current_heading or base_title, '\n'.join(current_content).strip()))

            # if only one segment found, fall back to single-file behavior
            if len(segments) <= 1:
                # write as single CSV
                title = base_title
                csv_name = f"{_slugify(args.artist)}_{_slugify(title)}.csv"
                out_path = os.path.join(args.out_folder, csv_name)
                with open(out_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['artist', 'title', 'lyrics'])
                    writer.writeheader()
                    writer.writerow({'artist': args.artist, 'title': title, 'lyrics': text})
                print('Wrote', out_path)
            else:
                # write each segment with heading-derived title
                for idx, (heading, content) in enumerate(segments, start=1):
                    # derive a clean title
                    seg_title = heading if heading else f"{base_title}_part{idx:02d}"
                    seg_title = re.sub(r"\s+", ' ', seg_title).strip()
                    csv_name = f"{_slugify(args.artist)}_{_slugify(seg_title)}.csv"
                    # avoid name collisions
                    out_path = os.path.join(args.out_folder, csv_name)
                    k = 1
                    while os.path.exists(out_path):
                        out_path = os.path.join(args.out_folder, f"{_slugify(args.artist)}_{_slugify(seg_title)}_{k}.csv")
                        k += 1
                    with open(out_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=['artist', 'title', 'lyrics'])
                        writer.writeheader()
                        writer.writerow({'artist': args.artist, 'title': seg_title, 'lyrics': content})
                    print('Wrote', out_path)

            continue

        # default single-file behavior
        title = base_title
        csv_name = f"{_slugify(args.artist)}_{_slugify(title)}.csv"
        out_path = os.path.join(args.out_folder, csv_name)
        # write single-row CSV with columns artist,title,lyrics
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['artist', 'title', 'lyrics'])
            writer.writeheader()
            writer.writerow({'artist': args.artist, 'title': title, 'lyrics': text})
        print('Wrote', out_path)


if __name__ == '__main__':
    main()
