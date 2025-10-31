bible_word2vec — fork for Bible text Word2Vec exploration

Purpose

This folder is a self-contained fork of the main Word2Vec lyrics project, adapted to prepare and train Word2Vec models from Word documents (`.doc` / `.docx`) and to export Topogram-compatible CSVs (nodes + edges in the single-file format used by `topogram-sample.csv`). The fork produces:

- a combined Topogram CSV for the whole input folder (global graph), and
- per-file or per-chapter Topogram CSVs (one CSV per input file or per detected chapter/heading) under a `per_song_graphs/` folder.

What is included

- `scripts/prepare_bible_from_docx.py` — converts Word files to per-file CSVs compatible with the training script. It supports `.docx` (via `python-docx`) and `.doc` fallback via `antiword`/`catdoc`. It also supports:
	- `--split-by-headings` to split large documents into multiple segments using simple heading heuristics (ALL CAPS, CHAPTER markers, or book-name matches from a config), and
	- `--config <path>` to load a JSON with tuning parameters (book names, extra stopwords, token_min_length).

- `scripts/train_from_folder.py` — a Bible-tailored copy of the training/export script. It trains a Word2Vec model on the generated per-file CSVs and writes a Topogram-compatible CSV with nodes first and edges after (matching the header in `topogram-sample.csv`). The script also writes a manifest CSV and per-file graphs under `per_song_graphs/` by default.

- `run_all.sh` — small wrapper that runs preparation and training in sequence (convenience wrapper).

- `config.sample.json` — example JSON with `extra_stopwords`, `token_min_length` and a `book_names` list to help heading detection.

Quick start (one-shot)

1) Install Python deps inside your venv:

```bash
pip install -r bible_word2vec/requirements.txt
```

2) Make sure you have a `.doc` extraction tool if you have legacy `.doc` files (one of these):

```bash
# Debian/Ubuntu example
sudo apt install antiword
# or
sudo apt install catdoc
```

3) Run the wrapper to convert and train (simple):

```bash
./bible_word2vec/run_all.sh /abs/path/to/doc_files data/Bible_CSV "Bible"
```

This will:
- convert each `.doc`/`.docx` into a per-file CSV under `data/Bible_CSV/` (one CSV per file),
- train Word2Vec on that CSV folder and write a combined Topogram CSV `data/Bible_CSV/bible_topogram.csv`,
- write `data/Bible_CSV/manifest.csv` (dedupe/groups) and per-file/per-chapter Topogram CSVs under `data/Bible_CSV/per_song_graphs/`.

If you want to split large files by headings or pass a JSON config, run prepare + train manually (this gives full control):

```bash
# convert and split using a JSON config
python3 bible_word2vec/scripts/prepare_bible_from_docx.py --input-folder /abs/path/to/doc_files --out-folder data/Bible_CSV --artist "Bible" --split-by-headings --config bible_word2vec/config.sample.json

# then train with the forked train script (generates combined and per-file graphs)
python3 bible_word2vec/scripts/train_from_folder.py --folder data/Bible_CSV --out data/Bible_CSV/bible_topogram.csv --min-token-length 3 --light-edges --manifest data/Bible_CSV/manifest.csv
```

Notes on outputs and Topogram compatibility

- The combined Topogram CSV `bible_topogram.csv` matches the single-file nodes+edges style used in `topogram-sample.csv` (nodes first as rows with id,name,label,... then edge rows with node-columns empty and `source`,`target`,`edgeWeight` filled).
- Per-file/per-chapter CSVs are written to `data/Bible_CSV/per_song_graphs/`. Each file is itself a Topogram-compatible CSV using the same header.
- The training script supports `--light-edges` (default in the fork) which leaves textual edge fields empty and only fills `source`,`target`,`edgeWeight`. Use `--no-light-edges` to include textual `relationship` values.

Topogram header mapping (brief)

- Node rows: columns `id,name,label,description,color,fillColor,weight,rawWeight,lat,lng,start,end,time,date,...` — fill `id`, `name`/`label`, `weight`/`rawWeight`, and optional `start`/`end` year metadata.
- Edge rows: leave node columns empty, fill `source` (node id), `target` (node id), `edgeWeight` (similarity), and optionally `relationship`/`edgeLabel`.

Configuration and tuning

- `bible_word2vec/config.sample.json` is a starting point. Example fields:
	- `extra_stopwords`: array of lower-case tokens to remove
	- `token_min_length`: minimum token length to keep
	- `book_names`: list of book names (helps the `--split-by-headings` heuristic)

- You can also pass `--min-token-length` and `--extra-stopwords` directly to `train_from_folder.py`.

Troubleshooting & tips

- If your Word files use paragraph styles for headings (not all-caps or 'CHAPTER' text), we can extend `prepare_bible_from_docx.py` to inspect `python-docx` paragraph styles (more robust). Tell me if you want that.
- For large vocabularies, reduce edge clutter by increasing `--min-count` or lowering top-n similarities (edit the script to use topn=5) or keep `--light-edges` on.

Next steps I can take for you

- Add an option to `run_all.sh` to accept `--config` and `--split-by-headings` and pass them through (convenience). I can add this if you prefer one command that accepts the JSON config.
- Improve heading detection by using `python-docx` paragraph style names (strongly recommended if your files are well-structured Word docs).

If you'd like, I can implement the `run_all.sh` config flag now and/or add paragraph-style-based splitting.

New command-line options (added in this fork)
------------------------------------------------

This fork adds a few additional options to `scripts/train_from_folder.py` to let you control the embedding backend and the node-selection strategy used when exporting Topogram CSVs.

- `--embedding {word2vec,fasttext,glove}` — choose which embedding backend to use.
	- `word2vec` (default) trains a gensim Word2Vec model on the prepared per-file CSVs.
	- `fasttext` trains a gensim FastText model (subword-aware).
	- `glove` loads pre-trained GloVe vectors; you must pass `--glove-path /path/to/glove.txt`.

- `--glove-path <path>` — path to a pre-trained GloVe text file (required if `--embedding glove`). The script converts GloVe to word2vec format and loads it as KeyedVectors.

- `--max-nodes N` — limit the number of output nodes to the top N tokens. Useful to reduce graph size.

- `--select-by {connectivity,freq}` — when `--max-nodes` is set, choose how to select the top N tokens:
	- `connectivity` (default) — for each candidate token the script inspects its top-K similar neighbors (via `most_similar`) and ranks tokens by how many neighbors are present in the corpus; ties are broken by token frequency.
	- `freq` — selects the top N tokens by frequency (fast; uses model counts if available or falls back to corpus counts).

- `--similar-topn K` — how many neighbors to request from `most_similar` when computing connectivity (default: 10). Lower values are faster.

Examples
--------

1) Train with default Word2Vec and produce the combined Topogram CSV (no node limit):

```bash
python3 bible_word2vec/scripts/train_from_folder.py --folder data/Bible_CSV --out data/Bible_CSV/bible_topogram.csv
```

2) Train with FastText and limit output to the top 500 most connected tokens (connectivity selection):

```bash
python3 bible_word2vec/scripts/train_from_folder.py \
	--folder data/Bible_CSV --out data/Bible_CSV/bible_topogram_fasttext_top500.csv \
	--embedding fasttext --max-nodes 500 --select-by connectivity --similar-topn 10
```

3) Use pre-trained GloVe vectors (no training) and limit nodes by frequency:

```bash
python3 bible_word2vec/scripts/train_from_folder.py \
	--folder data/Bible_CSV --out data/Bible_CSV/bible_topogram_glove.csv \
	--embedding glove --glove-path /path/to/glove.6B.100d.txt --max-nodes 300 --select-by freq
```

Notes
-----

- Connectivity-based selection requires many `most_similar` calls (one per candidate token) and may be slow for very large vocabularies or large models. Use `--select-by freq` or reduce `--similar-topn` when you need speed.
- If you run with `--embedding glove` and the pre-trained file lacks token counts, the script falls back to counting occurrences in your corpus for frequency tie-breaks.
