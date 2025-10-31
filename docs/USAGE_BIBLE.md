Bible Word2Vec workflow — notes

This folder contains a small helper to convert `.docx` files to per-file CSVs suitable for the main training pipeline.

Steps

1. Place your .docx files in a folder (one file per book/part). Filenames will be used as the `title` column.
2. Convert them to CSV per-file:

```bash
python bible_word2vec/scripts/prepare_bible_from_docx.py --input-folder /path/to/docx --out-folder data/Bible_CSV --artist "Bible"
```

3. Run the central training script from the project root:

```bash
python scripts/train_from_folder.py --folder data/Bible_CSV --out data/bible_topogram.csv --min-token-length 3 --extra-stopwords="the,and,of"
```

Tips

- Use `--min-token-length` and `--extra-stopwords` to tune which words become nodes. Biblical texts include many short function words; set `--min-token-length 4` if you want to aggressively filter.
- Use `--light-edges` to create a visually lighter graph.

