#!/usr/bin/env bash
# Small wrapper to run prepare + train inside the bible_word2vec fork.
# Usage: ./run_all.sh /path/to/doc_files /path/to/output_folder
# Example: ./run_all.sh input_docs data/Bible_CSV

set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <input_doc_folder> <out_csv_folder> [artist_name]"
  exit 1
fi
INPUT_DIR="$1"
OUT_CSV_DIR="$2"
ARTIST_NAME="${3:-Bible}"

echo "Converting Word files in $INPUT_DIR -> $OUT_CSV_DIR (artist='$ARTIST_NAME')"
python3 bible_word2vec/scripts/prepare_bible_from_docx.py --input-folder "$INPUT_DIR" --out-folder "$OUT_CSV_DIR" --artist "$ARTIST_NAME"

# train within the fork using the copied training script and Bible-friendly defaults
OUT_TOPOGRAM="$OUT_CSV_DIR/bible_topogram.csv"
echo "Training Word2Vec and exporting topogram CSV to $OUT_TOPOGRAM"
python3 bible_word2vec/scripts/train_from_folder.py --folder "$OUT_CSV_DIR" --out "$OUT_TOPOGRAM" --min-token-length 3 --light-edges --manifest "$OUT_CSV_DIR/manifest.csv"

echo "Done. Outputs in $OUT_CSV_DIR (manifest + per_book graphs under per_song_graphs/)"
