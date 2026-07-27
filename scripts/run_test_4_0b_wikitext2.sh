#!/usr/bin/env bash
set -euo pipefail

ROOT="data/wikitext2-public"
OUT="results/test-4-0b/latest"
mkdir -p "$ROOT" "$OUT"

python experiments/prepare_wikitext2_bpe.py \
  --root "$ROOT" \
  --stats "$OUT/dataset_stats.json" \
  --vocab-size 2048

cp "$ROOT/tokenizer.json" "$OUT/tokenizer.json"

python experiments/test_4_0_wikitext2_scale64_controller.py \
  --source experiments/test_2_0_modal_moe_trainability.py \
  --progressive-source experiments/test_2_6_progressive_nested_modes.py \
  --text "$ROOT/manifest.json" \
  --self-test

python experiments/test_4_0_wikitext2_scale64_controller.py \
  --source experiments/test_2_0_modal_moe_trainability.py \
  --progressive-source experiments/test_2_6_progressive_nested_modes.py \
  --text "$ROOT/manifest.json" \
  --output-dir "$OUT" \
  --steps 400 \
  --seed 40400 \
  --threads 4 \
  --controller-train-batches 40 \
  --controller-tune-batches 16 \
  --test-batches 24 \
  --controller-steps 700 \
  --controller-hidden 64 \
  --oracle-tolerance 0.05 \
  2>&1 | tee wikitext2-public.log
