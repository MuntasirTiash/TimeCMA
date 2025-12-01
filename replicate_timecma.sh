#!/usr/bin/env bash
set -euo pipefail

# ======= CONFIG (edit these) =======
# Where you unzipped CSVs (files named ETTm1.csv, ETTh1.csv, etc.)
: "${TIMECMA_DATA_ROOT:=$HOME/timecma_data}"
# Where to store last-token embeddings (.h5)
: "${TIMECMA_EMB_ROOT:=$HOME/timecma_embeddings}"

# GPU selection (optional)
: "${CUDA_VISIBLE_DEVICES:=0}"

# ======= Sanity =======
echo "[Info] Using TIMECMA_DATA_ROOT=$TIMECMA_DATA_ROOT"
echo "[Info] Using TIMECMA_EMB_ROOT=$TIMECMA_EMB_ROOT"
mkdir -p "$TIMECMA_DATA_ROOT" "$TIMECMA_EMB_ROOT"

# ======= Patch code to accept env overrides (non-destructive, idempotent) =======
# We will modify only defaults in data loaders to read env vars when present.
apply_patch() {
python - <<'PY'
import io, os, re, sys, pathlib

def patch_file(p, patterns):
    txt = p.read_text()
    orig = txt
    for pat, repl in patterns:
        txt = re.sub(pat, repl, txt, flags=re.S)
    if txt != orig:
        p.write_text(txt)
        print(f"[Patched] {p}")
    else:
        print(f"[Skip] {p} (already patched)")


root = pathlib.Path(".")

# 1) data_provider/data_loader_emb.py: dataset ROOT and EMBEDDINGS path
p1 = root/"data_provider"/"data_loader_emb.py"
if p1.exists():
    # Replace default root_path string to env fallback
    patch_file(p1, [
        (r'def __init__\(self, root_path="?/mnt/sfs-common/dataset/"?,', r'def __init__(self, root_path=os.getenv("TIMECMA_DATA_ROOT","/mnt/sfs-common/dataset/"),'),
        # Replace any f-string or join that hardcodes "/mnt/sfs-common/TimeCMA/Embeddings"
        (r'"/mnt/sfs-common/TimeCMA/Embeddings"', r'os.getenv("TIMECMA_EMB_ROOT","/mnt/sfs-common/TimeCMA/Embeddings")'),
    ])

# 2) storage/store_emb.py: ensure it writes to TIMECMA_EMB_ROOT if set
p2 = root/"storage"/"store_emb.py"
if p2.exists():
    patch_file(p2, [
        (r'"/mnt/sfs-common/TimeCMA/Embeddings"', r'os.getenv("TIMECMA_EMB_ROOT","/mnt/sfs-common/TimeCMA/Embeddings")'),
        (r'root_path\s*=\s*"?/mnt/sfs-common/dataset/"?', r'root_path = os.getenv("TIMECMA_DATA_ROOT","/mnt/sfs-common/dataset/")'),
    ])

# 3) data_provider/data_loader.py (plain loader), in case code path uses it
p3 = root/"data_provider"/"data_loader.py"
if p3.exists():
    patch_file(p3, [
        (r'def __init__\(self, root_path="?/mnt/sfs-common/dataset/"?,', r'def __init__(self, root_path=os.getenv("TIMECMA_DATA_ROOT","/mnt/sfs-common/dataset/"),'),
    ])

print("[Done] Patch applied.")
PY
}

# ======= Generate embeddings for all datasets/splits =======
store_all_emb() {
  echo "[Step] Generating last-token embeddings (.h5) for all datasets/splits"
  # ETT family
  for ds in ETTm1 ETTm2 ETTh1 ETTh2; do
    for split in train val test; do
      echo ">> $ds / $split"
      python storage/store_emb.py --divide "$split" --data_path "$ds" --num_nodes 7 --input_len 96 --output_len 96
    done
  done

  # Weather
  for split in train val test; do
    echo ">> Weather / $split"
    python storage/store_emb.py --divide "$split" --data_path Weather --num_nodes 21 --input_len 96 --output_len 96
  done

  # ECL
  for split in train val test; do
    echo ">> ECL / $split"
    python storage/store_emb.py --divide "$split" --data_path ECL --num_nodes 321 --input_len 96 --output_len 96
  done

  # FRED and ILI (shorter context)
  for split in train val test; do
    echo ">> FRED / $split"
    python storage/store_emb.py --divide "$split" --data_path FRED --num_nodes 107 --input_len 36 --output_len 24
    echo ">> ILI / $split"
    python storage/store_emb.py --divide "$split" --data_path ILI --num_nodes 7 --input_len 36 --output_len 24
  done
}

# ======= Train/Eval across all datasets with paper hyperparams =======
run_all_experiments() {
  echo "[Step] Training/evaluating across datasets (logs under ./Results/<dataset>/...)"
  # ETTm1  (pred_len: 96,192,336,720)
  bash scripts/ETTm1.sh

  # ETTh1
  bash scripts/ETTh1.sh

  # ETTh2
  bash scripts/ETTh2.sh

  # ETTm2
  bash scripts/ETTm2.sh

  # Weather
  bash scripts/Weather.sh

  # ECL
  bash scripts/ECL.sh

  # FRED
  bash scripts/FRED.sh

  # ILI
  bash scripts/ILI.sh
}

# ======= Main menu =======
usage() {
  cat <<USAGE
Usage: $0 <step>

Steps:
  patch      - Patch code to honor TIMECMA_DATA_ROOT / TIMECMA_EMB_ROOT
  embeddings - Generate and save last-token embeddings for all datasets/splits
  train      - Run all dataset scripts (ETT*, Weather, ECL, FRED, ILI)
  all        - Do everything: patch -> embeddings -> train
USAGE
}

main() {
  local step="${1:-}"
  case "$step" in
    patch) apply_patch ;;
    embeddings) apply_patch; store_all_emb ;;
    train) run_all_experiments ;;
    all) apply_patch; store_all_emb; run_all_experiments ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
