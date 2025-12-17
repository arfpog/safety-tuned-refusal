#!/usr/bin/env bash
set -euo pipefail

# Run probing analysis on labeled responses.
# Env overrides:
#   MODEL          HF model id to reuse for probing (default: meta-llama/Meta-Llama-3-8B-Instruct)
#   DEVICE         Force device, e.g., cuda, cpu, mps (default: auto)
#   LABELED        Path to labeled CSV (default: data/responses_aligned_labeled.csv)
#   BATCH_SIZE     Batch size for hidden state extraction (default: 4)
#   N_SPLITS       CV folds for probes (default: 5)
#   N_SHUFFLES     Shuffled label runs (default: 5)
#   C              Inverse regularization strength (default: 1.0)
#   SAVE_PREFIX    Prefix for saved outputs (default: outputs/analysis)

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
DEVICE="${DEVICE:-}"
LABELED="${LABELED:-data/responses_aligned_labeled.csv}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_SPLITS="${N_SPLITS:-5}"
N_SHUFFLES="${N_SHUFFLES:-5}"
C="${C:-1.0}"
SAVE_PREFIX="${SAVE_PREFIX:-outputs/analysis}"

if [[ -n "${DEVICE}" ]]; then
  DEVICE_FLAG=(--device "${DEVICE}")
else
  DEVICE_FLAG=()
fi

safety-tuned-refusal analyze \
  --labeled-responses "${LABELED}" \
  --model "${MODEL}" \
  --batch-size "${BATCH_SIZE}" \
  --n-splits "${N_SPLITS}" \
  --n-shuffles "${N_SHUFFLES}" \
  -C "${C}" \
  --save-prefix "${SAVE_PREFIX}" \
  "${DEVICE_FLAG[@]}"
