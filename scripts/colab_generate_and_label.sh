#!/usr/bin/env bash
set -euo pipefail

# End-to-end prompt generation, aligned responses, and Gemini judging.
# Set GOOGLE_API_KEY in the environment before running.
#
# Env overrides:
#   MODEL             HF model id for aligned responses (default: meta-llama/Meta-Llama-3-8B-Instruct)
#   DEVICE            Force device, e.g., cuda, cpu, mps (default: auto)
#   PROMPTS           Path to prompts CSV (default: data/prompts.csv)
#   RESPONSES         Path to responses CSV (default: data/responses_aligned.csv)
#   LABELED           Path to labeled CSV (default: data/responses_aligned_labeled.csv)
#   JUDGE_MODEL       Gemini judge model (default: gemini-2.5-flash-lite)
#   RATE_LIMIT_RPM    Requests per minute for Gemini (default: 15)

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
DEVICE="${DEVICE:-}"
PROMPTS="${PROMPTS:-data/prompts.csv}"
RESPONSES="${RESPONSES:-data/responses_aligned.csv}"
LABELED="${LABELED:-data/responses_aligned_labeled.csv}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-flash-lite}"
RATE_LIMIT_RPM="${RATE_LIMIT_RPM:-15}"

if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "GOOGLE_API_KEY not set; export it before running judge."
  exit 1
fi

# 1) Generate prompts
safety-tuned-refusal generate-prompts --output "${PROMPTS}"

# 2) Generate aligned responses
if [[ -n "${DEVICE}" ]]; then
  DEVICE_FLAG=(--device "${DEVICE}")
else
  DEVICE_FLAG=()
fi

safety-tuned-refusal generate-responses \
  --prompts "${PROMPTS}" \
  --output "${RESPONSES}" \
  --model "${MODEL}" \
  "${DEVICE_FLAG[@]}"

# 3) Judge with Gemini
safety-tuned-refusal judge \
  --responses "${RESPONSES}" \
  --output "${LABELED}" \
  --judge-model "${JUDGE_MODEL}" \
  --rate-limit-rpm "${RATE_LIMIT_RPM}"
