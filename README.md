# safety-tuned-refusal

A runnable, modular take on the original Colab notebook for generating refusal-oriented prompts, collecting model responses, judging them with Gemini, and probing hidden states for safety and identity signals.

## Quickstart
- Create an environment and install: `python -m venv .venv && source .venv/bin/activate && pip install -e .`
- Generate prompts: `safety-tuned-refusal generate-prompts --output data/prompts.csv`
- Collect aligned responses (defaults to temp 0.7 and 3 samples/prompt; needs HF auth and GPU/CPU time):  
  `safety-tuned-refusal generate-responses --prompts data/prompts.csv --output data/responses_aligned.csv --model meta-llama/Meta-Llama-3-8B-Instruct --n-samples 5`
- Label with Gemini (needs `GOOGLE_API_KEY`):  
  `safety-tuned-refusal judge --responses data/responses_aligned.csv --output data/responses_aligned_labeled.csv`
- Report P(safety) by identity with bootstrap CIs (no HF model needed):  
  `safety-tuned-refusal report-safety --labeled-responses data/responses_aligned_labeled.csv --risk-level all --output data/identity_bootstrap.csv`
- Run probes/plots (heavy; expects GPU):  
  `safety-tuned-refusal analyze --labeled-responses data/responses_aligned_labeled.csv --save-prefix outputs/llama3`

## Commands
- `generate-prompts` -> writes the full prompt grid (`prompt_id`, identity metadata, `prompt_text`) to CSV.
- `generate-responses` -> loads an aligned HF model, wraps each prompt with the instruction, and writes `full_prompt`, `response_text`, and `sample_index` (multi-sample aware).
- `judge` -> calls Gemini to categorize each response as Refusal/Hedged/Direct, adds `behavior_label_llm`, `safety_binary`, and rationale columns.
- `report-safety` -> estimates P(safety) per identity with bootstrap CIs from labeled responses.
- `analyze` -> reuses the HF model to extract hidden states, trains linear probes, saves `.npy` tensors and `.png` plots, and prints a summary report.

Use `--help` on any subcommand for options (model id, device, temperatures, rate limits, etc.).

## Project layout
- `pyproject.toml` - dependencies and CLI entrypoint.
- `src/safety_tuned_refusal/prompts.py` - prompt templates, identity axes, instruction wrapper.
- `src/safety_tuned_refusal/inference.py` - HF model loader and response generation.
- `src/safety_tuned_refusal/judge.py` - Gemini-based judge with rate limiting.
- `src/safety_tuned_refusal/probes.py` - linear probes, cosine overlap, and plotting utilities.
- `src/safety_tuned_refusal/cli.py` - orchestration CLI wiring everything together.
- `7000_final.py` - original Colab notebook for reference.

## Notes and tips
- Set `GOOGLE_API_KEY` in your environment before running `judge`.
- GPU strongly recommended for `generate-responses` and required for the heavy `analyze` probe step; use `--device cpu` to force CPU when experimenting.
- HF chat template is used for prompts; pad token is aligned to EOS by default for LLaMA models.
- Use `--n-samples` (e.g., 5) and `--temperature 0.7` on `generate-responses` to follow the multi-sample Option A workflow.
- Output file sizes grow quickly (CSV + `.npy` tensors + `.png` plots); point `--output`/`--save-prefix` to a writable directory.
- The probing pipeline mirrors the notebook: it extracts last-token prompt states, first-response-token states, runs identity/safety probes, and plots overlap metrics.
