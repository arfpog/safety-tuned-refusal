"""
Model loading and response generation utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from .config import GenerationConfig


def resolve_device(device: str | None = None) -> str:
    """Pick an available device when not explicitly provided."""
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(config: GenerationConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """Load a causal LM and tokenizer with sensible defaults."""
    device = resolve_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    full_prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    n_samples: int = 1,
    device: str,
) -> str | list[str]:
    """Generate one or more responses given a fully formatted prompt."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0.0 or n_samples > 1
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": do_sample,
        "num_return_sequences": n_samples,
    }
    if do_sample:
        safe_temp = temperature if temperature > 0 else 1.0
        gen_kwargs["temperature"] = safe_temp

    output_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    texts = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated_ids]
    return texts[0] if n_samples == 1 else texts


def generate_responses_for_prompts(
    prompts_df: pd.DataFrame,
    build_instruction: Callable[[str], str],
    *,
    config: GenerationConfig,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Generate aligned responses for every prompt in a DataFrame.

    The prompts_df must contain a `prompt_text` column. Returns a new DataFrame
    with model_name, model_type, full_prompt, response_text, and sample_index columns added.
    """
    model, tokenizer, device = load_model(config)
    all_rows = []

    for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df), desc="Generating aligned responses"):
        prompt_text = row["prompt_text"]
        full_prompt = build_instruction(prompt_text)
        responses = generate_response(
            model,
            tokenizer,
            full_prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            n_samples=config.n_samples,
            device=device,
        )
        if isinstance(responses, str):
            responses = [responses]

        for sample_idx, response in enumerate(responses, start=1):
            prompt_id = row.get("prompt_id")
            new_row = row.to_dict()
            new_row.update(
                {
                    "model_name": config.model_name,
                    "model_type": "aligned",
                    "full_prompt": full_prompt,
                    "response_text": response,
                    "sample_index": sample_idx,
                    "sample_id": f"{prompt_id}_s{sample_idx}" if prompt_id else f"sample_{sample_idx}",
                    "temperature": config.temperature,
                    "n_samples": config.n_samples,
                }
            )
            all_rows.append(new_row)

    responses_df = pd.DataFrame(all_rows)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        responses_df.to_csv(output_csv, index=False)
    return responses_df


__all__ = [
    "resolve_device",
    "load_model",
    "generate_response",
    "generate_responses_for_prompts",
]
