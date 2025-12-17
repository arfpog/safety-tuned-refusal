"""
Command-line interface for the safety-tuned refusal toolkit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import GenerationConfig, JudgeConfig
from .inference import generate_responses_for_prompts
from .judge import label_responses
from .metrics import compute_identity_safety_rates
from .prompts import build_instruction, generate_prompts, save_prompts_csv
from .probes import print_summary_report, run_full_analysis
from .inference import load_model


def cmd_generate_prompts(args: argparse.Namespace):
    df = generate_prompts()
    save_prompts_csv(Path(args.output))
    print(f"Wrote {len(df)} prompts to {args.output}")


def cmd_generate_responses(args: argparse.Namespace):
    prompts_df = pd.read_csv(args.prompts)
    config = GenerationConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        n_samples=args.n_samples,
        device=args.device,
    )
    output_csv = Path(args.output) if args.output else None
    responses_df = generate_responses_for_prompts(
        prompts_df, build_instruction, config=config, output_csv=output_csv
    )
    print(f"Generated {len(responses_df)} responses")
    if output_csv:
        print(f"Saved to {output_csv}")


def cmd_judge(args: argparse.Namespace):
    responses_df = pd.read_csv(args.responses)
    config = JudgeConfig(
        provider=args.provider,
        model_name=args.judge_model,
        rate_limit_rpm=args.rate_limit_rpm,
        api_key_env=args.api_key_env,
        openrouter_base_url=args.openrouter_base_url,
        openrouter_site_url=args.openrouter_site_url,
        openrouter_app_name=args.openrouter_app_name,
    )
    output_csv = Path(args.output) if args.output else None
    labeled_df = label_responses(responses_df, config, output_csv=output_csv)
    print(f"Labeled {len(labeled_df)} responses")
    if output_csv:
        print(f"Saved to {output_csv}")


def cmd_analyze(args: argparse.Namespace):
    df = pd.read_csv(args.labeled_responses)
    gen_config = GenerationConfig(model_name=args.model, device=args.device)
    model, tokenizer, device = load_model(gen_config)
    results = run_full_analysis(
        model=model,
        tokenizer=tokenizer,
        df=df,
        device=device,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
        n_shuffles=args.n_shuffles,
        C=args.C,
        save_prefix=args.save_prefix,
    )
    print_summary_report(results, df)


def cmd_report_safety(args: argparse.Namespace):
    df = pd.read_csv(args.labeled_responses)
    risk_level = None if args.risk_level == "all" else args.risk_level

    summary_df = compute_identity_safety_rates(
        df,
        risk_level=risk_level,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        random_state=args.random_state,
    )
    if summary_df.empty:
        print("No rows to summarize after filtering.")
        return

    summary_df = summary_df.sort_values(
        ["risk_level", "axis_id", "p_safety"], ascending=[True, True, False]
    )
    display_cols = [
        "risk_level",
        "axis_id",
        "identity_id",
        "p_safety",
        "ci_lower",
        "ci_upper",
        "n",
    ]
    print(summary_df[display_cols])

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        print(f"Wrote safety summary to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline for safety-tuned refusal prompt generation, response collection, judging, and probing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prompts = subparsers.add_parser("generate-prompts", help="Generate prompts CSV.")
    p_prompts.add_argument(
        "--output", default="prompts.csv", help="Path to write prompts CSV."
    )
    p_prompts.set_defaults(func=cmd_generate_prompts)

    p_responses = subparsers.add_parser(
        "generate-responses", help="Generate model responses for prompts."
    )
    p_responses.add_argument("--prompts", default="prompts.csv", help="Input prompts CSV.")
    p_responses.add_argument(
        "--output", default="responses_aligned.csv", help="Where to write responses CSV."
    )
    p_responses.add_argument(
        "--model", default=GenerationConfig().model_name, help="HF model id to load for aligned responses."
    )
    p_responses.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max new tokens to generate."
    )
    p_responses.add_argument(
        "--temperature",
        type=float,
        default=GenerationConfig().temperature,
        help="Sampling temperature (default matches Option A: 0.7).",
    )
    p_responses.add_argument(
        "--n-samples",
        type=int,
        default=GenerationConfig().n_samples,
        help="Number of samples per prompt (Option A recommends 3-5).",
    )
    p_responses.add_argument(
        "--device",
        default=None,
        help="Force device (cpu, cuda, mps). Defaults to auto-detection.",
    )
    p_responses.set_defaults(func=cmd_generate_responses)

    p_judge = subparsers.add_parser("judge", help="Label responses with an LLM judge.")
    p_judge.add_argument("--responses", default="responses_aligned.csv", help="Input responses CSV.")
    p_judge.add_argument(
        "--output",
        default="responses_aligned_labeled.csv",
        help="Where to write labeled responses CSV.",
    )
    p_judge.add_argument(
        "--provider",
        choices=["gemini", "openrouter"],
        default=JudgeConfig().provider,
        help="Which judge backend to use.",
    )
    p_judge.add_argument(
        "--judge-model",
        default=JudgeConfig().model_name,
        help="Judge model name (Gemini model id or OpenRouter model slug).",
    )
    p_judge.add_argument(
        "--rate-limit-rpm",
        type=int,
        default=JudgeConfig().rate_limit_rpm,
        help="Rate limit in requests per minute.",
    )
    p_judge.add_argument(
        "--api-key-env",
        default=JudgeConfig().api_key_env,
        help="Env var that holds the judge API key (Gemini or OpenRouter).",
    )
    p_judge.add_argument(
        "--openrouter-base-url",
        default=JudgeConfig().openrouter_base_url,
        help="OpenRouter API base URL (only used with --provider openrouter).",
    )
    p_judge.add_argument(
        "--openrouter-site-url",
        default=JudgeConfig().openrouter_site_url,
        help="Optional OpenRouter HTTP-Referer header value.",
    )
    p_judge.add_argument(
        "--openrouter-app-name",
        default=JudgeConfig().openrouter_app_name,
        help="Optional OpenRouter X-Title header value.",
    )
    p_judge.set_defaults(func=cmd_judge)

    p_report = subparsers.add_parser(
        "report-safety",
        help="Estimate P(safety) with bootstrap CIs from labeled responses.",
    )
    p_report.add_argument(
        "--labeled-responses",
        default="responses_aligned_labeled.csv",
        help="CSV with behavior_label_llm and safety_binary columns.",
    )
    p_report.add_argument(
        "--risk-level",
        choices=["all", "benign", "safety"],
        default="all",
        help="Filter to a risk level before bootstrapping.",
    )
    p_report.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for CIs.",
    )
    p_report.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level for the interval.",
    )
    p_report.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for bootstrapping.",
    )
    p_report.add_argument(
        "--output",
        help="Optional path to save the bootstrap summary CSV.",
    )
    p_report.set_defaults(func=cmd_report_safety)

    p_analyze = subparsers.add_parser(
        "analyze", help="Run probing analysis on labeled responses."
    )
    p_analyze.add_argument(
        "--labeled-responses",
        default="responses_aligned_labeled.csv",
        help="CSV with behavior_label_llm and safety_binary columns.",
    )
    p_analyze.add_argument(
        "--model", default=GenerationConfig().model_name, help="HF model id to reuse for probing."
    )
    p_analyze.add_argument(
        "--device",
        default=None,
        help="Force device (cpu, cuda, mps). Defaults to auto-detection.",
    )
    p_analyze.add_argument("--batch-size", type=int, default=4)
    p_analyze.add_argument("--n-splits", type=int, default=5, help="CV folds for probes.")
    p_analyze.add_argument("--n-shuffles", type=int, default=5, help="Shuffled label runs.")
    p_analyze.add_argument(
        "-C", type=float, default=1.0, dest="C", help="Inverse regularization strength."
    )
    p_analyze.add_argument(
        "--save-prefix", default="analysis", help="Prefix for saved probe outputs."
    )
    p_analyze.set_defaults(func=cmd_analyze)

    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
