"""
Linear probing utilities for analyzing identity and safety signals.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from .metrics import compute_identity_safety_rates

def prepare_batch(tokenizer, prompts: List[str], add_generation_prompt: bool = True) -> Dict[str, torch.Tensor]:
    """
    Build a chat-style batch for a list of user prompts.
    Returns a dict with input_ids and attention_mask on CPU.
    """
    batch_of_conversations = [[{"role": "user", "content": p}] for p in prompts]
    enc = tokenizer.apply_chat_template(
        batch_of_conversations,
        add_generation_prompt=add_generation_prompt,
        padding=True,
        return_tensors="pt",
    )
    if isinstance(enc, torch.Tensor):
        input_ids = enc
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("pad_token_id is None")
        attention_mask = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


@torch.no_grad()
def get_prompt_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 8,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Returns a tensor of shape [N, L+1, d]:
      - N: number of prompts
      - L+1: embeddings + each transformer layer
      - d: hidden size
    Each slice [i, :, :] is the per-layer vector at the last prompt token.
    """
    all_batches = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting prompt hidden states"):
        batch_prompts = prompts[i : i + batch_size]
        enc = prepare_batch(tokenizer, batch_prompts, add_generation_prompt=True)

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states

        last_idx = attention_mask.sum(dim=1) - 1

        per_layer = []
        for h in hidden_states:
            per_layer.append(h[torch.arange(h.size(0), device=device), last_idx])

        batch_tensor = torch.stack(per_layer, dim=1)
        all_batches.append(batch_tensor.cpu())

        del outputs, hidden_states, batch_tensor, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(all_batches, dim=0)


@torch.no_grad()
def get_response_start_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    batch_size: int = 4,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Returns hidden states at the first generated token position.
    Shape: [N, L+1, d]

    This captures the model's internal state at the moment it decides how to respond.
    """
    all_batches = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting response-start hidden states"):
        batch_prompts = prompts[i : i + batch_size]
        batch_responses = responses[i : i + batch_size]

        full_texts = []
        prompt_lengths = []

        for p, r in zip(batch_prompts, batch_responses):
            messages = [{"role": "user", "content": p}]
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_lengths.append(len(prompt_ids))

            first_response_token = tokenizer.encode(r, add_special_tokens=False)[:1]
            first_response_text = tokenizer.decode(first_response_token)
            full_texts.append(prompt_text + first_response_text)

        enc = tokenizer(
            full_texts, padding=True, return_tensors="pt", add_special_tokens=False
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states

        per_layer = []
        for h in hidden_states:
            batch_vecs = []
            for b_idx, plen in enumerate(prompt_lengths):
                nonpad_start = (attention_mask[b_idx] == 0).sum().item()
                response_pos = nonpad_start + plen
                response_pos = min(response_pos, h.size(1) - 1)
                batch_vecs.append(h[b_idx, response_pos])
            per_layer.append(torch.stack(batch_vecs))

        batch_tensor = torch.stack(per_layer, dim=1)
        all_batches.append(batch_tensor.cpu())

        del outputs, hidden_states, batch_tensor, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(all_batches, dim=0)


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """
    Train L2-regularized logistic regression with stratified k-fold CV.

    Returns:
        mean_acc: Mean cross-validated accuracy
        std_acc: Std of cross-validated accuracy
        coef: Coefficients from model trained on all data (for direction analysis)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=random_state,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))

    clf_full = LogisticRegression(
        C=C, max_iter=max_iter, solver="lbfgs", random_state=random_state
    )
    clf_full.fit(X, y)

    return np.mean(accuracies), np.std(accuracies), clf_full.coef_


def probe_across_layers(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    n_shuffles: int = 5,
    n_splits: int = 5,
    C: float = 1.0,
    random_state: int = 42,
) -> Dict:
    """
    Train probes at each layer with both true and shuffled labels.

    Returns dict with:
        - 'true_acc': [L+1] array of mean accuracies on true labels
        - 'true_std': [L+1] array of std accuracies
        - 'shuffled_acc': [L+1] array of mean accuracies on shuffled labels
        - 'shuffled_std': [L+1] array of std accuracies
        - 'coefs': [L+1, d] or [L+1, n_classes-1, d] array of coefficients
    """
    n_samples, n_layers, _ = hidden_states.shape

    true_accs = []
    true_stds = []
    shuffled_accs = []
    shuffled_stds = []
    coefs = []

    for layer in tqdm(range(n_layers), desc="Probing layers"):
        X = hidden_states[:, layer, :]

        mean_acc, std_acc, coef = train_linear_probe(
            X, labels, n_splits=n_splits, C=C, random_state=random_state
        )
        true_accs.append(mean_acc)
        true_stds.append(std_acc)
        coefs.append(coef)

        shuffle_accs_layer = []
        for s in range(n_shuffles):
            shuffled_y = np.random.RandomState(random_state + s).permutation(labels)
            s_mean, _, _ = train_linear_probe(
                X, shuffled_y, n_splits=n_splits, C=C, random_state=random_state + s
            )
            shuffle_accs_layer.append(s_mean)
        shuffled_accs.append(np.mean(shuffle_accs_layer))
        shuffled_stds.append(np.std(shuffle_accs_layer))

    return {
        "true_acc": np.array(true_accs),
        "true_std": np.array(true_stds),
        "shuffled_acc": np.array(shuffled_accs),
        "shuffled_std": np.array(shuffled_stds),
        "coefs": coefs,
    }


def run_identity_probes(
    prompt_hidden_states: np.ndarray,
    df: pd.DataFrame,
    identity_col: str = "identity_id",
    axis_col: str = "axis_id",
    one_vs_rest: bool = True,
    **probe_kwargs,
) -> Dict:
    """
    Run identity probes for each axis and identity.
    """
    results = {}

    for axis in df[axis_col].unique():
        axis_mask = df[axis_col] == axis
        axis_indices = np.where(axis_mask)[0]
        X_axis = prompt_hidden_states[axis_indices]
        df_axis = df[axis_mask]

        results[axis] = {}

        if one_vs_rest:
            for identity in df_axis[identity_col].unique():
                y = (df_axis[identity_col] == identity).astype(int).values
                probe_result = probe_across_layers(X_axis, y, **probe_kwargs)
                results[axis][identity] = probe_result
        else:
            le = LabelEncoder()
            y = le.fit_transform(df_axis[identity_col].values)
            probe_result = probe_across_layers(X_axis, y, **probe_kwargs)
            probe_result["classes"] = le.classes_
            results[axis]["multiclass"] = probe_result

    return results


def run_safety_probes(
    response_hidden_states: np.ndarray,
    safety_labels: np.ndarray,
    **probe_kwargs,
) -> Dict:
    """Run safety behavior probes (safety-inflected vs direct)."""
    return probe_across_layers(response_hidden_states, safety_labels, **probe_kwargs)


def run_safety_probes_from_prompt(
    prompt_hidden_states: np.ndarray,
    safety_labels: np.ndarray,
    **probe_kwargs,
) -> Dict:
    """Run safety probes on prompt representations."""
    return probe_across_layers(prompt_hidden_states, safety_labels, **probe_kwargs)


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    v1 = v1.flatten()
    v2 = v2.flatten()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def compute_identity_safety_overlap(
    identity_probes: Dict,
    safety_probes: Dict,
    axes: List[str],
    identities_per_axis: Dict[str, str],
) -> Dict:
    """
    Compute cosine similarity between identity and safety probe normals per layer.
    """
    n_layers = len(safety_probes["coefs"])
    safety_coefs = safety_probes["coefs"]

    results = {}

    for axis in axes:
        identity = identities_per_axis.get(axis)
        if identity is None or identity not in identity_probes.get(axis, {}):
            continue

        identity_coefs = identity_probes[axis][identity]["coefs"]

        cosines = []
        for layer in range(n_layers):
            v_id = identity_coefs[layer]
            v_safe = safety_coefs[layer]
            cos = compute_cosine_similarity(v_id, v_safe)
            cosines.append(cos)

        results[axis] = np.array(cosines)

    return results


def plot_probe_accuracy_curve(
    probe_results: Dict,
    title: str = "Probe Accuracy by Layer",
    chance_level: float = 0.5,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot accuracy curve with true vs shuffled labels."""
    layers = np.arange(len(probe_results["true_acc"]))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(layers, probe_results["true_acc"], "b-", label="True labels", linewidth=2)
    ax.fill_between(
        layers,
        probe_results["true_acc"] - probe_results["true_std"],
        probe_results["true_acc"] + probe_results["true_std"],
        alpha=0.2,
        color="blue",
    )

    ax.plot(
        layers,
        probe_results["shuffled_acc"],
        "r--",
        label="Shuffled labels",
        linewidth=2,
    )
    ax.fill_between(
        layers,
        probe_results["shuffled_acc"] - probe_results["shuffled_std"],
        probe_results["shuffled_acc"] + probe_results["shuffled_std"],
        alpha=0.2,
        color="red",
    )

    ax.axhline(y=chance_level, color="gray", linestyle=":", label=f"Chance ({chance_level:.2f})")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cross-validated Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.3, 1.05])

    plt.tight_layout()
    return fig


def plot_identity_probes_by_axis(
    identity_probes: Dict, axes: List[str], figsize: Tuple[int, int] = (14, 10)
):
    """Plot identity probe accuracies for each axis."""
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.flatten()

    for idx, axis in enumerate(axes[:4]):
        ax = axs[idx]

        if axis not in identity_probes:
            continue

        for identity, results in identity_probes[axis].items():
            if identity == "multiclass":
                continue
            layers = np.arange(len(results["true_acc"]))
            ax.plot(layers, results["true_acc"], label=f"{identity} (true)")

        first_identity = list(identity_probes[axis].keys())[0]
        if first_identity != "multiclass":
            results = identity_probes[axis][first_identity]
            ax.plot(layers, results["shuffled_acc"], "k--", label="Shuffled", alpha=0.7)

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{axis} Identity Probes")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.3, 1.05])

    plt.tight_layout()
    return fig


def plot_safety_probe(
    safety_probes: Dict,
    safety_probes_from_prompt: Optional[Dict] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot safety probe accuracy from response and optionally from prompt."""
    layers = np.arange(len(safety_probes["true_acc"]))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        layers,
        safety_probes["true_acc"],
        "b-",
        label="Response-start (true)",
        linewidth=2,
    )
    ax.fill_between(
        layers,
        safety_probes["true_acc"] - safety_probes["true_std"],
        safety_probes["true_acc"] + safety_probes["true_std"],
        alpha=0.2,
        color="blue",
    )
    ax.plot(
        layers,
        safety_probes["shuffled_acc"],
        "b--",
        label="Response-start (shuffled)",
        alpha=0.7,
    )

    if safety_probes_from_prompt is not None:
        ax.plot(
            layers,
            safety_probes_from_prompt["true_acc"],
            "g-",
            label="Prompt (true)",
            linewidth=2,
        )
        ax.fill_between(
            layers,
            safety_probes_from_prompt["true_acc"] - safety_probes_from_prompt["true_std"],
            safety_probes_from_prompt["true_acc"] + safety_probes_from_prompt["true_std"],
            alpha=0.2,
            color="green",
        )
        ax.plot(
            layers,
            safety_probes_from_prompt["shuffled_acc"],
            "g--",
            label="Prompt (shuffled)",
            alpha=0.7,
        )

    ax.axhline(y=0.5, color="gray", linestyle=":", label="Chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cross-validated Accuracy", fontsize=12)
    ax.set_title("Safety Behavior Probe Accuracy", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.3, 1.05])

    plt.tight_layout()
    return fig


def plot_identity_safety_cosine(
    cosine_results: Dict, figsize: Tuple[int, int] = (10, 6)
):
    """Plot identity-safety cosine similarity across layers."""
    fig, ax = plt.subplots(figsize=figsize)

    for axis, cosines in cosine_results.items():
        layers = np.arange(len(cosines))
        ax.plot(layers, cosines, label=axis, linewidth=2)

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("Identity-Safety Direction Alignment by Layer", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 0.5])

    plt.tight_layout()
    return fig


def plot_group_safety_rates(
    df: pd.DataFrame, risk_level: str = "safety", figsize: Tuple[int, int] = (12, 5)
):
    """Plot safety response rates by identity group."""
    sub = df[df["risk_level"] == risk_level] if risk_level else df

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    safety_rates = sub.groupby("identity_id")["safety_binary"].mean().sort_values(ascending=False)
    axs[0].barh(range(len(safety_rates)), safety_rates.values)
    axs[0].set_yticks(range(len(safety_rates)))
    axs[0].set_yticklabels(safety_rates.index)
    axs[0].set_xlabel("P(safety response)")
    axs[0].set_title(f"Safety Rate by Identity ({risk_level} prompts)")
    axs[0].grid(True, alpha=0.3, axis="x")

    axis_rates = sub.groupby("axis_id")["safety_binary"].mean()
    axs[1].bar(axis_rates.index, axis_rates.values)
    axs[1].set_ylabel("P(safety response)")
    axs[1].set_title(f"Safety Rate by Axis ({risk_level} prompts)")
    axs[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def run_full_analysis(
    model,
    tokenizer,
    df: pd.DataFrame,
    device: str = "cuda",
    batch_size: int = 4,
    n_splits: int = 5,
    n_shuffles: int = 5,
    C: float = 1.0,
    save_prefix: str = "analysis",
):
    """
    Run the complete analysis pipeline on labeled data.
    """
    results = {}

    prompts = df["prompt_text"].tolist()
    responses = df["response_text"].tolist()

    identity_bootstrap_frames = [
        compute_identity_safety_rates(df, risk_level=None),
        compute_identity_safety_rates(df, risk_level="safety"),
        compute_identity_safety_rates(df, risk_level="benign"),
    ]
    results["identity_bootstrap_rates"] = pd.concat(
        identity_bootstrap_frames, ignore_index=True
    )

    prompt_hidden = get_prompt_hidden_states(
        model, tokenizer, prompts, batch_size=batch_size, device=device
    )
    prompt_hidden_np = prompt_hidden.float().numpy()

    response_hidden = get_response_start_hidden_states(
        model, tokenizer, prompts, responses, batch_size=batch_size, device=device
    )
    response_hidden_np = response_hidden.float().numpy()

    np.save(f"{save_prefix}_prompt_hidden.npy", prompt_hidden_np)
    np.save(f"{save_prefix}_response_hidden.npy", response_hidden_np)

    identity_probes = run_identity_probes(
        prompt_hidden_np,
        df,
        one_vs_rest=True,
        n_splits=n_splits,
        n_shuffles=n_shuffles,
        C=C,
    )
    results["identity_probes"] = identity_probes

    safety_labels = df["safety_binary"].values

    safety_probes = run_safety_probes(
        response_hidden_np,
        safety_labels,
        n_splits=n_splits,
        n_shuffles=n_shuffles,
        C=C,
    )
    results["safety_probes"] = safety_probes

    safety_probes_prompt = run_safety_probes_from_prompt(
        prompt_hidden_np,
        safety_labels,
        n_splits=n_splits,
        n_shuffles=n_shuffles,
        C=C,
    )
    results["safety_probes_prompt"] = safety_probes_prompt

    identities_to_compare = {
        "SO": "gay_teen",
        "GI": "trans_woman",
        "REL": "muslim_student",
        "RACE": "black_student",
    }

    cosine_overlap = compute_identity_safety_overlap(
        identity_probes,
        safety_probes,
        axes=df["axis_id"].unique().tolist(),
        identities_per_axis=identities_to_compare,
    )
    results["cosine_overlap"] = cosine_overlap

    fig_identity = plot_identity_probes_by_axis(
        identity_probes, axes=df["axis_id"].unique().tolist()
    )
    fig_identity.savefig(f"{save_prefix}_identity_probes.png", dpi=150, bbox_inches="tight")

    fig_safety = plot_safety_probe(safety_probes, safety_probes_prompt)
    fig_safety.savefig(f"{save_prefix}_safety_probes.png", dpi=150, bbox_inches="tight")

    fig_cosine = plot_identity_safety_cosine(cosine_overlap)
    fig_cosine.savefig(f"{save_prefix}_cosine_overlap.png", dpi=150, bbox_inches="tight")

    fig_rates = plot_group_safety_rates(df, risk_level="safety")
    fig_rates.savefig(f"{save_prefix}_safety_rates.png", dpi=150, bbox_inches="tight")

    plt.close("all")
    return results


def print_summary_report(results: Dict, df: pd.DataFrame):
    """Print a summary of key findings."""
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY REPORT")
    print("=" * 70)

    print("\n--- BEHAVIORAL ANALYSIS ---")
    print(f"Total samples: {len(df)}")
    print(f"Overall safety rate: {df['safety_binary'].mean():.3f}")
    print("\nSafety rate by risk level:")
    print(df.groupby("risk_level")["safety_binary"].mean())

    identity_rates = results.get("identity_bootstrap_rates")
    if identity_rates is None or identity_rates.empty:
        identity_rates = compute_identity_safety_rates(df, risk_level=None)

    if identity_rates is not None and not identity_rates.empty:
        print("\n--- P(SAFETY) BY IDENTITY (BOOTSTRAP) ---")
        for risk_level, sub in identity_rates.groupby("risk_level"):
            sub_sorted = sub.sort_values("p_safety", ascending=False)
            print(f"\nRisk level: {risk_level}")
            for _, row in sub_sorted.iterrows():
                print(
                    f"  {row['axis_id']}::{row['identity_id']} (n={row['n']}): "
                    f"p={row['p_safety']:.3f} "
                    f"CI[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
                )

    print("\n--- IDENTITY PROBES ---")
    for axis, probes in results.get("identity_probes", {}).items():
        print(f"\n{axis}:")
        for identity, probe_result in probes.items():
            if identity == "multiclass":
                continue
            max_true = np.max(probe_result["true_acc"])
            max_layer = np.argmax(probe_result["true_acc"])
            max_shuffled = np.max(probe_result["shuffled_acc"])
            print(
                f"  {identity}: max acc={max_true:.3f} at layer {max_layer} "
                f"(shuffled max={max_shuffled:.3f})"
            )

    print("\n--- SAFETY PROBES ---")
    if "safety_probes" in results:
        sp = results["safety_probes"]
        max_acc = np.max(sp["true_acc"])
        max_layer = np.argmax(sp["true_acc"])
        max_shuffled = np.max(sp["shuffled_acc"])
        print(
            f"Response-start: max acc={max_acc:.3f} at layer {max_layer} "
            f"(shuffled max={max_shuffled:.3f})"
        )

    if "safety_probes_prompt" in results:
        sp = results["safety_probes_prompt"]
        max_acc = np.max(sp["true_acc"])
        max_layer = np.argmax(sp["true_acc"])
        print(f"Prompt: max acc={max_acc:.3f} at layer {max_layer}")

    print("\n--- IDENTITY-SAFETY COSINE OVERLAP ---")
    if "cosine_overlap" in results:
        for axis, cosines in results["cosine_overlap"].items():
            max_cos = np.max(np.abs(cosines))
            max_layer = np.argmax(np.abs(cosines))
            late_avg = np.mean(cosines[-10:]) if len(cosines) >= 10 else np.mean(cosines)
            print(
                f"{axis}: max |cos|={max_cos:.3f} at layer {max_layer}, "
                f"late-layer avg={late_avg:.3f}"
            )


__all__ = [
    "prepare_batch",
    "get_prompt_hidden_states",
    "get_response_start_hidden_states",
    "train_linear_probe",
    "probe_across_layers",
    "run_identity_probes",
    "run_safety_probes",
    "run_safety_probes_from_prompt",
    "compute_identity_safety_overlap",
    "plot_probe_accuracy_curve",
    "plot_identity_probes_by_axis",
    "plot_safety_probe",
    "plot_identity_safety_cosine",
    "plot_group_safety_rates",
    "run_full_analysis",
    "print_summary_report",
]
