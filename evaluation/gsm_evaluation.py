"""
Extended GSM evaluation pipeline.

Provides:
  - Exact-number comparison metrics
  - Sample partitioning into 5 JSON categories
  - BERTScore CSV exports
  - Descriptive statistics computation
  - Distribution visualizations
"""

from __future__ import annotations

import csv
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def ensure_directory(path: Path) -> Path:
    """Create directory if it doesn't exist, return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_numeric(text: str) -> str | None:
    """Extract numeric value from text, handling common formats."""
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # Handle "#### N" format (GSM8K)
    match = re.search(r"####\s*([\d,.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    
    # Try to find any number
    # Handle negative numbers, decimals, etc.
    match = re.search(r"[\d,.\-]+", text)
    if match:
        return match.group(0).replace(",", "").strip()
    
    return None


def numeric_equivalent(val1: str, val2: str) -> bool:
    """Check if two string representations of numbers are equivalent."""
    try:
        num1 = float(val1.strip())
        num2 = float(val2.strip())
        return abs(num1 - num2) < 1e-9
    except (ValueError, AttributeError):
        return val1.strip() == val2.strip()


# ---------------------------------------------------------------------------
# Exact-Number Comparison Metrics
# ---------------------------------------------------------------------------

def compare_exact_numbers(
    ground_truths: list[str],
    prompt_responses: list[str],
    compressed_responses: list[str],
) -> dict[str, float]:
    """
    Compute exact-number comparison metrics.
    
    Returns:
      - gt_vs_prompt_avg: Average match score (ground_truth vs prompt_response)
      - gt_vs_compressed_avg: Average match score (ground_truth vs compressed_response)
    """
    gt_vs_prompt_matches = 0
    gt_vs_compressed_matches = 0
    total = 0
    
    for gt, prompt_resp, compressed_resp in zip(
        ground_truths, prompt_responses, compressed_responses
    ):
        if not gt or not prompt_resp or not compressed_resp:
            continue
        
        total += 1
        
        # Extract numeric values
        gt_num = extract_numeric(gt)
        prompt_num = extract_numeric(prompt_resp)
        compressed_num = extract_numeric(compressed_resp)
        
        # Compare with ground truth
        if gt_num and prompt_num and numeric_equivalent(gt_num, prompt_num):
            gt_vs_prompt_matches += 1
        
        if gt_num and compressed_num and numeric_equivalent(gt_num, compressed_num):
            gt_vs_compressed_matches += 1
    
    return {
        "GT_vs_PromptResponse_Avg": (
            gt_vs_prompt_matches / total if total else 0.0
        ),
        "GT_vs_CompressedResponse_Avg": (
            gt_vs_compressed_matches / total if total else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# GSM Sample Partitioning
# ---------------------------------------------------------------------------

def partition_gsm_samples(
    records: list[dict[str, Any]],
    ground_truths: list[str],
    prompt_responses: list[str],
    compressed_responses: list[str],
    indices: list[int],
) -> dict[str, list[dict[str, Any]]]:
    """
    Partition GSM samples into 5 mutually exclusive categories.
    
    Categories:
      - all_same: All three match
      - gt_and_compressed_same: GT and compressed match; prompt differs
      - gt_and_prompt_same: GT and prompt match; compressed differs
      - compressed_and_prompt_same: Compressed and prompt match; GT differs
      - all_different: All three differ
    
    Returns dict with category names as keys and list of enriched records as values.
    """
    categories = defaultdict(list)
    
    for idx, gt, prompt_resp, compressed_resp in zip(
        indices, ground_truths, prompt_responses, compressed_responses
    ):
        record = records[idx].copy()
        record["index"] = idx
        
        # Extract numeric values
        gt_num = extract_numeric(gt)
        prompt_num = extract_numeric(prompt_resp)
        compressed_num = extract_numeric(compressed_resp)
        
        # Compare using numeric equivalence
        gt_vs_prompt = (
            gt_num and prompt_num and numeric_equivalent(gt_num, prompt_num)
        )
        gt_vs_compressed = (
            gt_num and compressed_num and numeric_equivalent(gt_num, compressed_num)
        )
        prompt_vs_compressed = (
            prompt_num and compressed_num and numeric_equivalent(prompt_num, compressed_num)
        )
        
        # Categorize
        if gt_vs_prompt and gt_vs_compressed:
            categories["all_same"].append(record)
        elif gt_vs_compressed and not gt_vs_prompt:
            categories["gt_and_compressed_same"].append(record)
        elif gt_vs_prompt and not gt_vs_compressed:
            categories["gt_and_prompt_same"].append(record)
        elif prompt_vs_compressed and not gt_vs_prompt:
            categories["compressed_and_prompt_same"].append(record)
        else:
            categories["all_different"].append(record)
    
    return dict(categories)


def save_partitioned_samples(
    categories: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Save partitioned GSM samples to 5 separate JSON files."""
    ensure_directory(output_dir)
    
    for category_name, samples in categories.items():
        output_file = output_dir / f"{category_name}.json"
        output_file.write_text(
            json.dumps(samples, indent=2) + "\n",
            encoding="utf-8"
        )
        print(f"  Saved {len(samples)} samples to {output_file.name}")


def run_bertscore(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
) -> dict[str, list[float]]:
    """Compute individual BERTScore values for each prediction/reference pair."""
    from bert_score import score

    precision, recall, f1 = score(predictions, references, lang=lang, verbose=False)
    return {
        "precision": [float(p) for p in precision],
        "recall": [float(r) for r in recall],
        "f1": [float(f) for f in f1],
    }


def save_bertscore_csv(
    indices: list[int],
    bert_scores: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save individual BERTScore values to a CSV file."""
    ensure_directory(output_path.parent)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "BERTScore_P", "BERTScore_R", "BERTScore_F1"])

        for row_idx, idx in enumerate(indices):
            p = bert_scores.get("precision", [])[row_idx] if row_idx < len(bert_scores.get("precision", [])) else 0.0
            r = bert_scores.get("recall", [])[row_idx] if row_idx < len(bert_scores.get("recall", [])) else 0.0
            f = bert_scores.get("f1", [])[row_idx] if row_idx < len(bert_scores.get("f1", [])) else 0.0
            writer.writerow([idx, p, r, f])

    print(f"  Saved BERTScore CSV to {output_path.name}")


def export_bertscore_analysis(
    indices: list[int],
    bert_scores: dict[str, list[float]],
    output_dir: Path,
    csv_filename: str,
    stats_filename: str,
    plot_subdir: Path,
    metric_title: str,
) -> tuple[Path, Path, Path]:
    """Save BERTScore CSV, statistics, and distribution plots."""
    output_dir = ensure_directory(output_dir)

    csv_path = output_dir / csv_filename
    save_bertscore_csv(indices, bert_scores, csv_path)

    stats = compute_bertscore_statistics(indices, bert_scores)
    stats_path = output_dir / stats_filename
    save_statistics_json(stats, stats_path)

    print(print_statistics(stats, metric_title))

    plot_dir = ensure_directory(output_dir / plot_subdir)
    create_distribution_plots(indices, bert_scores, plot_dir, dataset_name=metric_title)

    return csv_path, stats_path, plot_dir

# ---------------------------------------------------------------------------
# Descriptive Statistics
# ---------------------------------------------------------------------------

def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute mean, median, and standard deviation."""
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
        }
    
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std_dev": float(np.std(arr)),
    }


def compute_bertscore_statistics(
    indices: list[int],
    bert_scores: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """
    Compute statistics for BERTScore metrics.
    
    Returns dict with keys 'P', 'R', 'F1', each containing stats.
    """
    # Normalize key names
    precision_key = next(
        (k for k in bert_scores.keys() if "precision" in k.lower() or "p" in k.lower()),
        None
    )
    recall_key = next(
        (k for k in bert_scores.keys() if "recall" in k.lower() or "r" in k.lower()),
        None
    )
    f1_key = next(
        (k for k in bert_scores.keys() if "f1" in k.lower() or "f" in k.lower()),
        None
    )
    
    stats = {}
    
    if precision_key:
        p_values = [bert_scores[precision_key][i] for i in indices]
        stats["P"] = compute_statistics(p_values)
    
    if recall_key:
        r_values = [bert_scores[recall_key][i] for i in indices]
        stats["R"] = compute_statistics(r_values)
    
    if f1_key:
        f_values = [bert_scores[f1_key][i] for i in indices]
        stats["F1"] = compute_statistics(f_values)
    
    return stats


def save_statistics_json(
    stats: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Save statistics to JSON file."""
    ensure_directory(output_path.parent)
    output_path.write_text(
        json.dumps(stats, indent=2) + "\n",
        encoding="utf-8"
    )
    print(f"  Saved statistics to {output_path.name}")


def print_statistics(
    stats: dict[str, dict[str, float]],
    metric_name: str = "BERTScore",
) -> str:
    """Pretty-print statistics and return as string."""
    lines = [f"\n{metric_name} Statistics:"]
    lines.append("=" * 50)
    
    for key in ["P", "R", "F1"]:
        if key in stats:
            s = stats[key]
            lines.append(f"\n{metric_name}_{key}:")
            lines.append(f"  Mean:       {s.get('mean', 0.0):.6f}")
            lines.append(f"  Median:     {s.get('median', 0.0):.6f}")
            lines.append(f"  Std Dev:    {s.get('std_dev', 0.0):.6f}")
    
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Distribution Visualizations
# ---------------------------------------------------------------------------

def create_distribution_plots(
    indices: list[int],
    bert_scores: dict[str, list[float]],
    output_dir: Path,
    dataset_name: str = "GSM",
) -> None:
    """
    Create histograms/KDE plots for BERTScore distributions.
    
    Generates 3 separate images: one each for P, R, F1.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available; skipping distribution plots")
        return
    
    ensure_directory(output_dir)
    
    # Normalize key names
    precision_key = next(
        (k for k in bert_scores.keys() if "precision" in k.lower() or "p" in k.lower()),
        None
    )
    recall_key = next(
        (k for k in bert_scores.keys() if "recall" in k.lower() or "r" in k.lower()),
        None
    )
    f1_key = next(
        (k for k in bert_scores.keys() if "f1" in k.lower() or "f" in k.lower()),
        None
    )
    
    metric_configs = [
        ("P", precision_key, "Precision"),
        ("R", recall_key, "Recall"),
        ("F1", f1_key, "F1-Score"),
    ]
    
    for abbrev, key, full_name in metric_configs:
        if key is None:
            continue
        
        values = [bert_scores[key][i] for i in indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram with KDE
        ax.hist(values, bins=20, density=True, alpha=0.7, color="skyblue", edgecolor="black")
        
        # Add KDE if scipy is available
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            x_range = np.linspace(min(values), max(values), 200)
            ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
            ax.legend()
        except ImportError:
            pass
        
        ax.set_title(f"BERTScore {abbrev} Distribution - {dataset_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel(f"BERTScore {full_name}", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(alpha=0.3)
        
        output_file = output_dir / f"bertscore_{abbrev.lower()}_distribution.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        print(f"  Saved distribution plot to {output_file.name}")
