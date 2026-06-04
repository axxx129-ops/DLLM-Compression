from __future__ import annotations
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.gsm_evaluation import (
    compare_exact_numbers,
    export_bertscore_analysis,
    partition_gsm_samples,
    run_bertscore,
    save_partitioned_samples,
    ensure_directory,
)

def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def fallback_bleu(reference: str, prediction: str) -> float:
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    if not ref_tokens or not pred_tokens:
        return 0.0

    precisions = []
    for n in range(1, 5):
        pred_ngrams = ngrams(pred_tokens, n)
        ref_ngrams = ngrams(ref_tokens, n)
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        precisions.append((overlap + 1) / (total + 1))

    brevity_penalty = 1.0
    if len(pred_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1 - len(ref_tokens) / len(pred_tokens))

    return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / 4)


def bleu_score(references: list[str], predictions: list[str]) -> dict[str, float]:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        smooth = SmoothingFunction().method1
        scores = [
            sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
            for ref, pred in zip(references, predictions)
            if ref.strip() and pred.strip()
        ]
    except ImportError:
        scores = [
            fallback_bleu(ref, pred)
            for ref, pred in zip(references, predictions)
            if ref.strip() and pred.strip()
        ]

    return {"BLEU": sum(scores) / len(scores) if scores else 0.0}


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for i, right_token in enumerate(right, start=1):
            current.append(previous[i - 1] + 1 if left_token == right_token else max(previous[i], current[-1]))
        previous = current
    return previous[-1]


def rouge_from_overlap(overlap: int, pred_total: int, ref_total: int) -> dict[str, float]:
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
    return {"F": f1, "P": precision, "R": recall}


def fallback_rouge_pair(reference: str, prediction: str) -> dict[str, dict[str, float]]:
    ref_tokens = reference.split()
    pred_tokens = prediction.split()

    rouge_1 = rouge_from_overlap(
        sum((ngrams(pred_tokens, 1) & ngrams(ref_tokens, 1)).values()),
        len(pred_tokens),
        len(ref_tokens),
    )
    rouge_2 = rouge_from_overlap(
        sum((ngrams(pred_tokens, 2) & ngrams(ref_tokens, 2)).values()),
        max(len(pred_tokens) - 1, 0),
        max(len(ref_tokens) - 1, 0),
    )
    rouge_l = rouge_from_overlap(lcs_length(pred_tokens, ref_tokens), len(pred_tokens), len(ref_tokens))
    return {"ROUGE_1": rouge_1, "ROUGE_2": rouge_2, "ROUGE_L": rouge_l}


def average_nested(scores: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    if not scores:
        return {
            "ROUGE_1": {"F": 0.0, "P": 0.0, "R": 0.0},
            "ROUGE_2": {"F": 0.0, "P": 0.0, "R": 0.0},
            "ROUGE_L": {"F": 0.0, "P": 0.0, "R": 0.0},
        }

    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for score in scores:
        for metric, values in score.items():
            for score_type, value in values.items():
                totals[metric][score_type] += value

    return {
        metric: {score_type: value / len(scores) for score_type, value in values.items()}
        for metric, values in totals.items()
    }

def rouge_score(references: list[str], predictions: list[str]) -> dict[str, dict[str, float]]:
    try:
        from rouge import Rouge

        rouge = Rouge()
        pairs = [
            rouge.get_scores([pred], [ref])[0]
            for ref, pred in zip(references, predictions)
            if ref.strip() and pred.strip()
        ]
        normalized = [
            {
                "ROUGE_1": {"F": p["rouge-1"]["f"], "P": p["rouge-1"]["p"], "R": p["rouge-1"]["r"]},
                "ROUGE_2": {"F": p["rouge-2"]["f"], "P": p["rouge-2"]["p"], "R": p["rouge-2"]["r"]},
                "ROUGE_L": {"F": p["rouge-l"]["f"], "P": p["rouge-l"]["p"], "R": p["rouge-l"]["r"]},
            }
            for p in pairs
        ]
    except ImportError:
        normalized = [
            fallback_rouge_pair(ref, pred)
            for ref, pred in zip(references, predictions)
            if ref.strip() and pred.strip()
        ]

    return average_nested(normalized)


def bert_score(references: list[str], predictions: list[str]) -> dict[str, float]:
    if not references:
        return {"BERTScore_P": 0.0, "BERTScore_R": 0.0, "BERTScore_F1": 0.0}

    import torch
    from bert_score import score

    precision, recall, f1 = score(predictions, references, lang="en", verbose=False)
    return {
        "BERTScore_P": float(torch.mean(precision)),
        "BERTScore_R": float(torch.mean(recall)),
        "BERTScore_F1": float(torch.mean(f1)),
    }


def flatten_scores(scores: dict[str, Any], prefix: str) -> dict[str, float]:
    flat = {}
    for metric, value in scores.items():
        if isinstance(value, dict):
            for score_type, nested_value in value.items():
                if isinstance(nested_value, dict):
                    for nested_type, final_value in nested_value.items():
                        flat[f"{prefix}_{score_type}_{nested_type}"] = float(final_value)
                else:
                    flat[f"{prefix}_{metric}_{score_type}"] = float(nested_value)
        else:
            flat[f"{prefix}_{metric}"] = float(value)
    return flat


def run_metrics(references: list[str], predictions: list[str], include_bertscore: bool) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    scores.update(bleu_score(references, predictions))
    scores.update(rouge_score(references, predictions))
    if include_bertscore:
        scores.update(bert_score(references, predictions))
    return scores


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(as_text(item) for item in value)
    return str(value)


def load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    raise ValueError(f"{path} must be a JSON list or an object with a 'results' list.")


def dataset_name(path: Path, records: list[dict[str, Any]]) -> str:
    for record in records:
        metadata = record.get("metadata", {})
        for key in ("dataset", "dataset_name", "task"):
            if metadata.get(key):
                return str(metadata[key])
    return path.stem.removesuffix("_results").removesuffix("-results")


def compression_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    origin = [float(r["result"]["origin_tokens"]) for r in records if "origin_tokens" in r.get("result", {})]
    compressed = [
        float(r["result"]["compressed_tokens"]) for r in records if "compressed_tokens" in r.get("result", {})
    ]
    avg_origin = sum(origin) / len(origin) if origin else 0.0
    avg_compressed = sum(compressed) / len(compressed) if compressed else 0.0
    return {
        "Average_Original_Prompt_Tokens": avg_origin,
        "Average_Compressed_Prompt_Tokens": avg_compressed,
        "Average_Compression_Rate": avg_compressed / avg_origin if avg_origin else 0.0,
    }

def exact_match_score(references: list[str], predictions: list[str]) -> dict[str, float]:
    total = 0
    correct = 0

    for ref, pred in zip(references, predictions):
        ref = ref.strip()
        pred = pred.strip()

        if not ref or not pred:
            continue

        total += 1

        if ref == pred:
            correct += 1

    return {
        "Exact_Match": correct / total if total else 0.0
    }

def compute_ratio_difference(references: list[str], predictions: list[str]) -> dict[str, float]:
    total = 0
    ratio_differences = []

    for ref, pred in zip(references, predictions):
        ref = ref.strip()
        pred = pred.strip()

        if not ref or not pred:
            continue

        total += 1

        try:
            ref_num = float(ref)
            pred_num = float(pred)
            if ref_num != 0:
                ratio_differences.append(abs(ref_num - pred_num) / abs(ref_num))
        except ValueError:
            continue

    return {
        "Average_Ratio_Difference": sum(ratio_differences) / len(ratio_differences) if ratio_differences else 0.0
    }

def evaluate_gsm_records(
    records: list[dict[str, Any]],
    include_bertscore: bool,
    name: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate GSM records with extended metrics including:
      - Exact-number comparison (GT vs prompt, GT vs compressed)
      - Sample partitioning into 5 JSON files
      - BERTScore CSV exports
      - Descriptive statistics
      - Distribution visualizations
    """
    prompts = []
    reconstructions = []

    ground_truths = []
    original_answers = []
    compressed_answers = []

    # Track indices for partitioning
    math_indices = []

    for idx, record in enumerate(records):
        prompt = as_text(record.get("prompt", ""))
        reconstruction = as_text(
            record.get("reconstruction_response", "")
        )
        if reconstruction.startswith("Answer:"):
            reconstruction = reconstruction[len("Answer:"):].strip()

        gt = as_text(record.get("ground_truth", ""))
        compressed_response = as_text(
            record.get("compressed_prompt_response", "")
        )

        prompt_response = as_text(
            record.get("prompt_response", "")
        )

        if prompt and reconstruction:
            prompts.append(prompt)
            reconstructions.append(reconstruction)

        if gt and prompt_response and compressed_response:
            ground_truths.append(gt)
            original_answers.append(prompt_response)
            compressed_answers.append(compressed_response)
            math_indices.append(idx)

    reconstruction_metrics = flatten_scores(
        run_metrics(
            prompts,
            reconstructions,
            include_bertscore
        ),
        "Reconstruction"
    )

    # Compute exact-match metrics
    math_metrics = exact_match_score(
        original_answers,
        compressed_answers
    )

    ratio_metrics = compute_ratio_difference(
        original_answers,
        compressed_answers
    )

    math_metrics.update(ratio_metrics)

    # Compute exact-number comparison metrics
    if ground_truths and original_answers and compressed_answers:
        number_metrics = compare_exact_numbers(
            ground_truths,
            original_answers,
            compressed_answers,
        )
        math_metrics.update(number_metrics)

    result = {
        "Dataset": name,
        "Dataset_Type": "gsm",
        "Total_Records": len(records),
        "Evaluated_Reconstruction_Records": len(prompts),
        "Evaluated_Math_Records": len(ground_truths),
        "Compression": compression_stats(records),
        "Reconstruction_Metrics": reconstruction_metrics,
        "Math_Metrics": math_metrics,
    }

    # Generate extended outputs if output directory is specified
    if output_dir and ground_truths:
        output_dir = ensure_directory(output_dir)

        # 1. Save partitioned samples
        print(f"\nPartitioning GSM samples...")
        categories = partition_gsm_samples(
            records,
            ground_truths,
            original_answers,
            compressed_answers,
            math_indices,
        )

        partition_dir = ensure_directory(output_dir / "partitions")
        save_partitioned_samples(categories, partition_dir)
        result["Partition_Dir"] = str(partition_dir)

        if include_bertscore and prompts and reconstructions:
            print(f"\nExporting BERTScore metrics for Prompt vs Reconstruction...")
            try:
                bert_scores = run_bertscore(reconstructions, prompts)

                csv_path, stats_path, plot_dir = export_bertscore_analysis(
                    indices=list(range(len(prompts))),
                    bert_scores=bert_scores,
                    output_dir=output_dir,
                    csv_filename="bertscores_prompt_vs_reconstructed.csv",
                    stats_filename="bertscore_stats_prompt_vs_reconstructed_response.json",
                    plot_subdir=Path("distribution_plots") / "prompt_vs_reconstructed",
                    metric_title="Prompt vs ReconstructedPrompt BERTScore",
                )

                result["BERTScore_Dir"] = str(output_dir)
                result["Stats_Dir"] = str(output_dir)
                result["Plots_Dir"] = str(plot_dir)

            except ImportError:
                print("  bert-score not available; skipping BERTScore CSV export")

    return result


def evaluate_summary_records(
    records: list[dict[str, Any]],
    include_bertscore: bool,
    name: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    prompt_responses = []
    compressed_responses = []
    indices = []

    for idx, record in enumerate(records):
        prompt_response = as_text(record.get("prompt_response", ""))
        compressed_response = as_text(record.get("compressed_prompt_response", ""))

        if prompt_response and compressed_response:
            prompt_responses.append(prompt_response)
            compressed_responses.append(compressed_response)
            indices.append(idx)

    response_metrics = flatten_scores(
        run_metrics(prompt_responses, compressed_responses, include_bertscore),
        "Summary",
    )

    result = {
        "Dataset": name,
        "Dataset_Type": "summarization",
        "Total_Records": len(records),
        "Evaluated_Records": len(prompt_responses),
        "Compression": compression_stats(records),
        "Summary_Metrics": response_metrics,
    }

    if include_bertscore and output_dir and prompt_responses:
        output_dir = ensure_directory(output_dir / name)
        try:
            print(f"\nExporting BERTScore metrics for {name}...")
            bert_scores = run_bertscore(prompt_responses, compressed_responses)

            csv_path, stats_path, plot_dir = export_bertscore_analysis(
                indices=indices,
                bert_scores=bert_scores,
                output_dir=output_dir,
                csv_filename=f"{name}_bertscores.csv",
                stats_filename=f"{name}_bertscore_statistics.json",
                plot_subdir=Path("distribution_plots") / name,
                metric_title=f"{name} BERTScore",
            )

            result["BERTScore_CSV"] = str(csv_path)
            result["BERTScore_Stats"] = str(stats_path)
            result["BERTScore_Plots"] = str(plot_dir)

        except ImportError:
            print("bert-score not installed; skipping BERTScore exports")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM prompt-compression result JSON files.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--gsm-output-dir", type=Path, help="Directory for extended GSM evaluation outputs (partitions, CSVs, plots)")
    parser.add_argument("--summary-output-dir", type=Path, help="Directory for summarization BERTScore outputs")
    parser.add_argument("--skip-bertscore", action="store_true")
    return parser.parse_args()


def is_gsm_dataset(records: list[dict[str, Any]]) -> bool:
    if not records:
        return False

    sample = records[0]
    if "ground_truth" in sample:
        return True
    if "reconstruction_response" in sample:
        return True

    metadata = sample.get("metadata", {})
    if "answer" in metadata:
        return True

    return False


def main() -> None:
    args = parse_args()
    datasets = []

    for path in args.inputs:
        records = load_records(path)
        name = dataset_name(path, records)
        include_bertscore = not args.skip_bertscore

        if is_gsm_dataset(records):
            # For GSM datasets, pass output directory for extended evaluation
            result = evaluate_gsm_records(
                records,
                include_bertscore,
                name,
                output_dir=args.gsm_output_dir,
            )
        else:
            result = evaluate_summary_records(
                records,
                include_bertscore,
                name,
                output_dir=args.summary_output_dir,
            )

        datasets.append(result)

    report = {"Datasets": datasets}
    rendered = json.dumps(report, indent=2)

    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)


if __name__ == "__main__":
    main()
