from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cumulative_max_per_round(data_frame: pd.DataFrame) -> np.ndarray:
    """Return the cumulative max true score per round for a single run."""
    if "round" not in data_frame.columns or "true_score" not in data_frame.columns:
        raise KeyError("Input data must contain 'round' and 'true_score' columns")

    num_rounds = int(data_frame["round"].max()) + 1
    per_round_max = np.full(num_rounds, -np.inf)

    for round_idx, group in data_frame.groupby("round"):
        per_round_max[int(round_idx)] = float(group["true_score"].max())

    # Replace -inf (rounds without entries) with the best value seen so far
    per_round_max = np.maximum.accumulate(np.where(np.isfinite(per_round_max), per_round_max, -np.inf))
    if np.isneginf(per_round_max[0]):
        raise ValueError("Unable to compute cumulative maxima; check input CSV format")

    return per_round_max


def read_run(path: Path) -> np.ndarray:
    """Load a single CSV (with JSON metadata on first line) and compute cumulative max."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            # Consume metadata JSON line if present
            first_line = handle.readline()
            try:
                json.loads(first_line)
                df = pd.read_csv(handle)
            except json.JSONDecodeError:
                # Metadata missing; rewind and load everything
                handle.seek(0)
                df = pd.read_csv(handle)
    except Exception as exc:  # pragma: no cover - propagate richer context
        raise RuntimeError(f"Failed to process {path}") from exc

    if df.empty:
        raise ValueError(f"No data rows found in {path}")

    return cumulative_max_per_round(df)


def aggregate_runs(csv_paths: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) cumulative maxima across a set of runs."""
    runs: List[np.ndarray] = []
    for csv_path in csv_paths:
        runs.append(read_run(csv_path))

    if not runs:
        raise ValueError("No CSV files provided for aggregation")

    min_length = min(run.shape[0] for run in runs)
    aligned_runs = np.vstack([run[:min_length] for run in runs])

    return aligned_runs.mean(axis=0), aligned_runs.std(axis=0)


def parse_round_step(file_name: str) -> int:
    """Infer the number of model queries per round from the first numeric pattern."""
    parts = file_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected file naming pattern: {file_name}")
    try:
        return int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Cannot infer round step from file name {file_name}") from exc


def collect_variant_series(base_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Gather mean/std series for all subdirectories whose name contains 'rna'."""
    variant_stats: Dict[str, Dict[str, np.ndarray]] = {}

    for variant_dir in sorted(p for p in base_dir.iterdir() if p.is_dir() and "rna" in p.name.lower()):
        csv_files = sorted(variant_dir.glob("*.csv"))
        if not csv_files:
            continue

        try:
            mean_values, std_values = aggregate_runs(csv_files)
            step = parse_round_step(csv_files[0].name)
        except Exception as exc:
            print(f"Skipping {variant_dir.name}: {exc}")
            continue

        rounds = np.arange(mean_values.shape[0]) * step
        variant_stats[variant_dir.name] = {
            "mean": mean_values,
            "std": std_values,
            "rounds": rounds,
        }

    if not variant_stats:
        raise ValueError("No RNA variants with valid CSV data found under musearch/")

    return variant_stats


def plot_variants(stats: Dict[str, Dict[str, np.ndarray]], output_path: Path) -> None:
    plt.figure(figsize=(6, 6), dpi=180)
    color_cycle = plt.cm.tab20(np.linspace(0, 1, len(stats)))

    for (variant, series), color in zip(stats.items(), color_cycle):
        mean = series["mean"]
        std = series["std"]
        rounds = series["rounds"]

        (line,) = plt.plot(rounds, mean, label=variant, color=color)
        plt.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.2)

    plt.title("μSearch RNA Variant Comparison", fontweight="bold")
    plt.xlabel("Number of Model Queries", fontweight="bold")
    plt.ylabel("Cumulative Max True Score", fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved comparison figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot μSearch RNA variant performance curves")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/figs/musearch_rna_variants.png"),
        help="Path to save the output figure",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    musearch_dir = repo_root / "efficiency" / "musearch"

    if not musearch_dir.exists():
        raise FileNotFoundError(f"Expected directory not found: {musearch_dir}")

    stats = collect_variant_series(musearch_dir)
    plot_variants(stats, args.output)


if __name__ == "__main__":
    main()
