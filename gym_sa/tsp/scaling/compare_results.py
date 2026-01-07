import argparse
import os
import re
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    print("This script requires pandas and matplotlib. Please install them:\n  pip install pandas matplotlib")
    raise


METRICS_TO_AVERAGE = ["best_objective", "avg_acceptance_rate", "std", "temperature"]
CSV_REQUIRED_COLUMNS = ["step", "best_objective", "avg_acceptance_rate", "std", "temperature", "runtime"]


def extract_chain_count_from_name(name: str) -> Optional[int]:
    """
    Extract the first integer found in a folder name to represent the number of chains.
    Supports names like '1', 'chains_4', '4chains', 'nchains-08', etc.
    """
    match = re.search(r"(\d+)", name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def discover_chain_dirs(root_dir: str) -> List[Tuple[int, str]]:
    """
    Return a list of tuples (chain_count, absolute_path) for subdirectories in root_dir
    that contain at least one CSV file.
    """
    chain_dirs: List[Tuple[int, str]] = []
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            chain_count = extract_chain_count_from_name(entry.name)
            if chain_count is None:
                continue
            csvs = glob(os.path.join(entry.path, "*.csv"))
            if not csvs:
                continue
            chain_dirs.append((chain_count, entry.path))
    chain_dirs.sort(key=lambda x: x[0])
    return chain_dirs


def read_and_validate_csv(path: str) -> Optional[pd.DataFrame]:
    """
    Read a CSV and ensure required columns are present. Coerce numeric types sensibly.
    Returns None if validation fails.
    """
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to read CSV '{path}': {exc}")
        return None

    missing = [c for c in CSV_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"Warning: CSV '{path}' missing required columns: {missing}. Skipping.")
        return None

    # Coerce numeric columns
    for col in ["step"] + METRICS_TO_AVERAGE + ["runtime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing steps
    df = df.dropna(subset=["step"])
    # Ensure integer-like step for safe sorting/grouping
    if not pd.api.types.is_integer_dtype(df["step"]):
        df["step"] = df["step"].round().astype(int)

    # Keep only relevant columns (but keep runtime if needed later)
    keep_cols = ["step"] + METRICS_TO_AVERAGE + ["runtime"]
    df = df[keep_cols]
    return df


def aggregate_folder(folder: str) -> Optional[pd.DataFrame]:
    """
    Aggregate all CSVs in a folder by averaging metrics across replicates at each step.
    Returns a DataFrame with columns: step + METRICS_TO_AVERAGE + runtime (averaged),
    sorted by step.
    """
    csv_paths = sorted(glob(os.path.join(folder, "*.csv")))
    if not csv_paths:
        return None

    frames: List[pd.DataFrame] = []
    for p in csv_paths:
        df = read_and_validate_csv(p)
        if df is None or df.empty:
            continue
        frames.append(df)

    if not frames:
        return None

    all_logs = pd.concat(frames, ignore_index=True)
    # Group by step and average the target metrics
    grouped = (
        all_logs.groupby("step", as_index=False)[METRICS_TO_AVERAGE + ["runtime"]]
        .mean()
        .sort_values("step")
        .reset_index(drop=True)
    )
    return grouped


def plot_metrics(aggregated_by_chains: Dict[int, pd.DataFrame], output_path: str) -> None:
    """
    Plot each metric in METRICS_TO_AVERAGE on its own subplot, comparing lines for different chain counts.
    """
    if not aggregated_by_chains:
        print("No aggregated data to plot.")
        return

    n_metrics = len(METRICS_TO_AVERAGE)
    nrows, ncols = 2, 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()

    # Consistent color cycle
    chain_counts_sorted = sorted(aggregated_by_chains.keys())

    for idx, metric in enumerate(METRICS_TO_AVERAGE):
        ax = axes[idx]
        for chains in chain_counts_sorted:
            df = aggregated_by_chains[chains]
            if metric not in df.columns or df.empty:
                continue
            ax.plot(df["step"], df[metric], label=f"{chains} chains", linewidth=2)

        if metric == "best_objective":
            ax.set_title("Objective (negated)")
            ax.set_ylabel("Objective (negated; minimization)")
        else:
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel("Step")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Put legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="lower center", ncols=min(4, len(labels)))
        plt.subplots_adjust(bottom=0.08)

    fig.suptitle("Simulated Annealing (Adaptive Cooling) - TSP: Averaged Metrics Across Replicates", y=0.98, fontsize=14)

    # Save and also display
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")
    # try:
    #     plt.show()
    # except Exception:
    #     # Headless environments may fail to show; ignore
    #     pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TSP SA results across different numbers of parallel chains.")
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Root directory containing per-chain subfolders (default: script directory).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to save the output figure (default: <root>/compare_results.png).",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    if not os.path.isdir(root_dir):
        print(f"Root directory does not exist or is not a directory: {root_dir}")
        sys.exit(1)

    chain_dirs = discover_chain_dirs(root_dir)
    if not chain_dirs:
        print(f"No valid chain directories with CSVs found under: {root_dir}")
        sys.exit(0)

    aggregated_by_chains: Dict[int, pd.DataFrame] = {}
    for chains, folder in chain_dirs:
        agg = aggregate_folder(folder)
        if agg is None or agg.empty:
            print(f"Warning: no valid data in '{folder}', skipping.")
            continue
        aggregated_by_chains[chains] = agg

    if not aggregated_by_chains:
        print("No aggregated data available after processing all folders.")
        sys.exit(0)

    # Convert maximization to minimization by negating the objective
    for df in aggregated_by_chains.values():
        if "best_objective" in df:
            df["best_objective"] = -df["best_objective"]

    output_path = args.out or os.path.join(root_dir, "compare_results.png")
    plot_metrics(aggregated_by_chains, output_path)


if __name__ == "__main__":
    main()


