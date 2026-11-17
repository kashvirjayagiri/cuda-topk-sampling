#!/usr/bin/env python3
"""Pair all four methods, collect speedups, and optionally plot a completed run."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

import numpy as np
from shared_pipeline import METHODS

CASE_FIELDS = ("source_sha256", "B", "V", "k", "temperature", "row_start",
               "timing_backend", "calls_per_sample")
GROUP_FIELDS = ("dataset_kind", "distribution", "V", "B", "k", "temperature", "method",
                "timing_backend", "calls_per_sample")


def load_rows(directory):
    directory = Path(directory)
    metadata = json.loads((directory/"metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError("Run is incomplete; do not treat partial measurements as final comparisons")
    with (directory/"summary.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or len(rows) != metadata.get("method_rows_written"):
        raise ValueError("Summary row count does not match the completed run metadata")
    for row in rows:
        for key in ("B", "V", "k", "row_start", "calls_per_sample", "warmup", "repetitions", "workspace_bytes"):
            row[key] = int(row[key])
        for key in ("temperature", "selection_median_us", "pipeline_median_us"):
            row[key] = float(row[key])
        if (row["correctness"] != "passed" or row["method"] not in METHODS
                or row["timing_backend"] != metadata["timing_backend"]
                or row["calls_per_sample"] != metadata["calls_per_sample"]
                or row["repetitions"] != metadata["repetitions"]
                or row["warmup"] != metadata["warmup"]):
            raise ValueError("Mixed timing settings, unknown method, or failed correctness row")
        for key in ("selection_median_us", "pipeline_median_us"):
            if not np.isfinite(row[key]) or row[key] <= 0:
                raise ValueError("Invalid timing value in summary")
    return metadata, rows


def paired_rows(rows):
    cases = defaultdict(dict)
    for row in rows:
        key = tuple(row[name] for name in CASE_FIELDS)
        if row["method"] in cases[key]:
            raise ValueError("Duplicate method/case row; cannot pair comparisons fairly")
        cases[key][row["method"]] = row
    output = []
    for methods in cases.values():
        if set(methods) != set(METHODS):
            raise ValueError("Each case must contain " + ", ".join(METHODS))
        for row in methods.values():
            result = dict(row)
            for scope in ("selection", "pipeline"):
                own = row[f"{scope}_median_us"]
                for baseline in ("full_sort", "pytorch"):
                    result[f"{scope}_speedup_vs_{baseline}"] = methods[baseline][f"{scope}_median_us"]/own
            output.append(result)
    return output


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in GROUP_FIELDS)].append(row)
    output = []
    for key, members in grouped.items():
        result = dict(zip(GROUP_FIELDS, key))
        result["window_count"] = len(members)
        result["group_count"] = len({x["source_sha256"] for x in members})
        for scope in ("selection", "pipeline"):
            values = [x[f"{scope}_median_us"] for x in members]
            result[f"{scope}_median_us"] = float(np.median(values))
            result[f"{scope}_between_window_p10_us"] = float(np.percentile(values, 10))
            result[f"{scope}_between_window_p90_us"] = float(np.percentile(values, 90))
            for baseline in ("full_sort", "pytorch"):
                result[f"{scope}_median_paired_speedup_vs_{baseline}"] = float(np.median(
                    [x[f"{scope}_speedup_vs_{baseline}"] for x in members]))
        result["workspace_bytes"] = max(x["workspace_bytes"] for x in members)
        output.append(result)
    return sorted(output, key=lambda r: tuple(str(r[k]) for k in GROUP_FIELDS))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(directory, aggregates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    output = Path(directory)/"plots"
    output.mkdir(exist_ok=True)
    conditions = sorted({(r["dataset_kind"], r["distribution"], r["V"], r["temperature"])
                         for r in aggregates})
    colors = dict(full_sort="#2563eb", hierarchical="#d97706",
                  radix="#7c3aed", pytorch="#059669")
    for kind, distribution, V, temperature in conditions:
        matching = [r for r in aggregates if (r["dataset_kind"], r["distribution"], r["V"], r["temperature"])
                    == (kind, distribution, V, temperature)]
        batches = sorted({r["B"] for r in matching})
        for scope in ("selection", "pipeline"):
            figure, axes = plt.subplots(1, len(batches), figsize=(4.3*len(batches), 4.4), squeeze=False)
            for ax, B in zip(axes[0], batches):
                for method in METHODS:
                    values = sorted((r for r in matching if r["B"] == B and r["method"] == method),
                                    key=lambda r: r["k"])
                    x = [r["k"] for r in values]
                    y = [r[f"{scope}_median_us"] for r in values]
                    ax.plot(x, y, marker="o", color=colors[method], label=method)
                    ax.fill_between(x, [r[f"{scope}_between_window_p10_us"] for r in values],
                                    [r[f"{scope}_between_window_p90_us"] for r in values],
                                    color=colors[method], alpha=.13)
                ax.set_title(f"B = {B}")
                ax.set_xlabel("k (selected candidates)")
                ax.set_ylabel("Latency (microseconds)")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ticks = sorted({r["k"] for r in matching})
                ax.set_xticks(ticks)
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.grid(True, which="major", alpha=.22)
            axes[0, 0].legend(fontsize=8, loc="upper left")
            label = "Selection + shared parallel sampler" if scope == "pipeline" else "Selection only"
            figure.suptitle(f"{label}\n{kind}: {distribution} | V={V:,}, T={temperature:g}", fontsize=11)
            figure.text(.5, .012, "Median of window medians.\nShading: p10–p90 across groups/windows; not a confidence interval.",
                        ha="center", fontsize=8)
            figure.tight_layout(rect=(0,.10,1,1))
            stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{kind}_{distribution}_V{V}_T{temperature:g}_{scope}")
            figure.savefig(output/f"{stem}.png", dpi=180)
            figure.savefig(output/f"{stem}.pdf")
            plt.close(figure)


def summarize_run(directory, plots=False):
    directory = Path(directory)
    metadata, rows = load_rows(directory)
    paired = paired_rows(rows)
    aggregates = aggregate_rows(paired)
    write_csv(directory/"comparisons.csv", paired)
    write_csv(directory/"aggregate.csv", aggregates)
    text = ["# Shared comparison: measurement notes", "",
            f"GPU: {metadata['gpu']}. Timing backend: {metadata['timing_backend']}.",
            f"Validated method/case rows: {len(rows)}. Repetitions per scope: {metadata['repetitions']}.",
            "", "All four methods use the same logits, FP32 precision, temperature, supplied random values, "
            "CUDA stream, and compiled parallel sampler. Selection and full pipeline are timed separately; "
            "pipeline time is measured directly, not computed by adding medians.", "",
            "`summary.csv` retains each dataset and row window. `comparisons.csv` adds paired speedups "
            "on identical configurations. `aggregate.csv` reports the median of window medians, separately "
            "by real/synthetic dataset kind, score distribution, vocabulary size, B, k, and temperature. "
            "Paired speedups are aggregated by their own median, not by dividing aggregate medians.", "",
            "Speedup = baseline latency / method latency. Values above 1 mean the method is faster. "
            "The PyTorch baseline includes tie repair/order and int32 conversion. Its raw Top-k diagnostic "
            "column does not satisfy the full ordering contract and is not used for those speedups. "
            "An advantage over this adapter is not automatically an advantage over bare torch.topk.", "",
            f"Random policy: {metadata['random_policy']}; CDF margin={metadata['random_margin']:g}. "
            "The exact values are in metadata.json. Boundary-conditioned inputs support robust numerical "
            "comparison, not claims about the statistical quality of random sampling.", "",
            "CUDA graph replay measurements exclude capture/setup and reduce host-dispatch differences. "
            "Eager measurements include API dispatch gaps and may include internal PyTorch workspace "
            "allocator costs. Do not combine the two timing modes in a comparison. Explicit workspace "
            "bytes exclude graph pools, library-internal temporaries, and allocator/process peaks. "
            "Radix workspace is the queried byte count, including CUB scratch, allocated before capture "
            "or timing. Graph and eager runs must be reported separately.", "",
            "Topic groups and row windows are different workloads, not independent timing repetitions. "
            "Plot shading shows workload variation, not a confidence interval. The real dataset is a "
            "controlled DistilGPT2 sample, not evidence about all models or text inputs.", "",
            "## Observed paired wins", ""]
    for method in ("hierarchical", "radix", "pytorch"):
        members = [r for r in paired if r["method"] == method]
        for baseline in ("full_sort", "pytorch"):
            if method == baseline: continue
            for scope in ("selection", "pipeline"):
                wins = sum(r[f"{scope}_speedup_vs_{baseline}"] > 1 for r in members)
                text.append(f"- {method} was faster than {baseline} for {scope} in {wins}/{len(members)} tested configurations/windows.")
    text += ["", "Use the CSVs/figures to discuss where performance changes with B, k, vocabulary "
             "size, and distribution. Avoid treating the win counts alone as an overall performance score.", ""]
    (directory/"analysis_notes.md").write_text("\n".join(text))
    if plots:
        make_plots(directory, aggregates)
    return paired, aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("--plots", action="store_true", help="Also create PNG/PDF report figures; requires matplotlib")
    args = parser.parse_args()
    try:
        paired, aggregates = summarize_run(args.results_directory, args.plots)
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(f"Saved paired comparisons ({len(paired)} rows), aggregate ({len(aggregates)} rows), and analysis_notes.md")
    if args.plots: print(f"Figures: {args.results_directory/'plots'}")


if __name__ == "__main__":
    main()
