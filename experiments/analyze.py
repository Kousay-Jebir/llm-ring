"""Summarize experiment runs into a table and charts.

Usage:
    python -m experiments.analyze --results-dir experiments/results/raw --out-dir experiments/results

Metrics (per run, then averaged per policy when runs are repeated):
    hit_rate            share of prompt tokens served from cache, follow-up turns only
    hit_rate_after      same, in the window right after the new replica became ready
    moved_after_scale   share of conversations whose turns straddling the scale-up
                        went to a different replica
    p50/p95 latency     end-to-end request latency seen by the client
    p95 prompt_ms       time spent (re)processing prompts: what caching saves
    imbalance           busiest replica's work / average work, before the scale-up
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

NUMERIC = ["turn", "t_start_s", "t_end_s", "status", "latency_ms", "est_prompt_tokens",
           "cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms"]


def load_runs(results_dir: Path) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    runs = []
    for meta_path in sorted(results_dir.glob("*.json")):
        csv_path = meta_path.with_suffix(".csv")
        if not csv_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(csv_path)
        for column in NUMERIC:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        runs.append((meta, frame))
    return runs


def _hit_rate(frame: pd.DataFrame) -> float:
    total = (frame["cache_n"] + frame["prompt_n"]).sum()
    return float(frame["cache_n"].sum() / total) if total else float("nan")


def run_metrics(meta: dict[str, Any], frame: pd.DataFrame, window_s: float) -> dict[str, Any]:
    ok = frame[frame["status"] == 200].dropna(subset=["cache_n", "prompt_n"])
    follow_ups = ok[ok["turn"] > 1]
    ready_s = meta.get("new_replica_ready_s")

    metrics: dict[str, Any] = {
        "run_id": meta["run_id"],
        "policy": meta["policy"],
        "requests": len(frame),
        "errors": int((frame["status"] != 200).sum()),
        "hit_rate": _hit_rate(follow_ups),
        "p50_latency_ms": float(ok["latency_ms"].quantile(0.50)) if len(ok) else float("nan"),
        "p95_latency_ms": float(ok["latency_ms"].quantile(0.95)) if len(ok) else float("nan"),
        "p95_prompt_ms": float(ok["prompt_ms"].quantile(0.95)) if len(ok) else float("nan"),
        "hit_rate_after": float("nan"),
        "moved_after_scale": float("nan"),
        "imbalance": float("nan"),
    }

    before = ok if ready_s is None else ok[ok["t_start_s"] < ready_s]
    work = (before["prompt_n"] + before["predicted_n"].fillna(0)).groupby(before["replica"]).sum()
    if len(work) and work.mean() > 0:
        metrics["imbalance"] = float(work.max() / work.mean())

    if ready_s is not None:
        after = follow_ups[(follow_ups["t_start_s"] >= ready_s) & (follow_ups["t_start_s"] < ready_s + window_s)]
        metrics["hit_rate_after"] = _hit_rate(after)
        moved = considered = 0
        for _, turns in ok.sort_values("turn").groupby("conversation_id"):
            records = turns.to_dict("records")
            for prev, cur in zip(records, records[1:]):
                if prev["t_start_s"] < ready_s <= cur["t_start_s"]:
                    considered += 1
                    moved += prev["replica"] != cur["replica"]
                    break
        if considered:
            metrics["moved_after_scale"] = moved / considered
    return metrics


def plot_hit_rate_over_time(runs, out: Path, bin_s: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for meta, frame in runs:
        ok = frame[(frame["status"] == 200) & (frame["turn"] > 1)].dropna(subset=["cache_n", "prompt_n"])
        if ok.empty:
            continue
        bins = (ok["t_start_s"] // bin_s) * bin_s
        grouped = ok.groupby(bins)
        rate = grouped["cache_n"].sum() / (grouped["cache_n"].sum() + grouped["prompt_n"].sum())
        line, = ax.plot(rate.index, rate.values, marker="o", markersize=3, label=meta["policy"])
        if meta.get("new_replica_ready_s") is not None:
            ax.axvline(meta["new_replica_ready_s"], color=line.get_color(), linestyle=":", alpha=0.6)
    ax.set(xlabel="time since start (s)", ylabel="prompt tokens served from cache",
           title="Cache hit rate over time (dotted: new replica ready)", ylim=(0, 1.05))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_work_per_replica(runs, out: Path, bin_s: float) -> None:
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 2.6 * len(runs)), sharex=True, squeeze=False)
    for ax, (meta, frame) in zip(axes[:, 0], runs):
        ok = frame[frame["status"] == 200].dropna(subset=["prompt_n"])
        work = ok.assign(work=ok["prompt_n"] + ok["predicted_n"].fillna(0),
                         bin=(ok["t_start_s"] // bin_s) * bin_s)
        table = work.pivot_table(index="bin", columns="replica", values="work", aggfunc="sum", fill_value=0)
        for replica in sorted(table.columns):
            ax.plot(table.index, table[replica], label=replica)
        if meta.get("new_replica_ready_s") is not None:
            ax.axvline(meta["new_replica_ready_s"], color="gray", linestyle=":")
        ax.set_title(meta["policy"], fontsize=10)
        ax.set_ylabel("tokens processed")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("time since start (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, out: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    summary["hit_rate"].plot.bar(ax=left, color="#4C9A8A")
    left.set(title="Cache hit rate (follow-up turns)", ylim=(0, 1), ylabel="")
    summary["p95_latency_ms"].plot.bar(ax=right, color="#C77C3B")
    right.set(title="p95 latency (ms)", ylabel="")
    for ax in (left, right):
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summarize(results_dir: Path, out_dir: Path, window_s: float = 60.0, bin_s: float = 15.0) -> pd.DataFrame:
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"no runs found in {results_dir} (expected matching .json and .csv files)")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run = pd.DataFrame([run_metrics(meta, frame, window_s) for meta, frame in runs])
    order = list(dict.fromkeys(per_run["policy"]))  # keep the order in which policies ran
    columns = ["requests", "errors", "hit_rate", "hit_rate_after", "moved_after_scale",
               "p50_latency_ms", "p95_latency_ms", "p95_prompt_ms", "imbalance"]
    summary = per_run.groupby("policy")[columns].mean().reindex(order)
    summary.insert(0, "runs", per_run.groupby("policy").size().reindex(order))

    per_run.to_csv(out_dir / "per_run.csv", index=False)
    rounded = summary.round(3)
    # DataFrame.to_markdown needs the optional 'tabulate' package; fall back to plain text.
    table = rounded.to_markdown() if _has_tabulate() else f"```\n{rounded.to_string()}\n```"
    (out_dir / "summary.md").write_text(f"# Results\n\n{table}\n", encoding="utf-8")
    plot_hit_rate_over_time(runs, out_dir / "hit_rate_over_time.png", bin_s)
    plot_work_per_replica(runs, out_dir / "work_per_replica.png", bin_s)
    plot_summary(summary, out_dir / "summary.png")
    return summary


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401  (optional, used by DataFrame.to_markdown)
    except ImportError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="experiments/results/raw")
    parser.add_argument("--out-dir", default="experiments/results")
    parser.add_argument("--window-s", type=float, default=60.0, help="window after the scale-up")
    parser.add_argument("--bin-s", type=float, default=15.0, help="time bin for charts")
    args = parser.parse_args()
    summary = summarize(Path(args.results_dir), Path(args.out_dir), args.window_s, args.bin_s)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.round(3))


if __name__ == "__main__":
    main()
