"""Summarize experiment runs. Runs of the same policy are combined.

Reads <run_id>.csv (traffic/generator.py) and <run_id>.json (experiments/run.py).
Writes per_run.csv, summary.md, summary.png and hit_rate_over_time.png.

  hit_rate           share of prompt tokens served from cache, follow-up turns only
  hit_rate_after     same, in the window right after the new replica became ready
  moved_after_scale  share of conversations that changed replica across the scale-up
  latency, prompt    p50/p95 latency and p95 prompt processing time
  imbalance          busiest replica's work / average work, before the scale-up
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from traffic.generator import SETTINGS  # noqa: E402

NAN = float("nan")


def load(results_dir: Path) -> pd.DataFrame:
    """All requests of all runs in one table, with each run's scale-up time."""
    frames = []
    for meta_file in sorted(results_dir.glob("*.json")):
        if meta_file.with_suffix(".csv").exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            frames.append(pd.read_csv(meta_file.with_suffix(".csv")).assign(
                run_id=meta["run_id"], policy=meta["policy"], ready_s=meta.get("new_replica_ready_s", NAN)))
    if not frames:
        raise SystemExit(f"no runs in {results_dir} (expected matching .json and .csv files)")
    return pd.concat(frames, ignore_index=True)


def hit_rate(rows: pd.DataFrame) -> float:
    total = (rows["cache_n"] + rows["prompt_n"]).sum()
    return rows["cache_n"].sum() / total if total else NAN


def run_metrics(run: pd.DataFrame, window_s: float) -> dict:
    ok = run[run["status"] == 200].dropna(subset=["cache_n", "prompt_n"])
    follow = ok[ok["turn"] > 1]
    ready = run["ready_s"].iloc[0]
    before = ok if pd.isna(ready) else ok[ok["t_start_s"] < ready]
    work = (before["prompt_n"] + before["predicted_n"].fillna(0)).groupby(before["replica"]).sum()

    moved = considered = 0
    if not pd.isna(ready):
        for _, turns in ok.sort_values("turn").groupby("conversation_id"):
            rows = turns.to_dict("records")
            for prev, cur in zip(rows, rows[1:]):
                if prev["t_start_s"] < ready <= cur["t_start_s"]:  # first pair straddling the scale-up
                    considered += 1
                    moved += prev["replica"] != cur["replica"]
                    break
    after = follow[(follow["t_start_s"] >= ready) & (follow["t_start_s"] < ready + window_s)]

    return {
        "run_id": run["run_id"].iloc[0],
        "policy": run["policy"].iloc[0],
        "requests": len(run),
        "errors": int((run["status"] != 200).sum()),
        "hit_rate": hit_rate(follow),
        "hit_rate_after": NAN if pd.isna(ready) else hit_rate(after),
        "moved_after_scale": moved / considered if considered else NAN,
        "p50_latency_ms": ok["latency_ms"].quantile(0.50),
        "p95_latency_ms": ok["latency_ms"].quantile(0.95),
        "p95_prompt_ms": ok["prompt_ms"].quantile(0.95),
        "imbalance": work.max() / work.mean() if len(work) and work.mean() > 0 else NAN,
    }


def plot_hit_rate(runs: pd.DataFrame, policies: list, bin_s: float, out: Path) -> None:
    """One line per policy: all its runs pooled, per time bin."""
    follow = runs[(runs["status"] == 200) & (runs["turn"] > 1)].dropna(subset=["cache_n", "prompt_n"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for policy in policies:
        rows = follow[follow["policy"] == policy]
        sums = rows.groupby((rows["t_start_s"] // bin_s) * bin_s)[["cache_n", "prompt_n"]].sum()
        line, = ax.plot(sums.index, sums["cache_n"] / sums.sum(axis=1), marker="o", markersize=3, label=policy)
        ready = runs.loc[runs["policy"] == policy].groupby("run_id")["ready_s"].first().mean()
        if not pd.isna(ready):
            ax.axvline(ready, color=line.get_color(), linestyle=":", alpha=0.7)
    ax.set(xlabel="time since start (s)", ylabel="prompt tokens served from cache", ylim=(0, 1.05),
           title="Cache hit rate over time, all runs per policy (dotted: new replica ready, average)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, out: Path) -> None:
    panels = [("hit_rate", "Cache hit rate (follow-ups)", "#4C9A8A"),
              ("p95_latency_ms", "p95 latency (ms)", "#C77C3B"),
              ("imbalance", "Imbalance (busiest / average)", "#6A6FB5")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (column, title, color) in zip(axes, panels):
        summary[column].plot.bar(ax=ax, color=color, title=title)
        ax.set(xlabel="", ylabel="")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[2].axhline(1, color="gray", linewidth=0.8)  # 1.0 = perfectly even
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    defaults = SETTINGS["analysis"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path(defaults["results_dir"]))
    parser.add_argument("--out-dir", type=Path, default=Path(defaults["out_dir"]))
    parser.add_argument("--window-s", type=float, default=defaults["window_s"], help="window after the scale-up")
    parser.add_argument("--bin-s", type=float, default=defaults["bin_s"], help="chart time bin")
    args = parser.parse_args()

    runs = load(args.results_dir)
    per_run = pd.DataFrame([run_metrics(run, args.window_s) for _, run in runs.groupby("run_id")])
    policies = list(dict.fromkeys(per_run["policy"]))  # in the order they ran
    summary = per_run.drop(columns="run_id").groupby("policy").mean().reindex(policies)
    summary.insert(0, "runs", per_run.groupby("policy").size())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.out_dir / "per_run.csv", index=False)
    (args.out_dir / "summary.md").write_text(f"# Results\n\n{summary.round(3).to_markdown()}\n", encoding="utf-8")
    plot_summary(summary, args.out_dir / "summary.png")
    plot_hit_rate(runs, policies, args.bin_s, args.out_dir / "hit_rate_over_time.png")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.round(3))


if __name__ == "__main__":
    main()