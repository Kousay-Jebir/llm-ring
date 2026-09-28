"""Summarize experiment runs into a table and charts. Defaults come from settings.toml.

Reads, for each run, <run_id>.csv (from traffic/generator.py; columns used: status, turn,
conversation_id, replica, t_start_s, latency_ms, cache_n, prompt_n, prompt_ms, predicted_n)
and <run_id>.json (from experiments/run.py; fields used: run_id, policy, new_replica_ready_s).

Metrics per run, averaged per policy when a policy has several runs:
  hit_rate           share of prompt tokens served from cache, follow-up turns only
  hit_rate_after     same, in the window right after the new replica became ready
  moved_after_scale  share of conversations whose turns straddling the scale-up changed replica
  p50/p95 latency    end-to-end latency seen by the client
  p95_prompt_ms      time spent (re)processing prompts: what caching saves
  imbalance          busiest replica's work / average work, before the scale-up
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from traffic.generator import SETTINGS  # noqa: E402

def load_runs(results_dir: Path) -> list[tuple[dict, pd.DataFrame]]:
    runs = []
    for meta_path in sorted(results_dir.glob("*.json")):
        if meta_path.with_suffix(".csv").exists():
            frame = pd.read_csv(meta_path.with_suffix(".csv"))
            frame["work"] = frame["prompt_n"] + frame["predicted_n"].fillna(0)  # tokens the replica computed
            runs.append((json.loads(meta_path.read_text(encoding="utf-8")), frame))
    return runs


def hit_rate(frame: pd.DataFrame) -> float:
    total = (frame["cache_n"] + frame["prompt_n"]).sum()
    return float(frame["cache_n"].sum() / total) if total else float("nan")


def run_metrics(meta: dict, frame: pd.DataFrame, window_s: float) -> dict:
    ok = frame[frame["status"] == 200].dropna(subset=["cache_n", "prompt_n"])
    follow = ok[ok["turn"] > 1]
    ready = meta.get("new_replica_ready_s")
    nan = float("nan")
    quantile = lambda col, q: float(ok[col].quantile(q)) if len(ok) else nan  # noqa: E731
    metrics = {"run_id": meta["run_id"], "policy": meta["policy"], "requests": len(frame),
               "errors": int((frame["status"] != 200).sum()), "hit_rate": hit_rate(follow),
               "p50_latency_ms": quantile("latency_ms", 0.50), "p95_latency_ms": quantile("latency_ms", 0.95),
               "p95_prompt_ms": quantile("prompt_ms", 0.95), "hit_rate_after": nan, "moved_after_scale": nan,
               "imbalance": nan}

    before = ok if ready is None else ok[ok["t_start_s"] < ready]
    work = before.groupby("replica")["work"].sum()
    if len(work) and work.mean() > 0:
        metrics["imbalance"] = float(work.max() / work.mean())

    if ready is not None:
        metrics["hit_rate_after"] = hit_rate(follow[(follow["t_start_s"] >= ready) & (follow["t_start_s"] < ready + window_s)])
        moved = considered = 0
        for _, turns in ok.sort_values("turn").groupby("conversation_id"):
            rows = turns.to_dict("records")
            for prev, cur in zip(rows, rows[1:]):
                if prev["t_start_s"] < ready <= cur["t_start_s"]:  # first pair straddling the scale-up
                    considered += 1
                    moved += prev["replica"] != cur["replica"]
                    break
        if considered:
            metrics["moved_after_scale"] = moved / considered
    return metrics


def binned(frame: pd.DataFrame, bin_s: float) -> pd.Series:
    return (frame["t_start_s"] // bin_s) * bin_s  # each point sits at the start of its bin


def plot_hit_rate(runs, out: Path, bin_s: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for meta, frame in runs:
        ok = frame[(frame["status"] == 200) & (frame["turn"] > 1)].dropna(subset=["cache_n", "prompt_n"])
        if ok.empty:
            continue
        sums = ok.groupby(binned(ok, bin_s))[["cache_n", "prompt_n"]].sum()
        rate = sums["cache_n"] / (sums["cache_n"] + sums["prompt_n"])
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
        table = ok.assign(bin=binned(ok, bin_s)).pivot_table(index="bin", columns="replica", values="work",
                                                             aggfunc="sum", fill_value=0)
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
    summary["hit_rate"].plot.bar(ax=left, color="#4C9A8A", title="Cache hit rate (follow-up turns)", ylim=(0, 1))
    summary["p95_latency_ms"].plot.bar(ax=right, color="#C77C3B", title="p95 latency (ms)")
    for ax in (left, right):
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summarize(results_dir: Path, out_dir: Path, window_s: float, bin_s: float) -> pd.DataFrame:
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"no runs found in {results_dir} (expected matching .json and .csv files)")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run = pd.DataFrame([run_metrics(meta, frame, window_s) for meta, frame in runs])
    order = list(dict.fromkeys(per_run["policy"]))  # the order in which policies ran
    columns = ["requests", "errors", "hit_rate", "hit_rate_after", "moved_after_scale",
               "p50_latency_ms", "p95_latency_ms", "p95_prompt_ms", "imbalance"]
    summary = per_run.groupby("policy")[columns].mean().reindex(order)
    summary.insert(0, "runs", per_run.groupby("policy").size().reindex(order))

    per_run.to_csv(out_dir / "per_run.csv", index=False)
    (out_dir / "summary.md").write_text(f"# Results\n\n{summary.round(3).to_markdown()}\n", encoding="utf-8")
    plot_hit_rate(runs, out_dir / "hit_rate_over_time.png", bin_s)
    plot_work_per_replica(runs, out_dir / "work_per_replica.png", bin_s)
    plot_summary(summary, out_dir / "summary.png")
    return summary


def main() -> None:
    defaults = SETTINGS["analysis"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=defaults["results_dir"])
    parser.add_argument("--out-dir", default=defaults["out_dir"])
    parser.add_argument("--window-s", type=float, default=defaults["window_s"], help="window after the scale-up")
    parser.add_argument("--bin-s", type=float, default=defaults["bin_s"], help="time bin for charts")
    args = parser.parse_args()
    summary = summarize(Path(args.results_dir), Path(args.out_dir), args.window_s, args.bin_s)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.round(3))


if __name__ == "__main__":
    main()
