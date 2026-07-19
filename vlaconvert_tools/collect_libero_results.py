#!/usr/bin/env python3
"""Collect vlaconvert LIBERO logs and build accuracy/latency tables.

This intentionally copies only files produced by the QuantVLA-converted
LIBERO path, plus the matching eval latency files for the requested suite.
Run it right after each experiment because eval logs are named by suite and
can be overwritten by the next fake/real run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_METRICS = (
    "step_total_ms",
    "client_roundtrip_ms",
    "policy_total_ms",
    "policy_model_get_action_ms",
    "policy_apply_transforms_ms",
    "env_step_ms",
)
STAT_KEYS = ("mean", "median", "p90", "p99", "min", "max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect vlaconvert LIBERO logs and summarize accuracy/latency."
    )
    parser.add_argument("--log-dir", default="/tmp/logs", help="Source log directory.")
    parser.add_argument("--output-dir", default="results", help="Destination results root.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Result folder name. Defaults to YYYYmmdd_HHMMSS_<mode>.",
    )
    parser.add_argument(
        "--mode",
        choices=("real", "fake"),
        required=True,
        help="Experiment mode for the eval logs being collected.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=list(DEFAULT_SUITES),
        help="LIBERO suites to collect.",
    )
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Also copy rollout mp4 files. Off by default because videos are large.",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Only rebuild summary tables from output-dir.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def collect_one_suite(
    log_dir: Path,
    run_dir: Path,
    mode: str,
    suite: str,
    copy_videos: bool,
) -> dict[str, Any]:
    suite_dir = run_dir / mode / suite
    suite_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []

    names = [
        f"libero_eval_{suite}.log",
        f"libero_eval_{suite}_latency_summary.json",
        f"libero_eval_{suite}_latency_steps.csv",
        f"libero_eval_{suite}_latency_steps.jsonl",
        f"quantvla_converted_{mode}_{suite}_replacement_report.json",
    ]

    for name in names:
        src = log_dir / name
        if copy_if_exists(src, suite_dir / name):
            copied.append(name)
        else:
            missing.append(name)

    if copy_videos:
        video_dir = suite_dir / "videos"
        for src in sorted(log_dir.glob("rollout_*.mp4")):
            if copy_if_exists(src, video_dir / src.name):
                copied.append(f"videos/{src.name}")

    manifest = {
        "mode": mode,
        "suite": suite,
        "source_log_dir": str(log_dir),
        "destination": str(suite_dir),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "copied": copied,
        "missing": missing,
    }
    with (suite_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def get_nested(data: dict[str, Any], *keys: str) -> Any:
    curr: Any = data
    for key in keys:
        if not isinstance(curr, dict) or key not in curr:
            return None
        curr = curr[key]
    return curr


def parse_log_success(log_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not log_path.exists():
        return result
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"# successes:\s+(\d+)\s+\(([\d.]+)%\)", text)
    if matches:
        successes, pct = matches[-1]
        result["log_num_successes"] = int(successes)
        result["log_success_rate_pct"] = float(pct)
    matches = re.findall(r"# episodes completed so far:\s+(\d+)", text)
    if matches:
        result["log_num_episodes"] = int(matches[-1])
    matches = re.findall(r"Current total success rate:\s+([\d.]+)", text)
    if matches:
        result["log_success_rate"] = float(matches[-1])
    return result


def summarize_suite_dir(suite_dir: Path, mode: str, suite: str) -> dict[str, Any]:
    latency = read_json(suite_dir / f"libero_eval_{suite}_latency_summary.json") or {}
    report = read_json(suite_dir / f"quantvla_converted_{mode}_{suite}_replacement_report.json") or {}
    log_values = parse_log_success(suite_dir / f"libero_eval_{suite}.log")

    row: dict[str, Any] = {
        "run": suite_dir.parents[1].name,
        "mode": mode,
        "suite": suite,
        "result_dir": str(suite_dir),
    }

    row["num_episodes"] = latency.get("num_episodes") or log_values.get("log_num_episodes")
    row["num_successes"] = latency.get("num_successes") or log_values.get("log_num_successes")
    success_rate = latency.get("success_rate")
    if success_rate is None:
        success_rate = log_values.get("log_success_rate")
    row["success_rate_pct"] = round(float(success_rate) * 100.0, 3) if success_rate is not None else None

    row["num_action_steps"] = get_nested(latency, "overall", "num_action_steps")
    row["target_linear_layers"] = report.get("target_linear_layers")
    row["successfully_replaced"] = report.get("successfully_replaced")
    row["unmatched_checkpoint_keys_count"] = report.get("unmatched_checkpoint_keys_count")
    row["unreplaced_target_layers_count"] = report.get("unreplaced_target_layers_count")
    row["fallback_fp16_layers_count"] = report.get("fallback_fp16_layers_count")

    for metric in DEFAULT_METRICS:
        for stat in STAT_KEYS:
            value = get_nested(latency, "overall", metric, stat)
            row[f"{metric}_{stat}"] = round(float(value), 4) if value is not None else None

    return row


def discover_result_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        for mode_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            mode = mode_dir.name
            if mode not in {"real", "fake"}:
                continue
            for suite_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
                rows.append(summarize_suite_dir(suite_dir, mode, suite_dir.name))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "run",
        "mode",
        "suite",
        "success_rate_pct",
        "num_successes",
        "num_episodes",
        "num_action_steps",
        "step_total_ms_mean",
        "step_total_ms_p90",
        "step_total_ms_p99",
        "client_roundtrip_ms_mean",
        "policy_model_get_action_ms_mean",
        "policy_total_ms_mean",
        "env_step_ms_mean",
        "successfully_replaced",
        "fallback_fp16_layers_count",
        "result_dir",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "run",
        "mode",
        "suite",
        "success_rate_pct",
        "num_successes",
        "num_episodes",
        "num_action_steps",
        "step_total_ms_mean",
        "step_total_ms_p90",
        "step_total_ms_p99",
        "client_roundtrip_ms_mean",
        "policy_model_get_action_ms_mean",
        "policy_total_ms_mean",
        "env_step_ms_mean",
        "successfully_replaced",
        "fallback_fp16_layers_count",
    ]
    lines = [
        "# LIBERO QuantVLA-converted Results",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = ["" if row.get(col) is None else str(row.get(col)) for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Latency Field Guide",
            "",
            "- `step_total_ms_*`: end-to-end one action step in LIBERO, including observation preprocessing, policy request, and env step.",
            "- `client_roundtrip_ms_*`: client-side request/response time to the inference server.",
            "- `policy_total_ms_*`: server-side policy timing if `GR00T_TIMING=1` was enabled.",
            "- `policy_model_get_action_ms_*`: model `get_action` time inside the policy path.",
            "- `env_step_ms_*`: simulator/environment step time after action is returned.",
            "- `*_mean`, `*_p90`, `*_p99`: use mean for typical runtime and p90/p99 for tail latency.",
            "",
            "For detailed outlier analysis, inspect each suite's `libero_eval_<suite>_latency_steps.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_summary(output_dir: Path) -> None:
    rows = discover_result_rows(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    write_csv(output_dir / "summary.csv", rows)
    (output_dir / "summary.md").write_text(markdown_table(rows), encoding="utf-8")


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.mode}"
    run_dir = output_dir / run_name

    manifests = []
    if not args.no_copy:
        for suite in args.suites:
            manifests.append(
                collect_one_suite(
                    log_dir=log_dir,
                    run_dir=run_dir,
                    mode=args.mode,
                    suite=suite,
                    copy_videos=args.copy_videos,
                )
            )
        with (run_dir / "collection_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifests, f, indent=2)

    write_summary(output_dir)
    print(f"Wrote results under: {output_dir}")
    print(f"Summary CSV: {output_dir / 'summary.csv'}")
    print(f"Summary Markdown: {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
