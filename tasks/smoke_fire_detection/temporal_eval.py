import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Offline event-level analysis on FIgLib detector cache (no torch/ultralytics)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    g0 = subparsers.add_parser("g0", help="G0 gate: AUROC of detector confidence pre/post ignition")
    g0.add_argument("--cache", required=True, help="Path to figlib_detector_cache.jsonl")
    g0.add_argument("--out", default="artifacts/smoke_fire_detection/gate_g0_auroc.json")
    g0.add_argument("--ignore-band-seconds", type=float, default=180)
    g0.add_argument("--bootstrap-samples", type=int, default=1000)
    g0.add_argument("--seed", type=int, default=20260707)

    temporal = subparsers.add_parser("temporal", help="N-of-M and EMA event-level metrics (AMOC) from FIgLib detector cache")
    temporal.add_argument("--cache", required=True, help="Path to figlib_detector_cache.jsonl")
    temporal.add_argument("--out", required=True)
    temporal.add_argument("--events-out")
    temporal.add_argument("--score", choices=["any", "smoke", "fire"], default="any")
    temporal.add_argument("--thresholds", default="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8")
    temporal.add_argument("--nofm", default="2:3,3:5,5:10")
    temporal.add_argument("--ema-alphas", default="0.1,0.3,0.5")
    temporal.add_argument("--ema-init", choices=["score", "zero"], default="score")
    temporal.add_argument("--bootstrap-samples", type=int, default=1000)
    temporal.add_argument("--seed", type=int, default=20260707)

    return parser.parse_args()

def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]

def read_cache_flat(cache_path: Path):
    records = []
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records

def label_of(record, ignore_band_seconds):
    offset = record["ignition_offset_seconds"]
    if offset < 0:
        return 0
    if offset >= ignore_band_seconds:
        return 1
    return None

def auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    start = 0
    while start < len(sorted_scores):
        end = start
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        if end > start:
            ranks[order[start:end + 1]] = ranks[order[start:end + 1]].mean()
        start = end + 1
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def auroc_bootstrap_ci(groups, group_ids, score_key, samples, seed):
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        resample = [rng.choice(group_ids) for _ in group_ids]
        scores = []
        labels = []
        for group_id in resample:
            for record in groups[group_id]:
                scores.append(record[score_key])
                labels.append(record["_label"])
        value = auroc(scores, labels)
        if value is not None:
            values.append(value)
    return {
        "ci95": [percentile(values, 2.5), percentile(values, 97.5)],
        "n_resamples_used": len(values),
        "n_resamples_requested": samples,
    }

def gate_decision(auc):
    if auc is None:
        return "undecided", "Khong du frame pos/neg de tinh AUROC."
    if auc >= 0.80:
        return "pass", "Detector du tin hieu transfer sang FIgLib -> mo G1 (N-of-M/EMA temporal experiment)."
    if auc >= 0.60:
        return "marginal", "Thu imgsz lon hon hoac tiling truoc, do lai G0."
    return "fail", "D-Fire khong transfer sang khoi xa/nho tren FIgLib -> khong xay temporal tren detector mu; pivot sang tile-classifier hoac fine-tune tren PYRONEAR-2025."

def cmd_g0(args):
    cache_path = Path(args.cache).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    records = read_cache_flat(cache_path)
    total_frames = len(records)
    labeled = []
    for record in records:
        label = label_of(record, args.ignore_band_seconds)
        if label is None:
            continue
        record["_label"] = label
        labeled.append(record)
    dropped_by_band = total_frames - len(labeled)

    by_sequence = defaultdict(list)
    by_camera = defaultdict(list)
    for record in labeled:
        by_sequence[record["sequence_id"]].append(record)
        by_camera[record["camera_id"]].append(record)

    sequence_ids = sorted(by_sequence.keys())
    camera_ids = sorted(by_camera.keys())

    score_keys = {
        "smoke": "max_smoke_confidence",
        "fire": "max_fire_confidence",
        "any": "max_any_confidence",
    }
    auroc_by_score = {}
    for name, key in score_keys.items():
        scores = [record[key] for record in labeled]
        labels = [record["_label"] for record in labeled]
        auroc_by_score[name] = auroc(scores, labels)

    primary_auc = auroc_by_score["smoke"]
    decision, next_action = gate_decision(primary_auc)

    bootstrap_event = auroc_bootstrap_ci(by_sequence, sequence_ids, "max_smoke_confidence", args.bootstrap_samples, args.seed)
    bootstrap_camera = auroc_bootstrap_ci(by_camera, camera_ids, "max_smoke_confidence", args.bootstrap_samples, args.seed)

    n_pos = sum(1 for record in labeled if record["_label"] == 1)
    n_neg = sum(1 for record in labeled if record["_label"] == 0)

    result = {
        "cache": str(cache_path),
        "design_decisions": {
            "ignore_band_seconds": args.ignore_band_seconds,
            "label_rule": "offset < 0 -> negative (pre-ignition); offset >= ignore_band_seconds -> positive (post-ignition); [0, ignore_band_seconds) excluded",
            "primary_gate_metric": "max_smoke_confidence",
            "auroc_method": "mann_whitney_u_rank_sum (numpy, tie-corrected, no scipy/sklearn dependency)",
            "bootstrap_event_unit": "sequence_id",
            "bootstrap_camera_unit": "camera_id",
        },
        "counts": {
            "total_frames": total_frames,
            "frames_dropped_by_ignore_band": dropped_by_band,
            "frames_used": len(labeled),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_events": len(sequence_ids),
            "n_cameras": len(camera_ids),
        },
        "auroc": auroc_by_score,
        "bootstrap_event_smoke": bootstrap_event,
        "bootstrap_camera_smoke": bootstrap_camera,
        "gate": {
            "name": "G0",
            "auroc_smoke": primary_auc,
            "decision": decision,
            "next_action": next_action,
        },
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"G0 AUROC report saved: {out_path}")
    console.print(f"AUROC(smoke) = {primary_auc} -> gate decision: {decision}")

def read_cache_sequences(cache_path: Path):
    sequences = defaultdict(list)
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sequences[record["sequence_id"]].append(record)
    for frames in sequences.values():
        frames.sort(key=lambda item: item["ignition_offset_seconds"])
    return sequences

def score_of(record, score_type):
    if score_type == "smoke":
        return record["max_smoke_confidence"]
    if score_type == "fire":
        return record["max_fire_confidence"]
    return record["max_any_confidence"]

def nofm_alarm_offset(frames, threshold, n, m):
    hits = [1 if frame["_score"] >= threshold else 0 for frame in frames]
    window_sum = 0
    for index in range(len(hits)):
        window_sum += hits[index]
        if index >= m:
            window_sum -= hits[index - m]
        if index >= m - 1 and window_sum >= n:
            return frames[index]["ignition_offset_seconds"]
    return None

def ema_alarm_offset(frames, threshold, alpha, init_mode):
    ema = frames[0]["_score"] if init_mode == "score" else 0.0
    if ema >= threshold:
        return frames[0]["ignition_offset_seconds"]
    for index in range(1, len(frames)):
        ema = alpha * frames[index]["_score"] + (1 - alpha) * ema
        if ema >= threshold:
            return frames[index]["ignition_offset_seconds"]
    return None

def negative_window_seconds(frames):
    negative_frames = [frame for frame in frames if frame["ignition_offset_seconds"] < 0]
    if len(negative_frames) < 2:
        return 0.0
    timestamps = [frame["timestamp_unix"] for frame in negative_frames]
    return float(max(timestamps) - min(timestamps))

def evaluate_rule(sequence_ids, sequences, alarm_fn):
    false_alarm_events = 0
    true_alarm_events = 0
    negative_total_seconds = 0.0
    ttds = []
    per_event = []
    for sequence_id in sequence_ids:
        frames = sequences[sequence_id]
        negative_total_seconds += negative_window_seconds(frames)
        alarm_offset = alarm_fn(frames)
        is_false_alarm = alarm_offset is not None and alarm_offset < 0
        is_detected = alarm_offset is not None and alarm_offset >= 0
        if is_false_alarm:
            false_alarm_events += 1
        if is_detected:
            true_alarm_events += 1
            ttds.append(alarm_offset)
        per_event.append({
            "sequence_id": sequence_id,
            "camera_id": frames[0].get("camera_id"),
            "alarm_offset_seconds": alarm_offset,
            "is_false_alarm": is_false_alarm,
            "is_detected": is_detected,
        })
    total_events = len(sequence_ids)
    negative_hours = negative_total_seconds / 3600
    false_alarm_rate = (false_alarm_events / negative_hours) if negative_hours > 0 else None
    denom = true_alarm_events + false_alarm_events
    metrics = {
        "total_events": total_events,
        "false_alarm_events": false_alarm_events,
        "true_alarm_events": true_alarm_events,
        "negative_window_hours": negative_hours,
        "false_alarm_rate_per_hour": false_alarm_rate,
        "event_precision": (true_alarm_events / denom) if denom > 0 else None,
        "event_recall": (true_alarm_events / total_events) if total_events > 0 else None,
        "ttd_mean_seconds": statistics.mean(ttds) if ttds else None,
        "ttd_median_seconds": statistics.median(ttds) if ttds else None,
    }
    return metrics, per_event

def temporal_bootstrap_ci(sequence_ids, sequences, alarm_fn, samples, seed):
    if samples <= 0:
        return {}
    rng = random.Random(seed)
    fa_rates = []
    ttd_means = []
    for _ in range(samples):
        resample = [rng.choice(sequence_ids) for _ in sequence_ids]
        metrics, _ = evaluate_rule(resample, sequences, alarm_fn)
        if metrics["false_alarm_rate_per_hour"] is not None:
            fa_rates.append(metrics["false_alarm_rate_per_hour"])
        if metrics["ttd_mean_seconds"] is not None:
            ttd_means.append(metrics["ttd_mean_seconds"])
    return {
        "false_alarm_rate_per_hour_ci95": [percentile(fa_rates, 2.5), percentile(fa_rates, 97.5)],
        "ttd_mean_seconds_ci95": [percentile(ttd_means, 2.5), percentile(ttd_means, 97.5)],
    }

def cmd_temporal(args):
    cache_path = Path(args.cache).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    sequences = read_cache_sequences(cache_path)
    for frames in sequences.values():
        for frame in frames:
            frame["_score"] = score_of(frame, args.score)
    sequence_ids = sorted(sequences.keys())

    thresholds = [float(value) for value in args.thresholds.split(",") if value]
    nofm_pairs = []
    for pair in args.nofm.split(","):
        if not pair:
            continue
        n_value, m_value = pair.split(":")
        nofm_pairs.append((int(n_value), int(m_value)))
    ema_alphas = [float(value) for value in args.ema_alphas.split(",") if value]

    rows = []
    for threshold in thresholds:
        for n, m in nofm_pairs:
            def alarm_fn(frames, threshold=threshold, n=n, m=m):
                return nofm_alarm_offset(frames, threshold, n, m)
            metrics, per_event = evaluate_rule(sequence_ids, sequences, alarm_fn)
            ci = temporal_bootstrap_ci(sequence_ids, sequences, alarm_fn, args.bootstrap_samples, args.seed)
            rows.append({"rule": "n_of_m", "threshold": threshold, "n": n, "m": m, **metrics, **ci, "events": per_event})
        for alpha in ema_alphas:
            def alarm_fn(frames, threshold=threshold, alpha=alpha):
                return ema_alarm_offset(frames, threshold, alpha, args.ema_init)
            metrics, per_event = evaluate_rule(sequence_ids, sequences, alarm_fn)
            ci = temporal_bootstrap_ci(sequence_ids, sequences, alarm_fn, args.bootstrap_samples, args.seed)
            rows.append({"rule": "ema", "threshold": threshold, "alpha": alpha, "ema_init": args.ema_init, **metrics, **ci, "events": per_event})

    if args.events_out:
        events_path = Path(args.events_out).resolve()
        events_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for row in rows:
            for event in row["events"]:
                lines.append(json.dumps({**event, "rule": row["rule"], "threshold": row["threshold"]}, ensure_ascii=False))
        events_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        console.print(f"Per-event detail saved: {events_path}")

    for row in rows:
        row.pop("events", None)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "cache": str(cache_path),
        "score": args.score,
        "sequences": len(sequence_ids),
        "bootstrap_samples": args.bootstrap_samples,
        "results": rows,
    }, indent=2), encoding="utf-8")
    console.print(f"Temporal eval saved: {out_path}")

def main():
    args = parse_args()
    if args.command == "g0":
        cmd_g0(args)
    elif args.command == "temporal":
        cmd_temporal(args)

if __name__ == "__main__":
    main()
