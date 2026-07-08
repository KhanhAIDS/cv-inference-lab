import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from rich.console import Console

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="N-of-M and EMA event-level metrics from FIgLib detector cache")
    parser.add_argument("--cache", required=True, help="Path to figlib_detector_cache.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--events-out")
    parser.add_argument("--score", choices=["any", "smoke", "fire"], default="any")
    parser.add_argument("--thresholds", default="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--nofm", default="2:3,3:5,5:10")
    parser.add_argument("--ema-alphas", default="0.1,0.3,0.5")
    parser.add_argument("--ema-init", choices=["score", "zero"], default="score")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()

def read_cache(cache_path: Path):
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

def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]

def bootstrap_ci(sequence_ids, sequences, alarm_fn, samples, seed):
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

def main():
    args = parse_args()
    cache_path = Path(args.cache).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    sequences = read_cache(cache_path)
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
            ci = bootstrap_ci(sequence_ids, sequences, alarm_fn, args.bootstrap_samples, args.seed)
            rows.append({"rule": "n_of_m", "threshold": threshold, "n": n, "m": m, **metrics, **ci, "events": per_event})
        for alpha in ema_alphas:
            def alarm_fn(frames, threshold=threshold, alpha=alpha):
                return ema_alarm_offset(frames, threshold, alpha, args.ema_init)
            metrics, per_event = evaluate_rule(sequence_ids, sequences, alarm_fn)
            ci = bootstrap_ci(sequence_ids, sequences, alarm_fn, args.bootstrap_samples, args.seed)
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

if __name__ == "__main__":
    main()
