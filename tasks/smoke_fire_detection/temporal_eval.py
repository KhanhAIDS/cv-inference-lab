import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
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

    diagnose = subparsers.add_parser("diagnose", help="Rung 0: zero-detection rate, offset-band AUROC, per-sequence AUROC spread (CPU-only, no new GPU run)")
    diagnose.add_argument("--cache", required=True, help="Path to figlib_detector_cache.jsonl")
    diagnose.add_argument("--out", default="artifacts/smoke_fire_detection/diagnose_g0.json")
    diagnose.add_argument("--score", choices=["any", "smoke", "fire"], default="smoke")
    diagnose.add_argument("--ignore-band-seconds", type=float, default=180)
    diagnose.add_argument("--positive-bands", default="180:600,600:1200,1200:100000", help="Comma-separated start:end offset seconds, open-ended via a large end")
    diagnose.add_argument("--low-auroc-threshold", type=float, default=0.6)
    diagnose.add_argument("--silent-confidence-threshold", type=float, default=0.3)

    detection_sizes = subparsers.add_parser("detection-sizes", help="Normalized box-area distribution of confident post-ignition detections, to gauge whether tiling still has headroom")
    detection_sizes.add_argument("--cache", required=True, help="Path to figlib_detector_cache.jsonl")
    detection_sizes.add_argument("--out", default="artifacts/smoke_fire_detection/detection_sizes.json")
    detection_sizes.add_argument("--score", choices=["smoke", "fire"], default="smoke")
    detection_sizes.add_argument("--confidence-threshold", type=float, default=0.3)
    detection_sizes.add_argument("--ignore-band-seconds", type=float, default=180)

    refine = subparsers.add_parser("refine-tile-agreement", help="Recompute a tiled detector cache keeping only detections confirmed by >=2 overlapping tiles (post-process, no GPU rerun)")
    refine.add_argument("--cache", required=True, help="Path to a tiled figlib_detector_cache*.jsonl (must contain per-tile detections)")
    refine.add_argument("--out", required=True)
    refine.add_argument("--iou-threshold", type=float, default=0.1)

    persistence = subparsers.add_parser("spatial-persistence", help="Recompute confidence as window-averaged confidence of spatially-matching detections across nearby frames (post-process, no GPU rerun)")
    persistence.add_argument("--cache", required=True, help="Path to figlib_detector_cache.jsonl")
    persistence.add_argument("--out", required=True)
    persistence.add_argument("--window-frames", type=int, default=5)
    persistence.add_argument("--distance-factor", type=float, default=3.0, help="Match radius = distance_factor * max(bbox diag of the two detections)")
    persistence.add_argument("--sequence-ids", help="Comma-separated sequence_id subset, for cheap verification before a full run")

    differencing = subparsers.add_parser("frame-differencing", help="Motion-energy score from a rolling grayscale-median background (no detector, no GPU) -- probe whether silent sequences have any usable signal at all")
    differencing.add_argument("--cache", required=True, help="Any figlib_detector_cache.jsonl, used only for sequence_id/frame_path/offset listing")
    differencing.add_argument("--out", required=True)
    differencing.add_argument("--background-window", type=int, default=5)
    differencing.add_argument("--thumbnail-size", type=int, default=256)
    differencing.add_argument("--anomaly-fraction", type=float, default=0.05, help="Fraction of highest-diff pixels averaged into the score, to catch localized motion instead of global illumination drift")
    differencing.add_argument("--sequence-ids", help="Comma-separated sequence_id subset")

    compare = subparsers.add_parser("compare-candidates", help="Paired AUROC comparison for two aligned detector caches")
    compare.add_argument("--baseline-cache", required=True)
    compare.add_argument("--candidate-cache", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--ignore-band-seconds", type=float, default=180)
    compare.add_argument("--bootstrap-samples", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=20260707)

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

def cache_key(record):
    return record["sequence_id"], int(record["timestamp_unix"]), int(record["ignition_offset_seconds"])

def paired_auroc_bootstrap(groups, group_ids, base_key, candidate_key, samples, seed):
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        scores_base = []
        scores_candidate = []
        labels = []
        for group_id in (rng.choice(group_ids) for _ in group_ids):
            for record in groups[group_id]:
                scores_base.append(record[base_key])
                scores_candidate.append(record[candidate_key])
                labels.append(record["_label"])
        base_value = auroc(scores_base, labels)
        candidate_value = auroc(scores_candidate, labels)
        if base_value is not None and candidate_value is not None:
            deltas.append(candidate_value - base_value)
    return {
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))] if deltas else [None, None],
        "n_resamples_used": len(deltas),
        "n_resamples_requested": samples,
    }

def cmd_compare_candidates(args):
    baseline = {cache_key(record): record for record in read_cache_flat(Path(args.baseline_cache))}
    candidate = {cache_key(record): record for record in read_cache_flat(Path(args.candidate_cache))}
    if set(baseline) != set(candidate):
        missing_baseline = sorted(set(candidate) - set(baseline))
        missing_candidate = sorted(set(baseline) - set(candidate))
        raise ValueError(json.dumps({"alignment_mismatch": {
            "missing_baseline": missing_baseline[:10],
            "missing_candidate": missing_candidate[:10],
            "missing_baseline_count": len(missing_baseline),
            "missing_candidate_count": len(missing_candidate),
        }}))
    paired = []
    for key in sorted(baseline):
        base_record = baseline[key]
        candidate_record = candidate[key]
        label = label_of(base_record, args.ignore_band_seconds)
        if label is None:
            continue
        paired.append({
            "sequence_id": base_record["sequence_id"],
            "camera_id": base_record["camera_id"],
            "_label": label,
            "base_smoke": base_record["max_smoke_confidence"],
            "candidate_smoke": candidate_record["max_smoke_confidence"],
        })
    groups_event = defaultdict(list)
    groups_camera = defaultdict(list)
    for record in paired:
        groups_event[record["sequence_id"]].append(record)
        groups_camera[record["camera_id"]].append(record)
    baseline_scores = [record["base_smoke"] for record in paired]
    candidate_scores = [record["candidate_smoke"] for record in paired]
    labels = [record["_label"] for record in paired]
    baseline_auroc = auroc(baseline_scores, labels)
    candidate_auroc = auroc(candidate_scores, labels)
    event_bootstrap = paired_auroc_bootstrap(
        groups_event,
        sorted(groups_event),
        "base_smoke",
        "candidate_smoke",
        args.bootstrap_samples,
        args.seed,
    )
    camera_bootstrap = paired_auroc_bootstrap(
        groups_camera,
        sorted(groups_camera),
        "base_smoke",
        "candidate_smoke",
        args.bootstrap_samples,
        args.seed + 1,
    )
    report = {
        "baseline_cache": str(Path(args.baseline_cache).resolve()),
        "candidate_cache": str(Path(args.candidate_cache).resolve()),
        "ignore_band_seconds": args.ignore_band_seconds,
        "paired_frames": len(paired),
        "paired_events": len(groups_event),
        "paired_cameras": len(groups_camera),
        "baseline_auroc_smoke": baseline_auroc,
        "candidate_auroc_smoke": candidate_auroc,
        "delta_auroc_smoke": candidate_auroc - baseline_auroc,
        "paired_bootstrap_event": event_bootstrap,
        "paired_bootstrap_camera": camera_bootstrap,
        "winner_rule": "event and camera lower CI > 0",
        "decision": "candidate_better" if event_bootstrap["ci95"][0] > 0 and camera_bootstrap["ci95"][0] > 0 else "tie_or_not_proven",
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"Candidate comparison saved: {out_path}")

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

def parse_bands(spec):
    bands = []
    for part in spec.split(","):
        if not part:
            continue
        start, end = part.split(":")
        bands.append((float(start), float(end)))
    return bands

def cmd_diagnose(args):
    cache_path = Path(args.cache).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    score_key = {
        "smoke": "max_smoke_confidence",
        "fire": "max_fire_confidence",
        "any": "max_any_confidence",
    }[args.score]

    records = read_cache_flat(cache_path)
    labeled = []
    for record in records:
        label = label_of(record, args.ignore_band_seconds)
        if label is None:
            continue
        record["_label"] = label
        labeled.append(record)

    positives = [r for r in labeled if r["_label"] == 1]
    negatives = [r for r in labeled if r["_label"] == 0]
    neg_scores = [r[score_key] for r in negatives]
    neg_labels = [0 for _ in negatives]

    zero_rate_positive = sum(1 for r in positives if r[score_key] == 0.0) / len(positives) if positives else None
    zero_rate_negative = sum(1 for r in negatives if r[score_key] == 0.0) / len(negatives) if negatives else None

    bands = parse_bands(args.positive_bands)
    band_report = []
    for start, end in bands:
        band_positives = [r for r in positives if start <= r["ignition_offset_seconds"] < end]
        band_zero_rate = sum(1 for r in band_positives if r[score_key] == 0.0) / len(band_positives) if band_positives else None
        band_auc = auroc(neg_scores + [r[score_key] for r in band_positives], neg_labels + [1 for _ in band_positives]) if band_positives else None
        band_report.append({
            "offset_start_seconds": start,
            "offset_end_seconds": end,
            "n_positive_frames": len(band_positives),
            "zero_detection_rate": band_zero_rate,
            "auroc_vs_all_negative": band_auc,
        })

    by_sequence = defaultdict(list)
    for record in labeled:
        by_sequence[record["sequence_id"]].append(record)

    per_sequence_auroc = []
    for sequence_id, seq_records in by_sequence.items():
        seq_pos = [r for r in seq_records if r["_label"] == 1]
        seq_neg = [r for r in seq_records if r["_label"] == 0]
        if not seq_pos or not seq_neg:
            continue
        value = auroc(
            [r[score_key] for r in seq_neg] + [r[score_key] for r in seq_pos],
            [0 for _ in seq_neg] + [1 for _ in seq_pos],
        )
        if value is not None:
            per_sequence_auroc.append({
                "sequence_id": sequence_id,
                "auroc": value,
                "n_positive": len(seq_pos),
                "n_negative": len(seq_neg),
                "max_positive_confidence": max(r[score_key] for r in seq_pos),
                "max_negative_confidence": max(r[score_key] for r in seq_neg),
            })

    auroc_values = [item["auroc"] for item in per_sequence_auroc]
    low_auroc = [item for item in per_sequence_auroc if item["auroc"] < args.low_auroc_threshold]
    silent = [item for item in low_auroc if item["max_positive_confidence"] < args.silent_confidence_threshold and item["max_negative_confidence"] < args.silent_confidence_threshold]
    confused = [item for item in low_auroc if item["max_negative_confidence"] >= args.silent_confidence_threshold]
    other = [item for item in low_auroc if item not in silent and item not in confused]
    buckets = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)]
    histogram = [{
        "range": f"[{lo},{hi})" if hi != 1.0001 else f"[{lo},1.0]",
        "count": sum(1 for value in auroc_values if lo <= value < hi),
    } for lo, hi in buckets]

    per_sequence_auroc.sort(key=lambda item: item["auroc"])
    worst = per_sequence_auroc[:10]
    best = per_sequence_auroc[-10:]

    result = {
        "cache": str(cache_path),
        "score": args.score,
        "ignore_band_seconds": args.ignore_band_seconds,
        "n_positive_frames": len(positives),
        "n_negative_frames": len(negatives),
        "zero_detection_rate_positive": zero_rate_positive,
        "zero_detection_rate_negative": zero_rate_negative,
        "offset_band_auroc": band_report,
        "per_sequence_auroc_histogram": histogram,
        "per_sequence_auroc_percentiles": {
            "p10": percentile(auroc_values, 10),
            "p25": percentile(auroc_values, 25),
            "p50": percentile(auroc_values, 50),
            "p75": percentile(auroc_values, 75),
            "p90": percentile(auroc_values, 90),
        },
        "worst_sequences": worst,
        "best_sequences": best,
        "low_auroc_breakdown": {
            "low_auroc_threshold": args.low_auroc_threshold,
            "silent_confidence_threshold": args.silent_confidence_threshold,
            "n_low_auroc_sequences": len(low_auroc),
            "n_silent": len(silent),
            "n_confused": len(confused),
            "n_other": len(other),
            "silent_meaning": "max confidence stays below threshold on both sides of ignition -> camera likely has no visible plume in this window at all, not a detector failure per se",
            "confused_meaning": "pre-ignition confidence reaches threshold -> real false-trigger (haze/glare/other), detector is actively wrong not just blind",
            "silent_sequence_ids": [item["sequence_id"] for item in silent],
            "confused_sequence_ids": [item["sequence_id"] for item in confused],
        },
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"Diagnose report saved: {out_path}")
    console.print(f"zero_detection_rate_positive={zero_rate_positive} zero_detection_rate_negative={zero_rate_negative}")
    for band in band_report:
        console.print(f"band [{band['offset_start_seconds']},{band['offset_end_seconds']}) n={band['n_positive_frames']} auroc={band['auroc_vs_all_negative']} zero_rate={band['zero_detection_rate']}")
    console.print(f"low-AUROC(<{args.low_auroc_threshold}) sequences: {len(low_auroc)} -> silent={len(silent)} confused={len(confused)} other={len(other)}")

def cmd_detection_sizes(args):
    cache_path = Path(args.cache).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    areas = []
    samples = []
    for record in read_cache_flat(cache_path):
        if record["ignition_offset_seconds"] < args.ignore_band_seconds:
            continue
        detections = [d for d in record["detections"] if d["class_name"] == args.score]
        if not detections:
            continue
        best = max(detections, key=lambda d: d["confidence"])
        if best["confidence"] < args.confidence_threshold:
            continue
        with Image.open(record["frame_path"]) as img:
            width, height = img.size
        x1, y1, x2, y2 = best["xyxy"]
        area = ((x2 - x1) * (y2 - y1)) / (width * height)
        areas.append(area)
        samples.append({
            "sequence_id": record["sequence_id"],
            "frame_path": record["frame_path"],
            "confidence": best["confidence"],
            "normalized_area": area,
            "image_size": [width, height],
        })

    samples.sort(key=lambda item: item["normalized_area"])
    result = {
        "cache": str(cache_path),
        "score": args.score,
        "confidence_threshold": args.confidence_threshold,
        "ignore_band_seconds": args.ignore_band_seconds,
        "n_qualifying_detections": len(areas),
        "normalized_area_percentiles": {f"p{p}": percentile(areas, p) for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]},
        "smallest_samples": samples[:15],
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"Detection-sizes report saved: {out_path}")
    console.print(f"n={len(areas)} percentiles={result['normalized_area_percentiles']}")

def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

def tile_confirmed_detections(detections, iou_threshold):
    confirmed = []
    for i, detection in enumerate(detections):
        for j, other in enumerate(detections):
            if i == j:
                continue
            if detection["class_id"] == other["class_id"] and box_iou(detection["xyxy"], other["xyxy"]) >= iou_threshold:
                confirmed.append(detection)
                break
    return confirmed

def cmd_refine_tile_agreement(args):
    cache_path = Path(args.cache).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as out_file:
        for record in read_cache_flat(cache_path):
            confirmed = tile_confirmed_detections(record["detections"], args.iou_threshold)
            smoke_confidences = [d["confidence"] for d in confirmed if d["class_id"] == 0]
            fire_confidences = [d["confidence"] for d in confirmed if d["class_id"] == 1]
            any_confidences = [d["confidence"] for d in confirmed]
            record["detections"] = confirmed
            record["max_smoke_confidence"] = max(smoke_confidences) if smoke_confidences else 0.0
            record["max_fire_confidence"] = max(fire_confidences) if fire_confidences else 0.0
            record["max_any_confidence"] = max(any_confidences) if any_confidences else 0.0
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    console.print(f"Refined cache saved: {out_path} ({written} frames, iou_threshold={args.iou_threshold})")

def bbox_center_and_diag(xyxy):
    x1, y1, x2, y2 = xyxy
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return center, diag

def best_match_confidence(center, radius, detections, class_id):
    best = 0.0
    for detection in detections:
        if detection["class_id"] != class_id:
            continue
        other_center, _ = bbox_center_and_diag(detection["xyxy"])
        dist = ((other_center[0] - center[0]) ** 2 + (other_center[1] - center[1]) ** 2) ** 0.5
        if dist <= radius and detection["confidence"] > best:
            best = detection["confidence"]
    return best

def spatial_persistence_scores(frames, window_frames, distance_factor, class_id):
    scores = [0.0] * len(frames)
    for i, frame in enumerate(frames):
        best_score = 0.0
        for detection in frame["detections"]:
            if detection["class_id"] != class_id:
                continue
            center, diag = bbox_center_and_diag(detection["xyxy"])
            radius = distance_factor * max(diag, 1e-6)
            accumulated = detection["confidence"]
            for j in range(max(0, i - window_frames), i):
                accumulated += best_match_confidence(center, radius, frames[j]["detections"], class_id)
            score = accumulated / (window_frames + 1)
            if score > best_score:
                best_score = score
        scores[i] = best_score
    return scores

def cmd_spatial_persistence(args):
    cache_path = Path(args.cache).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    sequences = read_cache_sequences(cache_path)
    if args.sequence_ids:
        wanted = set(args.sequence_ids.split(","))
        sequences = {sid: frames for sid, frames in sequences.items() if sid in wanted}

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as out_file:
        for frames in sequences.values():
            smoke_scores = spatial_persistence_scores(frames, args.window_frames, args.distance_factor, class_id=0)
            fire_scores = spatial_persistence_scores(frames, args.window_frames, args.distance_factor, class_id=1)
            for frame, smoke_score, fire_score in zip(frames, smoke_scores, fire_scores):
                frame["max_smoke_confidence"] = smoke_score
                frame["max_fire_confidence"] = fire_score
                frame["max_any_confidence"] = max(smoke_score, fire_score)
                out_file.write(json.dumps(frame, ensure_ascii=False) + "\n")
                written += 1

    console.print(f"Spatial-persistence cache saved: {out_path} ({written} frames, window={args.window_frames}, distance_factor={args.distance_factor})")

def load_gray_thumbnail(frame_path, size):
    with Image.open(frame_path) as image:
        return np.asarray(image.convert("L").resize((size, size)), dtype=np.float32)

def frame_differencing_scores(frames, background_window, thumbnail_size, anomaly_fraction):
    grays = [load_gray_thumbnail(frame["frame_path"], thumbnail_size) for frame in frames]
    scores = [0.0] * len(frames)
    for i in range(len(frames)):
        window = grays[max(0, i - background_window):i]
        if not window:
            continue
        background = np.median(np.stack(window), axis=0)
        diff = np.abs(grays[i] - background).flatten()
        k = max(1, int(anomaly_fraction * diff.size))
        top = np.partition(diff, -k)[-k:]
        scores[i] = float(top.mean()) / 255.0
    return scores

def cmd_frame_differencing(args):
    cache_path = Path(args.cache).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    sequences = read_cache_sequences(cache_path)
    if args.sequence_ids:
        wanted = set(args.sequence_ids.split(","))
        sequences = {sid: frames for sid, frames in sequences.items() if sid in wanted}

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as out_file:
        for frames in sequences.values():
            scores = frame_differencing_scores(frames, args.background_window, args.thumbnail_size, args.anomaly_fraction)
            for frame, score in zip(frames, scores):
                record = {
                    "sequence_id": frame["sequence_id"],
                    "camera_id": frame.get("camera_id"),
                    "frame_path": frame["frame_path"],
                    "timestamp_unix": frame["timestamp_unix"],
                    "ignition_offset_seconds": frame["ignition_offset_seconds"],
                    "weak_event_label": frame.get("weak_event_label"),
                    "detections": [],
                    "max_smoke_confidence": score,
                    "max_fire_confidence": score,
                    "max_any_confidence": score,
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    console.print(f"Frame-differencing cache saved: {out_path} ({written} frames, background_window={args.background_window}, thumbnail_size={args.thumbnail_size}, anomaly_fraction={args.anomaly_fraction})")

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
                detail = {**event, "rule": row["rule"], "threshold": row["threshold"]}
                if "n" in row:
                    detail["n"] = row["n"]
                    detail["m"] = row["m"]
                if "alpha" in row:
                    detail["alpha"] = row["alpha"]
                    detail["ema_init"] = row["ema_init"]
                lines.append(json.dumps(detail, ensure_ascii=False))
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
    elif args.command == "diagnose":
        cmd_diagnose(args)
    elif args.command == "detection-sizes":
        cmd_detection_sizes(args)
    elif args.command == "refine-tile-agreement":
        cmd_refine_tile_agreement(args)
    elif args.command == "spatial-persistence":
        cmd_spatial_persistence(args)
    elif args.command == "frame-differencing":
        cmd_frame_differencing(args)
    elif args.command == "compare-candidates":
        cmd_compare_candidates(args)

if __name__ == "__main__":
    main()
