import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from itertools import cycle, islice
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from ultralytics import YOLO

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Run model over data and produce a report")
    subparsers = parser.add_subparsers(dest="command", required=True)

    accuracy = subparsers.add_parser("accuracy", help="Evaluate YOLO model (accuracy and latency)")
    accuracy.add_argument("--weights", required=True)
    accuracy.add_argument("--data", required=True, help="Path to dataset.yaml")
    accuracy.add_argument("--split", default="test", choices=["train", "val", "test"])
    accuracy.add_argument("--out", default="artifacts/smoke_fire_detection/eval_report.json")
    accuracy.add_argument("--conf", type=float, default=0.25)
    accuracy.add_argument("--iou", type=float, default=0.6)
    accuracy.add_argument("--imgsz", type=int, default=640)
    accuracy.add_argument("--device")
    accuracy.add_argument("--warmup", type=int, default=50)
    accuracy.add_argument("--measured", type=int, default=500)

    detector_cache = subparsers.add_parser("detector-cache", help="Run YOLO detector over a FIgLib index and cache per-frame scores")
    detector_cache.add_argument("--index", required=True, help="Path to figlib index JSONL")
    detector_cache.add_argument("--weights", required=True)
    detector_cache.add_argument("--out", required=True)
    detector_cache.add_argument("--conf", type=float, default=0.05)
    detector_cache.add_argument("--iou", type=float, default=0.6)
    detector_cache.add_argument("--imgsz", type=int, default=640)
    detector_cache.add_argument("--device")
    detector_cache.add_argument("--limit", type=int)
    detector_cache.add_argument("--data-root", default=".", help="Runtime root for relative frame_path values")
    detector_cache.add_argument("--split", choices=["dev", "test", "all"], default="all")
    detector_cache.add_argument("--batch", type=int, default=1)
    detector_cache.add_argument("--resume", action="store_true")
    detector_cache.add_argument("--candidate-revision", default="")
    detector_cache.add_argument("--pilot", action="store_true")
    detector_cache.add_argument("--sequence-ids", help="Comma-separated sequence_id allowlist to restrict the index to (for targeted pilot runs)")

    return parser.parse_args()

def numeric(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def class_name(names, class_id: int):
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if class_id < len(names):
        return str(names[class_id])
    return str(class_id)

def per_class_metrics(metric_box, names):
    classes = {}
    class_indexes = getattr(metric_box, "ap_class_index", [])
    class_result = getattr(metric_box, "class_result", None)
    maps = getattr(metric_box, "maps", None)
    for metric_index, class_id in enumerate(class_indexes):
        class_id = int(class_id)
        precision = None
        recall = None
        map50 = None
        map50_95 = None
        if callable(class_result):
            try:
                precision, recall, map50, map50_95 = class_result(metric_index)
            except (IndexError, TypeError, ValueError):
                pass
        if maps is not None and map50_95 is None:
            try:
                map50_95 = maps[class_id]
            except (IndexError, TypeError, ValueError):
                pass
        classes[class_name(names, class_id)] = {
            "class_id": class_id,
            "precision": numeric(precision),
            "recall": numeric(recall),
            "mAP50": numeric(map50),
            "mAP50_95": numeric(map50_95),
        }
    return classes

def split_images(data_yaml: Path, split: str):
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(data.get("path", data_yaml.parent)).resolve()
    split_file = data.get(split)
    if not split_file:
        return []
    split_file = Path(split_file)
    if not split_file.is_absolute():
        split_file = base / split_file
    if not split_file.exists():
        return []
    if split_file.is_dir():
        return sorted(
            path
            for path in split_file.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    return [Path(line.strip()) for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]

def box_records(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    classes = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    coordinates = boxes.xyxy.cpu().tolist()
    names = result.names
    records = []
    for class_id, confidence, xyxy in zip(classes, confidences, coordinates):
        class_id = int(class_id)
        records.append({
            "class_id": class_id,
            "class_name": str(names.get(class_id, class_id)) if isinstance(names, dict) else str(class_id),
            "confidence": float(confidence),
            "xyxy": [float(value) for value in xyxy],
        })
    return records

def read_index(index_path: Path):
    records = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records

def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def model_class_map(model):
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    class_map = {str(int(class_id)): str(name) for class_id, name in items}
    smoke = [int(class_id) for class_id, name in class_map.items() if "smoke" in name.lower()]
    fire = [int(class_id) for class_id, name in class_map.items() if "fire" in name.lower()]
    if not smoke:
        raise ValueError(f"model.names has no smoke class: {class_map}")
    return {"names": class_map, "smoke_class_id": smoke[0], "fire_class_id": fire[0] if fire else None}

def frame_key(record):
    return record["sequence_id"], int(record["timestamp_unix"]), int(record["ignition_offset_seconds"])

def load_resume_keys(path: Path):
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            key = frame_key(record)
            if key in keys:
                raise ValueError(f"duplicate frame key in cache: {key}")
            keys.add(key)
    return keys

def resolve_frame_path(record, data_root: Path):
    path = Path(record["frame_path"])
    if not path.is_absolute():
        path = data_root / path
    return path

def pilot_records(records, limit):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["sequence_id"]].append(record)
    selected = []
    used_cameras = set()
    sequence_limit = max(1, (limit + 2) // 3)
    for sequence_id in sorted(grouped):
        sequence = sorted(grouped[sequence_id], key=lambda item: item["ignition_offset_seconds"])
        camera_id = sequence[0]["camera_id"]
        if camera_id in used_cameras:
            continue
        used_cameras.add(camera_id)
        negative = [record for record in sequence if record["ignition_offset_seconds"] < 0]
        positive = [record for record in sequence if record["ignition_offset_seconds"] >= 0]
        choices = []
        if negative:
            choices.append(negative[-1])
        if positive:
            choices.append(positive[0])
            choices.append(positive[-1])
        selected.extend(choices)
        if len(used_cameras) >= sequence_limit:
            break
    return selected[:limit]

def cmd_accuracy(args):
    weights_path = Path(args.weights).resolve()
    data_path = Path(args.data).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    model = YOLO(weights_path)
    console.print("Starting accuracy evaluation...")
    out_path = Path(args.out).resolve()
    val_kwargs = {
        "data": str(data_path),
        "split": args.split,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "save_json": True,
        "project": str(out_path.parent),
        "name": f"{args.split}_accuracy",
        "exist_ok": True,
    }
    if args.device:
        val_kwargs["device"] = args.device
    metrics = model.val(**val_kwargs)

    results = {
        "split": args.split,
        "weights": str(weights_path),
        "data": str(data_path),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device or str(getattr(model, "device", "auto")),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "mAP50": numeric(getattr(metrics.box, "map50", None)),
        "mAP50_95": numeric(getattr(metrics.box, "map", None)),
        "precision": numeric(getattr(metrics.box, "mp", None)),
        "recall": numeric(getattr(metrics.box, "mr", None)),
        "classes": per_class_metrics(metrics.box, model.names),
    }

    console.print("Starting latency profiling...")
    images = [image for image in split_images(data_path, args.split) if image.exists()]
    if images:
        predict_kwargs = {
            "verbose": False,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
        }
        if args.device:
            predict_kwargs["device"] = args.device
        for image_path in islice(cycle(images), args.warmup):
            model(str(image_path), **predict_kwargs)

        latencies = []
        for image_path in islice(cycle(images), args.measured):
            start = time.perf_counter()
            model(str(image_path), **predict_kwargs)
            latencies.append((time.perf_counter() - start) * 1000)

        results.update({
            "latency_source_images": len(images),
            "warmup": args.warmup,
            "samples": args.measured,
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "fps": float(1000 / np.mean(latencies)),
        })
    else:
        console.print("[yellow]Warning: No images found for latency profiling.[/yellow]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(f"Eval report saved: {out_path}")

def cmd_detector_cache(args):
    index_path = Path(args.index).resolve()
    weights_path = Path(args.weights).resolve()
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    records = read_index(index_path)
    if args.split != "all":
        records = [record for record in records if record.get("split") == args.split]
    if args.sequence_ids:
        allowlist = set(args.sequence_ids.split(","))
        records = [record for record in records if record["sequence_id"] in allowlist]
    if args.pilot:
        records = pilot_records(records, args.limit or 30)
        args.limit = None
    if args.limit:
        records = records[:args.limit]

    model = YOLO(weights_path)
    class_map = model_class_map(model)
    weights_hash = sha256_file(weights_path)
    smoke_class_id = class_map["smoke_class_id"]
    fire_class_id = class_map["fire_class_id"]
    data_root = Path(args.data_root).resolve()
    predict_kwargs = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resume_keys = load_resume_keys(out_path) if args.resume else set()
    pending = [record for record in records if frame_key(record) not in resume_keys]
    if args.resume:
        mode = "a"
    else:
        mode = "w"
    written = 0
    failed = []
    with out_path.open(mode, encoding="utf-8") as out_file:
        for batch_start in range(0, len(pending), max(1, args.batch)):
            batch = pending[batch_start:batch_start + max(1, args.batch)]
            if len(batch) == 1:
                batches = [[record] for record in batch]
            else:
                batches = [batch]
            for current_batch in batches:
                paths = [resolve_frame_path(record, data_root) for record in current_batch]
                try:
                    start = time.perf_counter()
                    results = model.predict(source=[str(path) for path in paths], **predict_kwargs)
                    if len(results) != len(current_batch):
                        raise ValueError(f"batch alignment mismatch: {len(current_batch)} records, {len(results)} results")
                    detections_by_record = [box_records(result) for result in results]
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    latencies = [elapsed_ms / len(current_batch)] * len(current_batch)
                except Exception as batch_error:
                    if len(current_batch) == 1:
                        failed.append({"frame_path": str(paths[0]), "error": str(batch_error)})
                        continue
                    detections_by_record = []
                    latencies = []
                    for record, path in zip(current_batch, paths):
                        try:
                            start = time.perf_counter()
                            result = model.predict(source=str(path), **predict_kwargs)[0]
                            detections_by_record.append(box_records(result))
                            latencies.append((time.perf_counter() - start) * 1000)
                        except Exception as exc:
                            failed.append({"frame_path": str(path), "error": str(exc)})
                            detections_by_record.append(None)
                            latencies.append(None)
                for record, path, detections, latency_ms in zip(current_batch, paths, detections_by_record, latencies):
                    if detections is None:
                        continue
                    smoke_confidences = [d["confidence"] for d in detections if d["class_id"] == smoke_class_id]
                    fire_confidences = [d["confidence"] for d in detections if fire_class_id is not None and d["class_id"] == fire_class_id]
                    any_confidences = [d["confidence"] for d in detections]
                    cache_record = {
                        "sequence_id": record["sequence_id"],
                        "camera_id": record["camera_id"],
                        "frame_path": str(path),
                        "frame_path_index": record["frame_path"],
                        "timestamp_unix": record["timestamp_unix"],
                        "ignition_offset_seconds": record["ignition_offset_seconds"],
                        "weak_event_label": record["weak_event_label"],
                        "model_weights": str(weights_path),
                        "conf": args.conf,
                        "iou": args.iou,
                        "imgsz": args.imgsz,
                        "split": record.get("split"),
                        "candidate_revision": args.candidate_revision,
                        "weights_sha256": weights_hash,
                        "class_map": class_map,
                        "detections": detections,
                        "max_smoke_confidence": max(smoke_confidences) if smoke_confidences else 0.0,
                        "max_fire_confidence": max(fire_confidences) if fire_confidences else 0.0,
                        "max_any_confidence": max(any_confidences) if any_confidences else 0.0,
                        "latency_ms": latency_ms,
                    }
                    out_file.write(json.dumps(cache_record, ensure_ascii=False) + "\n")
                    out_file.flush()
                    written += 1

    if failed:
        errors_path = out_path.with_suffix(out_path.suffix + ".errors.json")
        errors_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
        console.print(f"[yellow]{len(failed)} frame(s) failed and were skipped: {errors_path}[/yellow]")

    console.print(f"Detector cache saved: {out_path} ({written} frames)")

def main():
    args = parse_args()
    if args.command == "accuracy":
        cmd_accuracy(args)
    elif args.command == "detector-cache":
        cmd_detector_cache(args)

if __name__ == "__main__":
    main()
