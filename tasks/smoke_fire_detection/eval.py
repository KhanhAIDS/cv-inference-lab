import argparse
import json
import platform
import sys
import time
from itertools import cycle, islice
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
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

    hard_negatives = subparsers.add_parser("hard-negatives", help="Extract false positives from empty-label images")
    hard_negatives.add_argument("--weights", required=True)
    hard_negatives.add_argument("--data", required=True, help="Path to dataset.yaml")
    hard_negatives.add_argument("--split", default="test", choices=["train", "val", "test"])
    hard_negatives.add_argument("--out", default="artifacts/smoke_fire_detection/hard_negatives.jsonl")
    hard_negatives.add_argument("--review-dir", default="artifacts/smoke_fire_detection/hard_negative_review")
    hard_negatives.add_argument("--conf", type=float, default=0.25)
    hard_negatives.add_argument("--iou", type=float, default=0.6)
    hard_negatives.add_argument("--imgsz", type=int, default=640)
    hard_negatives.add_argument("--device")
    hard_negatives.add_argument("--limit", type=int)
    hard_negatives.add_argument("--max-candidates", type=int, default=1000)
    hard_negatives.add_argument("--max-review-images", type=int, default=200)

    detector_cache = subparsers.add_parser("detector-cache", help="Run YOLO detector over a FIgLib index and cache per-frame scores")
    detector_cache.add_argument("--index", required=True, help="Path to figlib index JSONL")
    detector_cache.add_argument("--weights", required=True)
    detector_cache.add_argument("--out", required=True)
    detector_cache.add_argument("--conf", type=float, default=0.05)
    detector_cache.add_argument("--iou", type=float, default=0.6)
    detector_cache.add_argument("--imgsz", type=int, default=640)
    detector_cache.add_argument("--device")
    detector_cache.add_argument("--limit", type=int)
    detector_cache.add_argument("--tile-grid", help="COLSxROWS, e.g. 2x2 - run detector per native-resolution tile instead of one global resize")
    detector_cache.add_argument("--tile-overlap", type=float, default=0.15)
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
    return [Path(line.strip()) for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]

def label_path_for_image(image_path: Path):
    if image_path.parent.name == "images":
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")

def is_empty_label(image_path: Path):
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return True
    return not any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())

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

def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def read_index(index_path: Path):
    records = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records

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

def cmd_hard_negatives(args):
    weights_path = Path(args.weights).resolve()
    data_path = Path(args.data).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    images = [image for image in split_images(data_path, args.split) if image.exists()]
    empty_images = [image for image in images if is_empty_label(image)]
    if args.limit:
        empty_images = empty_images[:args.limit]

    model = YOLO(weights_path)
    predict_kwargs = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    candidates = []
    for image_path in empty_images:
        result = model.predict(source=str(image_path), **predict_kwargs)[0]
        detections = box_records(result)
        if not detections:
            continue
        max_confidence = max(detection["confidence"] for detection in detections)
        candidates.append({
            "image": str(image_path),
            "label": str(label_path_for_image(image_path)),
            "split": args.split,
            "max_confidence": float(max_confidence),
            "detections": detections,
        })

    candidates.sort(key=lambda item: item["max_confidence"], reverse=True)
    candidates = candidates[:args.max_candidates]
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    out_path = Path(args.out).resolve()
    write_jsonl(out_path, candidates)

    review_dir = Path(args.review_dir).resolve()
    if args.max_review_images > 0 and candidates:
        review_dir.mkdir(parents=True, exist_ok=True)
        for candidate in candidates[:args.max_review_images]:
            image_path = Path(candidate["image"])
            result = model.predict(source=str(image_path), **predict_kwargs)[0]
            result.save(filename=str(review_dir / f"{candidate['rank']:04d}_{image_path.name}"))

    summary = {
        "command": sys.argv,
        "weights": str(weights_path),
        "data": str(data_path),
        "split": args.split,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device or str(getattr(model, "device", "auto")),
        "source_images": len(images),
        "empty_label_images": len(empty_images),
        "candidate_images": len(candidates),
        "output": str(out_path),
        "review_dir": str(review_dir),
        "python": sys.version,
        "platform": platform.platform(),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(f"Hard negatives saved: {out_path}")
    console.print(f"Summary saved: {summary_path}")

def compute_tiles(width, height, cols, rows, overlap):
    step_x = width / cols
    step_y = height / rows
    tile_w = min(float(width), step_x * (1 + overlap))
    tile_h = min(float(height), step_y * (1 + overlap))
    tiles = []
    for row in range(rows):
        for col in range(cols):
            cx = step_x * (col + 0.5)
            cy = step_y * (row + 0.5)
            x1 = min(float(width), max(tile_w, cx + tile_w / 2))
            y1 = min(float(height), max(tile_h, cy + tile_h / 2))
            x0 = x1 - tile_w
            y0 = y1 - tile_h
            tiles.append((x0, y0, x1, y1))
    return tiles

def tiled_detections(model, frame_path, cols, rows, overlap, predict_kwargs):
    with Image.open(frame_path) as img:
        width, height = img.size
        tiles = compute_tiles(width, height, cols, rows, overlap)
        crops = [img.crop((int(x0), int(y0), int(x1), int(y1))) for x0, y0, x1, y1 in tiles]
    results = model.predict(source=crops, **predict_kwargs)
    detections = []
    for (x0, y0, x1, y1), result in zip(tiles, results):
        for detection in box_records(result):
            dx0, dy0, dx1, dy1 = detection["xyxy"]
            detection["xyxy"] = [dx0 + x0, dy0 + y0, dx1 + x0, dy1 + y0]
            detections.append(detection)
    return detections

def cmd_detector_cache(args):
    index_path = Path(args.index).resolve()
    weights_path = Path(args.weights).resolve()
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    records = read_index(index_path)
    if args.sequence_ids:
        allowlist = set(args.sequence_ids.split(","))
        records = [record for record in records if record["sequence_id"] in allowlist]
    if args.limit:
        records = records[:args.limit]

    tile_cols = tile_rows = None
    if args.tile_grid:
        tile_cols, tile_rows = (int(value) for value in args.tile_grid.lower().split("x"))

    model = YOLO(weights_path)
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
    written = 0
    failed = []
    with out_path.open("w", encoding="utf-8") as out_file:
        for record in records:
            frame_path = record["frame_path"]
            try:
                start = time.perf_counter()
                if tile_cols:
                    detections = tiled_detections(model, frame_path, tile_cols, tile_rows, args.tile_overlap, predict_kwargs)
                else:
                    result = model.predict(source=frame_path, **predict_kwargs)[0]
                    detections = box_records(result)
                latency_ms = (time.perf_counter() - start) * 1000
            except Exception as exc:
                failed.append({"frame_path": frame_path, "error": str(exc)})
                continue
            smoke_confidences = [d["confidence"] for d in detections if d["class_id"] == 0]
            fire_confidences = [d["confidence"] for d in detections if d["class_id"] == 1]
            any_confidences = [d["confidence"] for d in detections]
            cache_record = {
                "sequence_id": record["sequence_id"],
                "camera_id": record["camera_id"],
                "frame_path": frame_path,
                "timestamp_unix": record["timestamp_unix"],
                "ignition_offset_seconds": record["ignition_offset_seconds"],
                "weak_event_label": record["weak_event_label"],
                "model_weights": str(weights_path),
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "tile_grid": args.tile_grid,
                "tile_overlap": args.tile_overlap if args.tile_grid else None,
                "detections": detections,
                "max_smoke_confidence": max(smoke_confidences) if smoke_confidences else 0.0,
                "max_fire_confidence": max(fire_confidences) if fire_confidences else 0.0,
                "max_any_confidence": max(any_confidences) if any_confidences else 0.0,
                "latency_ms": latency_ms,
            }
            out_file.write(json.dumps(cache_record, ensure_ascii=False) + "\n")
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
    elif args.command == "hard-negatives":
        cmd_hard_negatives(args)
    elif args.command == "detector-cache":
        cmd_detector_cache(args)

if __name__ == "__main__":
    main()