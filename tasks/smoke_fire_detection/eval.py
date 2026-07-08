import argparse
import json
import platform
import sys
import time
from itertools import cycle, islice
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from ultralytics import YOLO

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model (accuracy and latency)")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default="artifacts/smoke_fire_detection/eval_report.json")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--measured", type=int, default=500)
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

def main():
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    data_path = Path(args.data).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    model = YOLO(weights_path)
    console.print("Starting accuracy evaluation...")
    val_kwargs = {
        "data": str(data_path),
        "split": args.split,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "save_json": True,
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

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(f"Eval report saved: {out_path}")

if __name__ == "__main__":
    main()
