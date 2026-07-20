import argparse
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
from ultralytics import __version__ as ultralytics_version

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

    rfdetr_accuracy = subparsers.add_parser("rfdetr-accuracy", help="Evaluate an RF-DETR checkpoint (COCO bbox mAP) on a YOLO-format dataset split")
    rfdetr_accuracy.add_argument("--weights", required=True)
    rfdetr_accuracy.add_argument("--data", required=True, help="Path to data.yaml/dataset.yaml (YOLO-format)")
    rfdetr_accuracy.add_argument("--split", default="val", choices=["train", "val", "test"])
    rfdetr_accuracy.add_argument("--out", default="artifacts/smoke_fire_detection/eval_report.json")
    rfdetr_accuracy.add_argument("--conf", type=float, default=0.001, help="Low threshold to preserve the full precision-recall curve for mAP")
    rfdetr_accuracy.add_argument("--imgsz", type=int, default=640)
    rfdetr_accuracy.add_argument("--batch", type=int, default=8)
    rfdetr_accuracy.add_argument("--class-names", default="", help="Comma-separated class name order overriding the checkpoint's own class_names")
    rfdetr_accuracy.add_argument("--limit", type=int)

    detector_cache = subparsers.add_parser("detector-cache", help="Run a detector over a FIgLib index and cache per-frame scores")
    detector_cache.add_argument("--index", required=True, help="Path to figlib index JSONL")
    detector_cache.add_argument("--weights", required=True)
    detector_cache.add_argument("--out", required=True)
    detector_cache.add_argument("--backend", choices=["yolo", "rfdetr"], default="yolo")
    detector_cache.add_argument("--candidate-id", default="", help="Candidate ID recorded in cache records and sidecar meta")
    detector_cache.add_argument("--class-names", default="", help="Comma-separated class name order overriding the model's own class map, e.g. 'smoke' or 'smoke,fire'")
    detector_cache.add_argument("--conf", type=float, default=0.05)
    detector_cache.add_argument("--iou", type=float, default=0.6, help="Ignored for --backend rfdetr (native postprocess, no external NMS)")
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

    profile = subparsers.add_parser("profile", help="Repeated batch=1 latency + peak VRAM profiling for a candidate")
    profile.add_argument("--backend", choices=["yolo", "rfdetr"], default="yolo")
    profile.add_argument("--weights", required=True)
    profile.add_argument("--images", required=True, help="Directory of real images to sample from")
    profile.add_argument("--candidate-id", default="")
    profile.add_argument("--class-names", default="")
    profile.add_argument("--conf", type=float, default=0.05)
    profile.add_argument("--iou", type=float, default=0.6, help="Ignored for --backend rfdetr")
    profile.add_argument("--imgsz", type=int, default=1280)
    profile.add_argument("--device")
    profile.add_argument("--warmup", type=int, default=30)
    profile.add_argument("--measured", type=int, default=300)
    profile.add_argument("--rounds", type=int, default=3)
    profile.add_argument("--out", default="artifacts/smoke_fire_detection/profile_report.json")

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
    base = Path(data.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = data_yaml.parent / base
    base = base.resolve()
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

def class_map_from_names(names):
    class_map = {str(index): str(name) for index, name in enumerate(names)}
    smoke = [int(class_id) for class_id, name in class_map.items() if "smoke" in name.lower()]
    fire = [int(class_id) for class_id, name in class_map.items() if "fire" in name.lower()]
    if not smoke:
        raise ValueError(f"no smoke class in names: {names}")
    return {"names": class_map, "smoke_class_id": smoke[0], "fire_class_id": fire[0] if fire else None}

def model_class_map(model):
    names = model.names
    ordered = [names[key] for key in sorted(names, key=int)] if isinstance(names, dict) else list(names)
    return class_map_from_names(ordered)

def rfdetr_box_records(detection, class_names):
    records = []
    for class_id, confidence, xyxy in zip(detection.class_id, detection.confidence, detection.xyxy):
        class_id = int(class_id)
        name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        records.append({
            "class_id": class_id,
            "class_name": str(name),
            "confidence": float(confidence),
            "xyxy": [float(value) for value in xyxy],
        })
    return records

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

def yolo_label_path(image_path: Path):
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")

def read_yolo_labels(label_path: Path, image_width, image_height):
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, width, height = (float(value) for value in parts[1:5])
        abs_w = width * image_width
        abs_h = height * image_height
        x_min = cx * image_width - abs_w / 2
        y_min = cy * image_height - abs_h / 2
        boxes.append({"class_id": class_id, "bbox": [x_min, y_min, abs_w, abs_h]})
    return boxes

def build_coco_ground_truth(image_paths, names):
    from faster_coco_eval import COCO
    from PIL import Image

    images = []
    annotations = []
    ann_id = 0
    for image_id, image_path in enumerate(image_paths):
        with Image.open(image_path) as img:
            width, height = img.size
        images.append({"id": image_id, "file_name": str(image_path), "width": width, "height": height})
        for box in read_yolo_labels(yolo_label_path(image_path), width, height):
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": box["class_id"],
                "bbox": box["bbox"],
                "area": box["bbox"][2] * box["bbox"][3],
                "iscrowd": 0,
            })
            ann_id += 1
    categories = [{"id": index, "name": name} for index, name in enumerate(names)]
    coco = COCO()
    coco.dataset = {"images": images, "annotations": annotations, "categories": categories}
    coco.createIndex()
    return coco

def cmd_rfdetr_accuracy(args):
    import torch
    from rfdetr import RFDETRLarge
    from rfdetr.evaluation.coco_eval import CocoEvaluator

    data_path = Path(args.data).resolve()
    weights_path = Path(args.weights).resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    image_paths = [path for path in split_images(data_path, args.split) if path.exists()]
    if args.limit:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise ValueError(f"no images found for split {args.split} in {data_path}")

    model = RFDETRLarge.from_checkpoint(str(weights_path))
    if args.class_names:
        names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    else:
        names = list(model.class_names)

    coco_gt = build_coco_ground_truth(image_paths, names)
    evaluator = CocoEvaluator(coco_gt, ["bbox"])

    predictions = {}
    for batch_start in range(0, len(image_paths), max(1, args.batch)):
        batch_paths = image_paths[batch_start:batch_start + max(1, args.batch)]
        detections = model.predict(
            [str(path) for path in batch_paths],
            threshold=args.conf,
            shape=(args.imgsz, args.imgsz),
            include_source_image=False,
        )
        if not isinstance(detections, list):
            detections = [detections]
        for offset, detection in enumerate(detections):
            image_id = batch_start + offset
            predictions[image_id] = {
                "boxes": torch.as_tensor(detection.xyxy, dtype=torch.float32),
                "scores": torch.as_tensor(detection.confidence, dtype=torch.float32),
                "labels": torch.as_tensor(detection.class_id, dtype=torch.int64),
            }

    evaluator.update(predictions)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    coco_eval = evaluator.coco_eval["bbox"]
    stats = coco_eval.stats
    precisions = coco_eval.eval["precision"]
    classes = {}
    for class_index, name in enumerate(names):
        per_class = precisions[:, :, class_index, 0, -1]
        valid = per_class[per_class > -1]
        per_class_50 = precisions[0, :, class_index, 0, -1]
        valid_50 = per_class_50[per_class_50 > -1]
        classes[name] = {
            "class_id": class_index,
            "mAP50": float(valid_50.mean()) if valid_50.size else None,
            "mAP50_95": float(valid.mean()) if valid.size else None,
        }

    results = {
        "backend": "rfdetr",
        "split": args.split,
        "weights": str(weights_path),
        "data": str(data_path),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "images": len(image_paths),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "framework": f"rfdetr=={rfdetr_version()}",
        "mAP50": float(stats[1]),
        "mAP50_95": float(stats[0]),
        "AR_100": float(stats[8]),
        "classes": classes,
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(f"RF-DETR eval report saved: {out_path}")

def load_backend(args, weights_path):
    if args.class_names:
        override_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    else:
        override_names = None

    if args.backend == "yolo":
        model = YOLO(weights_path)
        class_map = class_map_from_names(override_names) if override_names else model_class_map(model)
        predict_kwargs = {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz, "verbose": False}
        if args.device:
            predict_kwargs["device"] = args.device

        def predict_batch(paths):
            results = model.predict(source=[str(path) for path in paths], **predict_kwargs)
            if len(results) != len(paths):
                raise ValueError(f"batch alignment mismatch: {len(paths)} paths, {len(results)} results")
            return [box_records(result) for result in results]

        def predict_one(path):
            return box_records(model.predict(source=str(path), **predict_kwargs)[0])

        framework_version = f"ultralytics=={ultralytics_version}"
        postprocess = "nms"
        applied_iou = args.iou
    elif args.backend == "rfdetr":
        from rfdetr import RFDETRLarge
        model = RFDETRLarge.from_checkpoint(str(weights_path))
        class_map = class_map_from_names(override_names) if override_names else class_map_from_names(model.class_names)

        def predict_batch(paths):
            detections = model.predict(
                [str(path) for path in paths],
                threshold=args.conf,
                shape=(args.imgsz, args.imgsz),
                include_source_image=False,
            )
            if not isinstance(detections, list):
                detections = [detections]
            if len(detections) != len(paths):
                raise ValueError(f"batch alignment mismatch: {len(paths)} paths, {len(detections)} results")
            return [rfdetr_box_records(detection, model.class_names) for detection in detections]

        def predict_one(path):
            detection = model.predict(str(path), threshold=args.conf, shape=(args.imgsz, args.imgsz), include_source_image=False)
            return rfdetr_box_records(detection, model.class_names)

        framework_version = f"rfdetr=={rfdetr_version()}"
        postprocess = "native"
        applied_iou = None
    else:
        raise ValueError(f"unsupported backend: {args.backend}")

    return predict_batch, predict_one, class_map, framework_version, postprocess, applied_iou

def rfdetr_version():
    from importlib.metadata import version
    return version("rfdetr")

def load_existing_errors(errors_path):
    if not errors_path.exists():
        return []
    return json.loads(errors_path.read_text(encoding="utf-8"))

def runtime_info():
    import torch
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return info

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

    predict_batch, predict_one, class_map, framework_version, postprocess, applied_iou = load_backend(args, weights_path)
    smoke_class_id = class_map["smoke_class_id"]
    fire_class_id = class_map["fire_class_id"]
    data_root = Path(args.data_root).resolve()

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
    run_start = time.perf_counter()
    with out_path.open(mode, encoding="utf-8") as out_file:
        for batch_start in range(0, len(pending), max(1, args.batch)):
            current_batch = pending[batch_start:batch_start + max(1, args.batch)]
            paths = [resolve_frame_path(record, data_root) for record in current_batch]
            try:
                start = time.perf_counter()
                detections_by_record = predict_batch(paths)
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
                        detections_by_record.append(predict_one(path))
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
                    "backend": args.backend,
                    "candidate_id": args.candidate_id,
                    "model_weights": str(weights_path),
                    "conf": args.conf,
                    "iou": applied_iou,
                    "postprocess": postprocess,
                    "imgsz": args.imgsz,
                    "split": record.get("split"),
                    "candidate_revision": args.candidate_revision,
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
    elapsed_seconds = time.perf_counter() - run_start

    errors_path = out_path.with_suffix(out_path.suffix + ".errors.json")
    existing_errors = load_existing_errors(errors_path) if args.resume else []
    merged_by_path = {entry["frame_path"]: entry for entry in existing_errors}
    for entry in failed:
        merged_by_path[entry["frame_path"]] = entry
    merged_errors = list(merged_by_path.values())
    if merged_errors:
        errors_path.write_text(json.dumps(merged_errors, indent=2), encoding="utf-8")
        console.print(f"[yellow]{len(merged_errors)} frame(s) failed total ({len(failed)} this run): {errors_path}[/yellow]")

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta = {
        "candidate_id": args.candidate_id,
        "backend": args.backend,
        "checkpoint": str(Path(args.weights)),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": applied_iou,
        "postprocess": postprocess,
        "framework": framework_version,
        "runtime": runtime_info(),
        "batch": args.batch,
        "command": sys.argv,
        "index_rows": len(records),
        "written_rows": written,
        "failed_rows": len(merged_errors),
        "elapsed_seconds": elapsed_seconds,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    console.print(f"Detector cache saved: {out_path} ({written} frames)")

def cmd_profile(args):
    images_dir = Path(args.images).resolve()
    weights_path = Path(args.weights).resolve()
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    images = sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"no images found under {images_dir}")

    _, predict_one, class_map, framework_version, postprocess, applied_iou = load_backend(args, weights_path)

    import torch
    cuda_available = torch.cuda.is_available()

    rounds = []
    for round_index in range(args.rounds):
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
        for image_path in islice(cycle(images), args.warmup):
            predict_one(image_path)
        latencies = []
        for image_path in islice(cycle(images), args.measured):
            start = time.perf_counter()
            predict_one(image_path)
            latencies.append((time.perf_counter() - start) * 1000)
        rounds.append({
            "round": round_index,
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "fps": float(1000 / np.mean(latencies)),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if cuda_available else None,
        })

    results = {
        "candidate_id": args.candidate_id,
        "backend": args.backend,
        "checkpoint": str(Path(args.weights)),
        "class_map": class_map,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": applied_iou,
        "postprocess": postprocess,
        "framework": framework_version,
        "runtime": runtime_info(),
        "batch": 1,
        "warmup": args.warmup,
        "measured": args.measured,
        "source_images": len(images),
        "command": sys.argv,
        "rounds": rounds,
        "mean_ms": float(np.mean([r["mean_ms"] for r in rounds])),
        "p50_ms": float(np.mean([r["p50_ms"] for r in rounds])),
        "p95_ms": float(np.mean([r["p95_ms"] for r in rounds])),
        "fps": float(np.mean([r["fps"] for r in rounds])),
        "peak_vram_bytes": max((r["peak_vram_bytes"] for r in rounds if r["peak_vram_bytes"] is not None), default=None),
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(f"Profile report saved: {out_path}")

def main():
    args = parse_args()
    if args.command == "accuracy":
        cmd_accuracy(args)
    elif args.command == "rfdetr-accuracy":
        cmd_rfdetr_accuracy(args)
    elif args.command == "detector-cache":
        cmd_detector_cache(args)
    elif args.command == "profile":
        cmd_profile(args)

if __name__ == "__main__":
    main()
