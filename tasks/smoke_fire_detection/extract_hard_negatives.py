import argparse
import json
import platform
import sys
from pathlib import Path

import yaml
from rich.console import Console
from ultralytics import YOLO

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Extract false positives from empty-label images")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default="artifacts/smoke_fire_detection/hard_negatives.jsonl")
    parser.add_argument("--review-dir", default="artifacts/smoke_fire_detection/hard_negative_review")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--max-review-images", type=int, default=200)
    return parser.parse_args()

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

def main():
    args = parse_args()
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

if __name__ == "__main__":
    main()
