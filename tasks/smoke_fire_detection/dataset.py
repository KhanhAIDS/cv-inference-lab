import argparse
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from rich.console import Console

console = Console()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALID_CLASS_IDS = {0, 1}

def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--data-root", required=True, help="Path to raw dataset")
    parser.add_argument("--out", required=True, help="Path to output split folder")
    parser.add_argument("--audit-out", help="Path to output audit JSON file")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--workers", type=int)
    return parser.parse_args()

def parse_label_line(line: str):
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
    except ValueError:
        return None
    if class_id not in VALID_CLASS_IDS:
        return None
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return None
    return class_id

def image_paths(split_dir: Path):
    images_dir = split_dir / "images"
    if not images_dir.exists():
        return []
    return sorted(path.resolve() for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

def write_list(path: Path, images):
    path.write_text("\n".join(str(image) for image in images) + "\n", encoding="utf-8")

def empty_stats():
    return {
        "images": 0,
        "labels": 0,
        "empty_labels": 0,
        "invalid_label_lines": 0,
        "class_counts": {0: 0, 1: 0},
    }

def merge_stats(total, item):
    total["images"] += item["images"]
    total["labels"] += item["labels"]
    total["empty_labels"] += item["empty_labels"]
    total["invalid_label_lines"] += item["invalid_label_lines"]
    total["class_counts"][0] += item["class_counts"].get(0, 0)
    total["class_counts"][1] += item["class_counts"].get(1, 0)

def read_label(label_path: Path):
    stats = empty_stats()
    stats["images"] = 1
    if not label_path.exists():
        stats["empty_labels"] = 1
        return "empty", stats
    stats["labels"] = 1
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        stats["empty_labels"] = 1
        return "empty", stats
    class_ids = set()
    for line in lines:
        class_id = parse_label_line(line)
        if class_id is None:
            stats["invalid_label_lines"] += 1
        else:
            stats["class_counts"][class_id] = stats["class_counts"].get(class_id, 0) + 1
            class_ids.add(class_id)
    if 0 in class_ids and 1 in class_ids:
        return "smoke_and_fire", stats
    if 0 in class_ids:
        return "smoke_only", stats
    if 1 in class_ids:
        return "fire_only", stats
    return "empty", stats

def process_image(args):
    image_path, labels_dir = args
    label_type, stats = read_label(labels_dir / f"{image_path.stem}.txt")
    return image_path, label_type, stats

def scan_split(split_dir: Path, workers: int):
    labels_dir = split_dir / "labels"
    paths = image_paths(split_dir)
    stats = empty_stats()
    if workers > 1 and paths:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(process_image, ((path, labels_dir) for path in paths)))
    else:
        results = [process_image((path, labels_dir)) for path in paths]
    for _, _, item_stats in results:
        merge_stats(stats, item_stats)
    return results, stats

def worker_count(value):
    if value and value > 0:
        return value
    return min(32, os.cpu_count() or 1)

def main():
    args = parse_args()
    random.seed(args.seed)
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out).resolve()
    workers = worker_count(args.workers)
    train_results, train_stats = scan_split(data_root / "train", workers)
    groups = defaultdict(list)

    for image_path, label_type, _ in train_results:
        groups[label_type].append(image_path)

    train_images = []
    val_images = []
    for paths in groups.values():
        random.shuffle(paths)
        val_count = int(len(paths) * args.val_ratio)
        val_images.extend(paths[:val_count])
        train_images.extend(paths[val_count:])

    test_results, test_stats = scan_split(data_root / "test", workers)
    test_images = [image_path for image_path, _, _ in test_results]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_list(out_dir / "train.txt", sorted(train_images))
    write_list(out_dir / "val.txt", sorted(val_images))
    if test_images:
        write_list(out_dir / "test.txt", test_images)

    yaml_data = {
        "path": str(out_dir),
        "train": "train.txt",
        "val": "val.txt",
        "names": {0: "smoke", 1: "fire"},
    }
    if test_images:
        yaml_data["test"] = "test.txt"
        
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
    console.print(f"Split saved: {out_dir}")
    console.print(f"Train: {len(train_images)} Val: {len(val_images)} Test: {len(test_images)}")

    if args.audit_out:
        audit_data = {
            "train": train_stats,
            "test": test_stats,
        }
        audit_path = Path(args.audit_out).resolve()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit_data, indent=2), encoding="utf-8")
        console.print(f"Audit saved: {audit_path}")

if __name__ == "__main__":
    main()
