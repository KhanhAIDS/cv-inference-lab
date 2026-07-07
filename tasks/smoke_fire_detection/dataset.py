import argparse
import random
from collections import defaultdict
from pathlib import Path

import yaml
from rich.console import Console

console = Console()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--data-root", required=True, help="Path to raw dataset")
    parser.add_argument("--out", required=True, help="Path to output split folder")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()

def label_type(label_path: Path):
    if not label_path.exists():
        return "empty"
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return "empty"
    class_ids = {line.split()[0] for line in lines if line.split()}
    if "0" in class_ids and "1" in class_ids:
        return "smoke_and_fire"
    if "0" in class_ids:
        return "smoke_only"
    if "1" in class_ids:
        return "fire_only"
    return "empty"

def image_paths(split_dir: Path):
    images_dir = split_dir / "images"
    if not images_dir.exists():
        return []
    return sorted(path.resolve() for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

def write_list(path: Path, images):
    path.write_text("\n".join(str(image) for image in images) + "\n", encoding="utf-8")

def main():
    args = parse_args()
    random.seed(args.seed)
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out).resolve()
    train_labels_dir = data_root / "train" / "labels"
    groups = defaultdict(list)

    for image_path in image_paths(data_root / "train"):
        groups[label_type(train_labels_dir / f"{image_path.stem}.txt")].append(image_path)

    train_images = []
    val_images = []
    for paths in groups.values():
        random.shuffle(paths)
        val_count = int(len(paths) * args.val_ratio)
        val_images.extend(paths[:val_count])
        train_images.extend(paths[val_count:])

    test_images = image_paths(data_root / "test")
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

if __name__ == "__main__":
    main()
