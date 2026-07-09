import argparse
import json
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from rich.console import Console

console = Console()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALID_CLASS_IDS = {0, 1}
FIGLIB_FRAME_PATTERN = re.compile(r"^(\d+)_([+-]\d+)$")

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset splits/index")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dfire = subparsers.add_parser("dfire", help="Create D-Fire YOLO train/val/test splits")
    dfire.add_argument("--data-root", required=True, help="Path to raw dataset")
    dfire.add_argument("--out", required=True, help="Path to output split folder")
    dfire.add_argument("--audit-out", help="Path to output audit JSON file")
    dfire.add_argument("--val-ratio", type=float, default=0.1)
    dfire.add_argument("--seed", type=int, default=20260707)
    dfire.add_argument("--workers", type=int)

    dfire_dedup = subparsers.add_parser("dfire-dedup", help="Audit D-Fire train/val/test near-duplicates via perceptual hash")
    dfire_dedup.add_argument("--split-dir", required=True, help="Folder with train.txt/val.txt/test.txt")
    dfire_dedup.add_argument("--audit-out", required=True)
    dfire_dedup.add_argument("--hash-size", type=int, default=8)
    dfire_dedup.add_argument("--sample-pairs", type=int, default=20)
    dfire_dedup.add_argument("--workers", type=int)

    figlib = subparsers.add_parser("figlib", help="Build FIgLib frame index and audit")
    figlib.add_argument("--data-root", required=True, help="Path to FIgLib dataset root")
    figlib.add_argument("--index-out", required=True)
    figlib.add_argument("--audit-out", required=True)
    figlib.add_argument("--val-ratio", type=float, default=0.2)
    figlib.add_argument("--seed", type=int, default=20260707)

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
    return class_id, width * height

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
        "class_areas": {0: [], 1: []},
    }

def merge_stats(total, item):
    total["images"] += item["images"]
    total["labels"] += item["labels"]
    total["empty_labels"] += item["empty_labels"]
    total["invalid_label_lines"] += item["invalid_label_lines"]
    total["class_counts"][0] += item["class_counts"].get(0, 0)
    total["class_counts"][1] += item["class_counts"].get(1, 0)
    total["class_areas"][0].extend(item["class_areas"].get(0, []))
    total["class_areas"][1].extend(item["class_areas"].get(1, []))

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
        parsed = parse_label_line(line)
        if parsed is None:
            stats["invalid_label_lines"] += 1
            continue
        class_id, area = parsed
        stats["class_counts"][class_id] = stats["class_counts"].get(class_id, 0) + 1
        stats["class_areas"][class_id].append(area)
        class_ids.add(class_id)
    if 0 in class_ids and 1 in class_ids:
        return "smoke_and_fire", stats
    if 0 in class_ids:
        return "smoke_only", stats
    if 1 in class_ids:
        return "fire_only", stats
    return "empty", stats

def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]

def value_percentiles(areas):
    if not areas:
        return None
    points = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    return {f"p{point}": percentile(areas, point) for point in points}

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

def cmd_dfire(args):
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
        def audit_entry(stats):
            entry = {key: value for key, value in stats.items() if key != "class_areas"}
            entry["class_area_normalized_percentiles"] = {
                "smoke": value_percentiles(stats["class_areas"][0]),
                "fire": value_percentiles(stats["class_areas"][1]),
            }
            return entry

        audit_data = {
            "train": audit_entry(train_stats),
            "test": audit_entry(test_stats),
        }
        audit_path = Path(args.audit_out).resolve()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit_data, indent=2), encoding="utf-8")
        console.print(f"Audit saved: {audit_path}")

def dhash(image_path: Path, hash_size: int):
    with Image.open(image_path) as img:
        pixels = np.asarray(img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS), dtype=np.int16)
    diff = pixels[:, :-1] > pixels[:, 1:]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return np.uint64(bits)

def hash_images(paths, hash_size: int, workers: int):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda path: dhash(path, hash_size), paths))

def min_hamming_distances(query_hashes, reference_hashes, chunk_size=500):
    query_arr = np.array(query_hashes, dtype=np.uint64)
    reference_arr = np.array(reference_hashes, dtype=np.uint64)
    min_dist = np.empty(len(query_arr), dtype=np.int64)
    min_idx = np.empty(len(query_arr), dtype=np.int64)
    for start in range(0, len(query_arr), chunk_size):
        chunk = query_arr[start:start + chunk_size]
        dist = np.bitwise_count(np.bitwise_xor(chunk[:, None], reference_arr[None, :]))
        min_dist[start:start + chunk_size] = dist.min(axis=1)
        min_idx[start:start + chunk_size] = dist.argmin(axis=1)
    return min_dist, min_idx

def dedup_report(query_paths, query_hashes, reference_paths, reference_hashes, sample_pairs):
    min_dist, min_idx = min_hamming_distances(query_hashes, reference_hashes)
    thresholds = [0, 2, 5, 8, 10]
    order = np.argsort(min_dist)
    sample = [
        {
            "query_image": str(query_paths[i]),
            "nearest_reference_image": str(reference_paths[min_idx[i]]),
            "hamming_distance": int(min_dist[i]),
        }
        for i in order[:sample_pairs]
    ]
    return {
        "count_le_threshold": {str(t): int((min_dist <= t).sum()) for t in thresholds},
        "min_distance_percentiles": value_percentiles(min_dist.tolist()),
        "closest_pairs_sample": sample,
    }

def cmd_dfire_dedup(args):
    split_dir = Path(args.split_dir).resolve()
    workers = worker_count(args.workers)

    def read_list(name):
        list_path = split_dir / name
        if not list_path.exists():
            return []
        return [Path(line) for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    train_paths = read_list("train.txt")
    val_paths = read_list("val.txt")
    test_paths = read_list("test.txt")

    console.print(f"Hashing train={len(train_paths)} val={len(val_paths)} test={len(test_paths)} images...")
    train_hashes = hash_images(train_paths, args.hash_size, workers)
    val_hashes = hash_images(val_paths, args.hash_size, workers)
    test_hashes = hash_images(test_paths, args.hash_size, workers)

    audit = {
        "hash_size_bits": args.hash_size * args.hash_size,
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "test_count": len(test_paths),
    }
    if val_paths:
        audit["val_vs_train"] = dedup_report(val_paths, val_hashes, train_paths, train_hashes, args.sample_pairs)
    if test_paths:
        audit["test_vs_train"] = dedup_report(test_paths, test_hashes, train_paths, train_hashes, args.sample_pairs)

    audit_path = Path(args.audit_out).resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    console.print(f"Audit saved: {audit_path}")

FIGLIB_CAMERA_BRANDS = ("mobo", "iqeye")

def figlib_camera_id_from_sequence(sequence_id: str):
    parts = sequence_id.split("_", 2)
    if len(parts) == 3:
        return parts[2]
    tokens = sequence_id.split("-")
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index] in FIGLIB_CAMERA_BRANDS:
            return "-".join(tokens[max(0, index - 2):])
    return "unknown"

def figlib_parse_frame_name(stem: str):
    match = FIGLIB_FRAME_PATTERN.match(stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))

def figlib_scan_sequence(sequence_dir: Path):
    sequence_id = sequence_dir.name
    camera_id = figlib_camera_id_from_sequence(sequence_id)
    videos = sorted(sequence_dir.glob("*.mp4"))
    video_path = videos[0] if videos else None
    frames = []
    bad_filename_frames = 0
    for image_path in sorted(sequence_dir.glob("*.jpg")):
        parsed = figlib_parse_frame_name(image_path.stem)
        if parsed is None:
            bad_filename_frames += 1
            continue
        timestamp_unix, offset = parsed
        frames.append({
            "timestamp_unix": timestamp_unix,
            "ignition_offset_seconds": offset,
            "frame_path": str(image_path.resolve()),
        })
    frames.sort(key=lambda item: item["ignition_offset_seconds"])
    readme_files = len(list(sequence_dir.glob("README.txt")))
    return {
        "sequence_id": sequence_id,
        "camera_id": camera_id,
        "video_path": str(video_path.resolve()) if video_path else None,
        "frames": frames,
        "readme_files": readme_files,
        "bad_filename_frames": bad_filename_frames,
        "missing_video": video_path is None,
        "empty": len(frames) == 0,
    }

def cmd_figlib(args):
    random.seed(args.seed)
    data_root = Path(args.data_root).resolve()
    sequence_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    sequences = [figlib_scan_sequence(d) for d in sequence_dirs]

    shuffled = sequences[:]
    random.shuffle(shuffled)
    val_count = int(len(shuffled) * args.val_ratio)
    val_ids = {seq["sequence_id"] for seq in shuffled[:val_count]}

    index_records = []
    for seq in sequences:
        split = "val" if seq["sequence_id"] in val_ids else "train"
        for frame in seq["frames"]:
            index_records.append({
                "dataset": "FIgLib",
                "sequence_id": seq["sequence_id"],
                "camera_id": seq["camera_id"],
                "video_path": seq["video_path"],
                "frame_path": frame["frame_path"],
                "timestamp_unix": frame["timestamp_unix"],
                "ignition_offset_seconds": frame["ignition_offset_seconds"],
                "weak_event_label": "positive" if frame["ignition_offset_seconds"] >= 0 else "negative",
                "label_source": "filename_offset_sign",
                "split": split,
            })

    index_path = Path(args.index_out).resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in index_records]
    index_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    all_offsets = [frame["ignition_offset_seconds"] for seq in sequences for frame in seq["frames"]]
    audit = {
        "sequences": len(sequences),
        "videos": sum(1 for seq in sequences if seq["video_path"]),
        "frames": sum(len(seq["frames"]) for seq in sequences),
        "readme_files": sum(seq["readme_files"] for seq in sequences),
        "missing_video_sequences": [seq["sequence_id"] for seq in sequences if seq["missing_video"]],
        "empty_sequence_folders": [seq["sequence_id"] for seq in sequences if seq["empty"]],
        "bad_filename_frames": sum(seq["bad_filename_frames"] for seq in sequences),
        "offset_min_seconds": min(all_offsets) if all_offsets else None,
        "offset_max_seconds": max(all_offsets) if all_offsets else None,
        "unique_cameras": len({seq["camera_id"] for seq in sequences}),
        "train_sequences": len(sequences) - len(val_ids),
        "val_sequences": len(val_ids),
    }
    audit_path = Path(args.audit_out).resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    console.print(f"Index saved: {index_path} ({len(index_records)} frames)")
    console.print(f"Audit saved: {audit_path}")

def main():
    args = parse_args()
    if args.command == "dfire":
        cmd_dfire(args)
    elif args.command == "dfire-dedup":
        cmd_dfire_dedup(args)
    elif args.command == "figlib":
        cmd_figlib(args)

if __name__ == "__main__":
    main()