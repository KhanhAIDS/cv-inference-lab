import argparse
import hashlib
import io
import json
import random
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from PIL import Image
from rich.console import Console

console = Console()
FIGLIB_FRAME_PATTERN = re.compile(r"^(\d+)_([+-]\d+)$")

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset splits/index")
    subparsers = parser.add_subparsers(dest="command", required=True)

    figlib = subparsers.add_parser("figlib", help="Build FIgLib frame index and audit")
    figlib.add_argument("--data-root", required=True, help="Path to FIgLib dataset root")
    figlib.add_argument("--index-out", required=True)
    figlib.add_argument("--audit-out", required=True)
    figlib.add_argument("--val-ratio", type=float, default=0.3)
    figlib.add_argument("--seed", type=int, default=20260707)
    figlib.add_argument("--split-mode", choices=["sequence", "camera-disjoint"], default="camera-disjoint")
    figlib.add_argument("--subset-manifest")
    figlib.add_argument("--manifest-out")

    pyro_sdis = subparsers.add_parser("pyro-sdis", help="Convert Pyro-SDIS parquet shards to one-class YOLO")
    pyro_sdis.add_argument("--data-root", required=True, help="Folder containing parquet shards or its data/ child")
    pyro_sdis.add_argument("--out", required=True, help="Output YOLO dataset folder")
    pyro_sdis.add_argument("--audit-out", required=True)
    pyro_sdis.add_argument("--expected-shards", help="Snapshot JSON containing expected SHA-256 by shard filename")
    pyro_sdis.add_argument("--workers", type=int)

    return parser.parse_args()

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

def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def pyro_annotation_lines(value):
    if value is None or not str(value).strip():
        return [], []
    lines = []
    areas = []
    for raw_line in str(value).splitlines():
        parts = raw_line.split()
        if len(parts) != 5:
            raise ValueError(f"invalid annotation fields: {raw_line!r}")
        try:
            class_id = int(parts[0])
            coords = [float(item) for item in parts[1:]]
        except ValueError as error:
            raise ValueError(f"invalid annotation values: {raw_line!r}") from error
        if class_id != 1:
            raise ValueError(f"invalid source class: {class_id}")
        x_center, y_center, width, height = coords
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"bbox outside normalized range: {raw_line!r}")
        lines.append("0 " + " ".join(parts[1:]))
        areas.append(width * height)
    return lines, areas

def pyro_image_bytes(value):
    if isinstance(value, dict):
        value = value.get("bytes")
    if value is None:
        raise ValueError("image bytes missing")
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"unsupported image value: {type(value).__name__}")
    return bytes(value)

def pyro_shard_paths(data_root: Path):
    candidates = data_root / "data" if (data_root / "data").is_dir() else data_root
    paths = sorted(candidates.glob("*.parquet"))
    if not paths:
        raise ValueError(f"no parquet shards under {candidates}")
    return paths

def pyro_split_from_shard(path: Path):
    match = re.match(r"^(train|val)-\d{5}-of-\d{5}\.parquet$", path.name)
    if not match:
        raise ValueError(f"invalid shard filename: {path.name}")
    return match.group(1)

def cmd_pyro_sdis(args):
    started = time.perf_counter()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out).resolve()
    audit_path = Path(args.audit_out).resolve()
    shard_paths = pyro_shard_paths(data_root)
    expected_hashes = None
    expected_counts = None
    if args.expected_shards:
        expected_data = json.loads(Path(args.expected_shards).read_text(encoding="utf-8"))
        expected_hashes = expected_data.get("shards", expected_data)
        expected_counts = expected_data.get("expected")
        if not isinstance(expected_hashes, dict):
            raise ValueError("expected shard JSON must be an object or contain a shards object")
        shard_hashes = {path.name: sha256_file(path) for path in shard_paths}
    else:
        shard_hashes = {}
    hashes_match = expected_hashes is not None and shard_hashes == expected_hashes
    if expected_hashes is not None and not hashes_match:
        missing = sorted(set(expected_hashes) - set(shard_hashes))
        unexpected = sorted(set(shard_hashes) - set(expected_hashes))
        changed = sorted(name for name in set(shard_hashes) & set(expected_hashes) if shard_hashes[name] != expected_hashes[name])
        raise ValueError(json.dumps({"shard_hash_mismatch": {"missing": missing, "unexpected": unexpected, "changed": changed}}))
    if out_dir.exists():
        raise ValueError(f"output already exists: {out_dir}")
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent))
    stats = {
        "images": {"train": 0, "val": 0},
        "empty_annotations": 0,
        "bbox_count": 0,
        "source_class_counts": {},
        "output_class_counts": {},
        "image_dimensions": {},
        "duplicate_filenames": 0,
        "invalid_records": 0,
        "bbox_areas": [],
    }
    names = set()
    try:
        for split in ("train", "val"):
            (temp_dir / "images" / split).mkdir(parents=True)
            (temp_dir / "labels" / split).mkdir(parents=True)
        for shard_path in shard_paths:
            split = pyro_split_from_shard(shard_path)
            table = pq.read_table(shard_path, columns=["image", "annotations", "image_name"])
            for record in table.to_pylist():
                image_name = Path(str(record["image_name"])).name
                if not image_name or image_name in {".", ".."}:
                    raise ValueError("missing image_name")
                if image_name in names:
                    stats["duplicate_filenames"] += 1
                    raise ValueError(f"duplicate filename: {image_name}")
                names.add(image_name)
                image_bytes = pyro_image_bytes(record["image"])
                try:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.load()
                        dimensions = f"{image.width}x{image.height}"
                        if dimensions != "1280x720":
                            raise ValueError(f"unexpected image dimensions: {dimensions}")
                except Exception as error:
                    raise ValueError(f"invalid image {image_name}: {error}") from error
                lines, areas = pyro_annotation_lines(record["annotations"])
                image_out = temp_dir / "images" / split / image_name
                label_out = temp_dir / "labels" / split / f"{Path(image_name).stem}.txt"
                image_out.write_bytes(image_bytes)
                label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                stats["images"][split] += 1
                stats["image_dimensions"][dimensions] = stats["image_dimensions"].get(dimensions, 0) + 1
                if not lines:
                    stats["empty_annotations"] += 1
                stats["bbox_count"] += len(lines)
                stats["bbox_areas"].extend(areas)
                stats["source_class_counts"]["1"] = stats["source_class_counts"].get("1", 0) + len(lines)
                stats["output_class_counts"]["0"] = stats["output_class_counts"].get("0", 0) + len(lines)
        yaml_data = {"path": str(out_dir), "train": "images/train", "val": "images/val", "names": {0: "smoke"}}
        (temp_dir / "dataset.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
        audit = {
            "dataset": "Pyro-SDIS",
            "shards": shard_hashes,
            "expected_shard_hashes_provided": expected_hashes is not None,
            "shard_hashes_match_snapshot": hashes_match if expected_hashes is not None else None,
            "images": stats["images"],
            "total_images": sum(stats["images"].values()),
            "empty_annotations": stats["empty_annotations"],
            "bbox_count": stats["bbox_count"],
            "source_class_counts": stats["source_class_counts"],
            "output_class_counts": stats["output_class_counts"],
            "invalid_records": stats["invalid_records"],
            "duplicate_filenames": stats["duplicate_filenames"],
            "image_dimensions": stats["image_dimensions"],
            "bbox_area_normalized_percentiles": value_percentiles(stats["bbox_areas"]),
            "output_bytes": sum(path.stat().st_size for path in temp_dir.rglob("*") if path.is_file()),
            "runtime_seconds": time.perf_counter() - started,
        }
        if expected_counts is not None:
            actual_counts = {
                "train_images": audit["images"]["train"],
                "val_images": audit["images"]["val"],
                "total_images": audit["total_images"],
                "empty_annotations": audit["empty_annotations"],
                "bbox_count": audit["bbox_count"],
                "source_class": 1 if audit["source_class_counts"] == {"1": audit["bbox_count"]} else None,
                "output_class": 0 if audit["output_class_counts"] == {"0": audit["bbox_count"]} else None,
                "image_dimensions": next(iter(audit["image_dimensions"])) if len(audit["image_dimensions"]) == 1 else None,
            }
            differences = {key: {"expected": expected_counts.get(key), "actual": actual_counts[key]} for key in expected_counts if actual_counts.get(key) != expected_counts.get(key)}
            audit["expected_invariants"] = expected_counts
            audit["invariant_differences"] = differences
            if differences:
                raise ValueError(json.dumps({"snapshot_invariant_mismatch": differences}))
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        temp_dir.rename(out_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    console.print(f"Pyro-SDIS YOLO dataset saved: {out_dir}")
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

def figlib_relative_path(data_root: Path, path):
    if not path:
        return None
    return path.relative_to(data_root).as_posix()

def figlib_manifest(path):
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"sequence_ids": sorted(str(value) for value in data)}
    if not isinstance(data, dict):
        raise ValueError("FIgLib subset manifest must be a JSON object or sequence-id list")
    result = dict(data)
    if "sequence_ids" in result:
        result["sequence_ids"] = sorted(str(value) for value in result["sequence_ids"])
    for key in ("dev_sequence_ids", "test_sequence_ids"):
        if key in result:
            result[key] = sorted(str(value) for value in result[key])
    return result

def figlib_split_sequences(sequences, args, manifest):
    if manifest and "dev_sequence_ids" in manifest and "test_sequence_ids" in manifest:
        dev_ids = set(manifest["dev_sequence_ids"])
        test_ids = set(manifest["test_sequence_ids"])
        if dev_ids & test_ids:
            raise ValueError("FIgLib manifest has sequence overlap between dev and test")
        selected_ids = dev_ids | test_ids
        available_ids = {seq["sequence_id"] for seq in sequences}
        missing = sorted(selected_ids - available_ids)
        if missing:
            raise ValueError(f"FIgLib manifest sequences missing from data root: {missing[:5]}")
        selected = [seq for seq in sequences if seq["sequence_id"] in selected_ids]
    else:
        selected = sequences
        if manifest and manifest.get("sequence_ids") is not None:
            selected_ids = set(manifest["sequence_ids"])
            available_ids = {seq["sequence_id"] for seq in sequences}
            missing = sorted(selected_ids - available_ids)
            if missing:
                raise ValueError(f"FIgLib manifest sequences missing from data root: {missing[:5]}")
            selected = [seq for seq in sequences if seq["sequence_id"] in selected_ids]

        target = int(round(len(selected) * (1 - args.val_ratio)))
        if args.split_mode == "camera-disjoint":
            groups = defaultdict(list)
            for sequence in selected:
                groups[sequence["camera_id"]].append(sequence)
            shuffled_cameras = sorted(groups)
            random.Random(args.seed).shuffle(shuffled_cameras)
            dev_ids = set()
            dev_count = 0
            for camera_id in shuffled_cameras:
                group = groups[camera_id]
                if dev_count < target or not dev_ids:
                    dev_ids.update(seq["sequence_id"] for seq in group)
                    dev_count += len(group)
            dev_cameras = {seq["camera_id"] for seq in selected if seq["sequence_id"] in dev_ids}
            test_ids = {seq["sequence_id"] for seq in selected if seq["sequence_id"] not in dev_ids}
            if dev_cameras & {seq["camera_id"] for seq in selected if seq["sequence_id"] in test_ids}:
                raise ValueError("camera-disjoint split overlap")
        else:
            shuffled = selected[:]
            random.Random(args.seed).shuffle(shuffled)
            dev_ids = {seq["sequence_id"] for seq in shuffled[:target]}
            test_ids = {seq["sequence_id"] for seq in selected if seq["sequence_id"] not in dev_ids}

    split_by_id = {sequence_id: "dev" for sequence_id in dev_ids}
    split_by_id.update({sequence_id: "test" for sequence_id in test_ids})
    return selected, split_by_id

def cmd_figlib(args):
    data_root = Path(args.data_root).resolve()
    sequence_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    sequences = [figlib_scan_sequence(d) for d in sequence_dirs]
    manifest = figlib_manifest(args.subset_manifest)
    sequences, split_by_id = figlib_split_sequences(sequences, args, manifest)

    index_records = []
    for seq in sequences:
        split = split_by_id[seq["sequence_id"]]
        for frame in seq["frames"]:
            index_records.append({
                "dataset": "FIgLib",
                "sequence_id": seq["sequence_id"],
                "camera_id": seq["camera_id"],
                "video_path": figlib_relative_path(data_root, Path(seq["video_path"])) if seq["video_path"] else None,
                "frame_path": figlib_relative_path(data_root, Path(frame["frame_path"])),
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
        "dev_sequences": sum(1 for seq in sequences if split_by_id[seq["sequence_id"]] == "dev"),
        "test_sequences": sum(1 for seq in sequences if split_by_id[seq["sequence_id"]] == "test"),
        "dev_cameras": sorted({seq["camera_id"] for seq in sequences if split_by_id[seq["sequence_id"]] == "dev"}),
        "test_cameras": sorted({seq["camera_id"] for seq in sequences if split_by_id[seq["sequence_id"]] == "test"}),
        "split_mode": args.split_mode,
        "subset_manifest": str(Path(args.subset_manifest).resolve()) if args.subset_manifest else None,
    }
    audit["camera_overlap"] = sorted(set(audit["dev_cameras"]) & set(audit["test_cameras"]))
    if audit["camera_overlap"]:
        raise ValueError(f"FIgLib camera overlap: {audit['camera_overlap']}")
    audit_path = Path(args.audit_out).resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if args.manifest_out:
        split_manifest = {
            "dataset": "FIgLib",
            "seed": args.seed,
            "split_mode": args.split_mode,
            "sequence_ids": sorted(seq["sequence_id"] for seq in sequences),
            "dev_sequence_ids": sorted(sequence_id for sequence_id, split in split_by_id.items() if split == "dev"),
            "test_sequence_ids": sorted(sequence_id for sequence_id, split in split_by_id.items() if split == "test"),
        }
        manifest_path = Path(args.manifest_out).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
        console.print(f"Manifest saved: {manifest_path}")

    console.print(f"Index saved: {index_path} ({len(index_records)} frames)")
    console.print(f"Audit saved: {audit_path}")

def main():
    args = parse_args()
    if args.command == "figlib":
        cmd_figlib(args)
    elif args.command == "pyro-sdis":
        cmd_pyro_sdis(args)

if __name__ == "__main__":
    main()
