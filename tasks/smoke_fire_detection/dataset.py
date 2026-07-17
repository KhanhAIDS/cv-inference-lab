import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

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

    return parser.parse_args()

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

if __name__ == "__main__":
    main()
