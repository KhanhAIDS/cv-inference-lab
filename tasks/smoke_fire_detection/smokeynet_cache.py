import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

RESIZE_DIMENSIONS = (1392, 1856)
CROP_HEIGHT = 1040
TILE_DIMENSIONS = (224, 224)
TILE_OVERLAP = 20


def calculate_num_tiles():
    num_tiles_height = 1 + (CROP_HEIGHT - TILE_DIMENSIONS[0]) // (TILE_DIMENSIONS[0] - TILE_OVERLAP)
    num_tiles_width = 1 + (RESIZE_DIMENSIONS[1] - TILE_DIMENSIONS[1]) // (TILE_DIMENSIONS[1] - TILE_OVERLAP)
    return num_tiles_height, num_tiles_width


def tile_image(img, num_tiles_height, num_tiles_width):
    bytelength = img.nbytes // img.size
    img = np.lib.stride_tricks.as_strided(
        img,
        shape=(num_tiles_height, num_tiles_width, TILE_DIMENSIONS[0], TILE_DIMENSIONS[1], 3),
        strides=(
            RESIZE_DIMENSIONS[1] * (TILE_DIMENSIONS[0] - TILE_OVERLAP) * bytelength * 3,
            (TILE_DIMENSIONS[1] - TILE_OVERLAP) * bytelength * 3,
            RESIZE_DIMENSIONS[1] * bytelength * 3,
            bytelength * 3,
            bytelength,
        ),
        writeable=False,
    )
    return img.reshape((-1, TILE_DIMENSIONS[0], TILE_DIMENSIONS[1], 3))


def normalize_image(img):
    img = img / 255
    return (img - 0.5) / 0.5


def preprocess_image(img_array, num_tiles_height, num_tiles_width):
    img = Image.fromarray(img_array).convert("RGB")
    img = img.resize((RESIZE_DIMENSIONS[1], RESIZE_DIMENSIONS[0]))
    img = np.ascontiguousarray(np.array(img)[-CROP_HEIGHT:])
    img = tile_image(img, num_tiles_height, num_tiles_width)
    return normalize_image(img)


def generate_input_data(current_img, previous_img, num_tiles_height, num_tiles_width):
    x = [preprocess_image(current_img, num_tiles_height, num_tiles_width),
         preprocess_image(previous_img, num_tiles_height, num_tiles_width)]
    x = np.transpose(np.stack(x), (1, 0, 4, 2, 3))
    x = np.expand_dims(x, axis=0).astype(np.float64)
    return x


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def read_index(index_path):
    records = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def frame_key(sequence_id, record):
    return sequence_id, int(record["timestamp_unix"]), int(record["ignition_offset_seconds"])


def load_resume_keys(path):
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            keys.add((record["sequence_id"], int(record["timestamp_unix"]), int(record["ignition_offset_seconds"])))
    return keys


def main():
    parser = argparse.ArgumentParser(description="SmokeyNet (ONNX, 2-frame tile classifier) reference cache over FIgLib index")
    parser.add_argument("--index", required=True)
    parser.add_argument("--model", required=True, help="Path to SmokeyNet model.onnx")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sequence-ids", help="Comma-separated sequence_id allowlist, for pilot runs")
    parser.add_argument("--candidate-id", default="smokeynet_reference")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    index_path = Path(args.index).resolve()
    model_path = Path(args.model).resolve()
    data_root = Path(args.data_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_index(index_path)
    if args.split != "all":
        records = [r for r in records if r.get("split") == args.split]
    if args.sequence_ids:
        wanted = set(args.sequence_ids.split(","))
        records = [r for r in records if r["sequence_id"] in wanted]

    by_sequence = defaultdict(list)
    for record in records:
        by_sequence[record["sequence_id"]].append(record)
    for frames in by_sequence.values():
        frames.sort(key=lambda r: r["timestamp_unix"])

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    num_tiles_height, num_tiles_width = calculate_num_tiles()

    resume_keys = load_resume_keys(out_path) if args.resume else set()
    total_pairs = sum(max(0, len(frames) - 1) for frames in by_sequence.values())

    written = 0
    failed = []
    n_first_frames_skipped = 0
    n_skipped_resume = 0
    mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as out_file:
        for sequence_id, frames in by_sequence.items():
            if frames:
                n_first_frames_skipped += 1
            for i in range(1, len(frames)):
                if args.limit and written >= args.limit:
                    break
                current_record = frames[i]
                previous_record = frames[i - 1]
                if frame_key(sequence_id, current_record) in resume_keys:
                    n_skipped_resume += 1
                    continue
                current_path = data_root / current_record["frame_path"]
                previous_path = data_root / previous_record["frame_path"]
                start = time.time()
                try:
                    current_img = np.array(Image.open(current_path).convert("RGB"))
                    previous_img = np.array(Image.open(previous_path).convert("RGB"))
                    x = generate_input_data(current_img, previous_img, num_tiles_height, num_tiles_width)
                    outputs = session.run(None, {input_name: x})[0]
                    max_smoke_confidence = float(sigmoid(outputs).max())
                    latency_ms = (time.time() - start) * 1000
                except Exception as exc:
                    failed.append({"frame_path": str(current_path), "error": str(exc)})
                    continue
                out_record = {
                    "sequence_id": sequence_id,
                    "camera_id": current_record["camera_id"],
                    "frame_path": str(current_path),
                    "frame_path_index": current_record["frame_path"],
                    "previous_frame_path_index": previous_record["frame_path"],
                    "timestamp_unix": current_record["timestamp_unix"],
                    "ignition_offset_seconds": current_record["ignition_offset_seconds"],
                    "weak_event_label": current_record["weak_event_label"],
                    "backend": "smokeynet",
                    "candidate_id": args.candidate_id,
                    "model_weights": str(model_path),
                    "split": current_record.get("split", args.split),
                    "detections": [],
                    "max_smoke_confidence": max_smoke_confidence,
                    "max_fire_confidence": 0.0,
                    "max_any_confidence": max_smoke_confidence,
                    "latency_ms": latency_ms,
                }
                out_file.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                out_file.flush()
                written += 1
                if args.progress_every and written % args.progress_every == 0:
                    print(f"progress: {written + n_skipped_resume}/{total_pairs} pairs (this run: {written} written, {n_skipped_resume} resumed-skipped, {len(failed)} failed)", flush=True)
            if args.limit and written >= args.limit:
                break

    if failed:
        Path(str(out_path) + ".errors.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
    Path(str(out_path) + ".meta.json").write_text(json.dumps({
        "candidate_id": args.candidate_id,
        "model": str(model_path),
        "n_frames_written_this_run": written,
        "n_frames_skipped_resume": n_skipped_resume,
        "n_frames_total_in_output": written + n_skipped_resume,
        "n_failed": len(failed),
        "n_sequences": len(by_sequence),
        "n_first_frames_skipped_no_previous_frame": n_first_frames_skipped,
        "temporal_delta": "1 native FIgLib sample step (60s cadence, matches SmokeyNet paper 2-frame spacing)",
        "preprocessing": "resize(1392x1856) -> crop bottom 1040 rows -> tile 224x224 overlap 20 -> normalize (x/255-0.5)/0.5, replicated from sagecontinuum/sage-smoke-detection src/inference.py",
    }, indent=2), encoding="utf-8")
    print(f"SmokeyNet cache saved: {out_path} ({written} frames, {len(failed)} failed, {len(by_sequence)} sequences, {n_first_frames_skipped} first-frames skipped for lack of a previous frame)")


if __name__ == "__main__":
    main()
