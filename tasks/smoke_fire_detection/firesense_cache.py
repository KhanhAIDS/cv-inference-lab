import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2

from eval import class_map_from_names, rfdetr_box_records, rfdetr_version

CATEGORY_DIRS = {
    "smoke_videos": "smoke",
    "fire_videos": "fire",
}


def discover_videos(root: Path):
    videos = []
    for category_dir, category in CATEGORY_DIRS.items():
        for label_dir, label in (("pos", 1), ("neg", 0)):
            folder = root / category_dir / label_dir
            for video_path in sorted(folder.glob("*.avi")):
                videos.append({
                    "video_id": video_path.stem,
                    "video_path": video_path,
                    "category": category,
                    "label": label,
                })
    return videos


def extract_frames(video_path: Path, target_fps: float):
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    stride = max(1, round(native_fps / target_fps))
    frames = []
    index = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        index += 1
    cap.release()
    return frames, native_fps, stride


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
    return info


def main():
    parser = argparse.ArgumentParser(description="RF-DETR near-field control eval over FIRESENSE (video-level pos/neg label, no ignition timestamps)")
    parser.add_argument("--root", default="datasets/near_field_fire_detection/FIRESENSE")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True, help="Video-level detector cache JSONL (one record per video)")
    parser.add_argument("--candidate-id", default="rfdetr_large_dfire")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--target-fps", type=float, default=1.0, help="Frames sampled per second of video (stride derived per-video from native fps)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit-videos", type=int)
    args = parser.parse_args()

    from rfdetr import RFDETRLarge

    root = Path(args.root).resolve()
    weights_path = Path(args.weights).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(root)
    if not videos:
        raise FileNotFoundError(f"no .avi videos found under {root}")
    if args.limit_videos:
        videos = videos[:args.limit_videos]

    model = RFDETRLarge.from_checkpoint(str(weights_path))
    class_map = class_map_from_names(model.class_names)
    smoke_class_id = class_map["smoke_class_id"]
    fire_class_id = class_map["fire_class_id"]

    written = 0
    failed = []
    run_start = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as out_file:
        for video in videos:
            try:
                frames, native_fps, stride = extract_frames(video["video_path"], args.target_fps)
            except Exception as exc:
                failed.append({"video_path": str(video["video_path"]), "error": str(exc)})
                continue

            frame_scores = []
            for batch_start in range(0, len(frames), max(1, args.batch)):
                batch = frames[batch_start:batch_start + max(1, args.batch)]
                detections_batch = model.predict(batch, threshold=args.conf, shape=(args.imgsz, args.imgsz), include_source_image=False)
                if not isinstance(detections_batch, list):
                    detections_batch = [detections_batch]
                for detection in detections_batch:
                    records = rfdetr_box_records(detection, model.class_names)
                    smoke_confidences = [d["confidence"] for d in records if d["class_id"] == smoke_class_id]
                    fire_confidences = [d["confidence"] for d in records if fire_class_id is not None and d["class_id"] == fire_class_id]
                    frame_scores.append({
                        "max_smoke_confidence": max(smoke_confidences) if smoke_confidences else 0.0,
                        "max_fire_confidence": max(fire_confidences) if fire_confidences else 0.0,
                    })

            max_smoke = max((f["max_smoke_confidence"] for f in frame_scores), default=0.0)
            max_fire = max((f["max_fire_confidence"] for f in frame_scores), default=0.0)
            video_record = {
                "video_id": video["video_id"],
                "video_path": str(video["video_path"]),
                "category": video["category"],
                "label": video["label"],
                "candidate_id": args.candidate_id,
                "backend": "rfdetr",
                "model_weights": str(weights_path),
                "conf": args.conf,
                "imgsz": args.imgsz,
                "native_fps": native_fps,
                "target_fps": args.target_fps,
                "frame_stride": stride,
                "n_frames_sampled": len(frames),
                "max_smoke_confidence": max_smoke,
                "max_fire_confidence": max_fire,
                "max_any_confidence": max(max_smoke, max_fire),
                "frame_scores": frame_scores,
            }
            out_file.write(json.dumps(video_record, ensure_ascii=False) + "\n")
            out_file.flush()
            written += 1
            print(f"progress: {written}/{len(videos)} videos ({video['video_id']}, {len(frames)} frames sampled)", flush=True)
    elapsed_seconds = time.perf_counter() - run_start

    if failed:
        Path(str(out_path) + ".errors.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")

    meta = {
        "candidate_id": args.candidate_id,
        "backend": "rfdetr",
        "checkpoint": str(weights_path),
        "framework": f"rfdetr=={rfdetr_version()}",
        "class_map": class_map,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "target_fps": args.target_fps,
        "batch": args.batch,
        "runtime": runtime_info(),
        "command": sys.argv,
        "n_videos": len(videos),
        "n_written": written,
        "n_failed": len(failed),
        "elapsed_seconds": elapsed_seconds,
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"FIRESENSE detector cache saved: {out_path} ({written} videos, {len(failed)} failed)")


if __name__ == "__main__":
    main()
