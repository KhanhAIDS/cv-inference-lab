import argparse
import os
import random
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "datasets" / "smoke_fire_detection" / "D-Fire"
SPLIT_DIR = REPO_ROOT / "artifacts" / "smoke_fire_detection" / "dfire_relabeled_split"
DEFAULT_PROJECT = REPO_ROOT / "artifacts" / "smoke_fire_detection" / "runs"
WEIGHTS_CACHE_DIR = Path.home() / ".cache" / "ultralytics_weights"
DEFAULT_MODEL = str(WEIGHTS_CACHE_DIR / "yolo26x.pt")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALID_CLASS_IDS = {0, 1}


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO26x fine-tune on relabeled D-Fire (companion run, not track 13c/13f)"
    )
    parser.add_argument("--run-name", default="dfire_yolo26x_relabeled")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def worker_count(value):
    if value and value > 0:
        return value
    return min(32, os.cpu_count() or 1)


def validate_protocol(args):
    if args.imgsz != 640 or args.epochs != 20 or args.patience != 0 or args.seed != 20260707:
        raise ValueError("D-Fire YOLO protocol is locked: imgsz=640, epochs=20, patience=0, seed=20260707")
    if args.val_ratio != 0.1:
        raise ValueError("D-Fire YOLO protocol is locked: val_ratio=0.1")


def parse_label_line(line):
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


def label_type(label_path):
    if not label_path.exists():
        return "empty"
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    class_ids = {parse_label_line(line) for line in lines}
    class_ids.discard(None)
    if 0 in class_ids and 1 in class_ids:
        return "smoke_and_fire"
    if 0 in class_ids:
        return "smoke_only"
    if 1 in class_ids:
        return "fire_only"
    return "empty"


def image_paths(split_dir):
    images_dir = split_dir / "images"
    if not images_dir.exists():
        return []
    return sorted(
        path.resolve()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_list(path, images):
    path.write_text("\n".join(str(image) for image in images) + "\n", encoding="utf-8")


def build_split(seed, val_ratio):
    random.seed(seed)
    train_images = image_paths(DATA_ROOT / "train")
    labels_dir = DATA_ROOT / "train" / "labels"
    with ThreadPoolExecutor(max_workers=worker_count(None)) as executor:
        types = list(executor.map(lambda p: label_type(labels_dir / f"{p.stem}.txt"), train_images))

    groups = defaultdict(list)
    for image_path, kind in zip(train_images, types):
        groups[kind].append(image_path)

    fit_images = []
    val_images = []
    for paths in groups.values():
        random.shuffle(paths)
        val_count = int(len(paths) * val_ratio)
        val_images.extend(paths[:val_count])
        fit_images.extend(paths[val_count:])

    test_images = image_paths(DATA_ROOT / "test")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    write_list(SPLIT_DIR / "train.txt", sorted(fit_images))
    write_list(SPLIT_DIR / "val.txt", sorted(val_images))
    write_list(SPLIT_DIR / "test.txt", test_images)
    data_yaml = {
        "path": str(SPLIT_DIR),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {0: "smoke", 1: "fire"},
    }
    (SPLIT_DIR / "dataset.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return SPLIT_DIR / "dataset.yaml", len(fit_images), len(val_images), len(test_images)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fsync_file(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path):
    if os.name == "nt":
        return
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def checkpoint_resume_state(path):
    import torch

    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"resume checkpoint is not a mapping: {path}")
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"resume checkpoint has invalid epoch: {path}")
    if checkpoint.get("ema") is None:
        raise ValueError(f"resume checkpoint has no EMA weights: {path}")
    required = {
        "optimizer": checkpoint.get("optimizer"),
        "scaler": checkpoint.get("scaler"),
        "scheduler_state": checkpoint.get("scheduler_state"),
        "train_args": checkpoint.get("train_args"),
        "updates": checkpoint.get("updates"),
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(f"resume checkpoint missing state {missing}: {path}")
    return {"epoch_zero_based": epoch, "completed_epoch_one_based": epoch + 1}


def write_resumable_checkpoint(source_path, resume_path, trainer):
    import torch

    source_path = Path(source_path)
    resume_path = Path(resume_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    fsync_file(source_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    checkpoint["scheduler_state"] = trainer.scheduler.state_dict()
    checkpoint["scaler"] = trainer.scaler.state_dict()
    checkpoint["resume_bundle_schema"] = "smoke-fire-resume-v1"
    checkpoint["resume_created_at_utc"] = utc_now()
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = resume_path.with_name(f".{resume_path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    checkpoint_resume_state(temporary)
    os.replace(temporary, resume_path)
    fsync_directory(resume_path.parent)
    return checkpoint_resume_state(resume_path)


def main():
    args = parse_args()
    validate_protocol(args)
    data_yaml, fit_count, val_count, test_count = build_split(args.seed, args.val_ratio)
    print(f"split: train={fit_count} val={val_count} test={test_count}")

    WEIGHTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.project).resolve() / args.run_name
    last_resume_path = run_dir / "weights" / "last_resume.pt"
    resuming = last_resume_path.exists()
    if resuming:
        resume_state = checkpoint_resume_state(last_resume_path)
        print(f"resuming from epoch {resume_state['completed_epoch_one_based']}")

    def on_model_save(trainer):
        write_resumable_checkpoint(run_dir / "weights" / "last.pt", last_resume_path, trainer)

    from ultralytics import YOLO

    if resuming:
        model = YOLO(str(last_resume_path))
        train_args = {"resume": str(last_resume_path), "data": str(data_yaml)}

        def restore_scheduler_state(trainer):
            import torch

            checkpoint = torch.load(last_resume_path, map_location="cpu", weights_only=False)
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state"])

        model.add_callback("on_pretrain_routine_end", restore_scheduler_state)
    else:
        model = YOLO(args.model)
        train_args = {
            "data": str(data_yaml),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "seed": args.seed,
            "deterministic": True,
            "workers": args.workers,
            "device": args.device,
            "project": str(Path(args.project).resolve()),
            "name": args.run_name,
            "exist_ok": True,
            "pretrained": True,
        }
    model.add_callback("on_model_save", on_model_save)
    model.train(**train_args)
    print("TRAIN_FINISHED")


if __name__ == "__main__":
    main()
