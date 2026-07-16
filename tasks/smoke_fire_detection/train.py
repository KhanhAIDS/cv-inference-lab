import argparse
import json
import os
import platform
import shutil
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath


STAGING_COUNTS = {
    "train_images": 29537,
    "val_images": 4099,
    "total_images": 33636,
    "labels": 33636,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Portable YOLO26x training with checkpoint validation")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--run-name", default="yolo26x_pyro_sdis")
    parser.add_argument("--project", default="artifacts/smoke_fire_detection/runs")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--runtime-id", default="unknown")
    parser.add_argument("--session-limit-seconds", type=int, default=0)
    parser.add_argument("--storage-limit", default="unknown")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--resume", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--max-epochs-per-invocation", type=int, default=0)
    parser.add_argument("--max-runtime-seconds", type=int, default=0)
    parser.add_argument("--dataset-stage-max-seconds", type=int, default=600)
    parser.add_argument("--dataset-archive", required=True)
    parser.add_argument("--free-quota", action="store_true")
    parser.add_argument("--paid-spend-usd", type=float)
    parser.add_argument("--paid-cap-usd", type=float, default=20.0)
    parser.add_argument("--paid-stop-threshold-usd", type=float, default=18.0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def fsync_directory(path):
    if os.name == "nt":
        return
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_file(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def runtime_metadata():
    import torch
    import ultralytics

    cuda_available = torch.cuda.is_available()
    gpu = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": properties.total_memory,
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": cuda_available,
        "gpu": gpu,
    }


def verified_dataset_identity(args):
    data_path = Path(args.data).resolve()
    if not data_path.is_file() and not args.dataset_archive:
        raise FileNotFoundError(data_path)
    data = read_yaml(data_path) if data_path.is_file() else {}
    if not isinstance(data, dict):
        raise ValueError("dataset.yaml is not a mapping")
    return {
        "dataset": data.get("dataset", "pyronear/pyro-sdis"),
        "data_path": str(data_path) if data_path.is_file() else None,
    }


def material_config(args):
    return {
        "model": args.model,
        "epochs_total": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "seed": args.seed,
        "amp": not args.no_amp,
        "save_period": args.save_period,
    }


def cost_metadata(args):
    if args.free_quota:
        return {
            "mode": "free_quota",
            "paid_spend_usd": 0.0,
            "paid_cap_usd": args.paid_cap_usd,
            "dashboard_status": "provider confirmation required",
        }
    return {
        "mode": "paid" if args.paid_spend_usd is not None else "unknown",
        "paid_spend_usd": args.paid_spend_usd,
        "paid_cap_usd": args.paid_cap_usd,
        "paid_stop_threshold_usd": args.paid_stop_threshold_usd,
        "dashboard_status": "provided" if args.paid_spend_usd is not None else "not available",
    }


def enforce_cost_gate(args):
    if args.free_quota:
        return
    if args.paid_spend_usd is None:
        raise ValueError("full training blocked: paid spend/quota dashboard is unknown")
    if args.paid_spend_usd >= args.paid_cap_usd:
        raise ValueError("full training blocked: paid cap reached")
    if args.paid_spend_usd >= args.paid_stop_threshold_usd and args.max_epochs_per_invocation > 1:
        raise ValueError("full training blocked: paid spend reached long-block stop threshold")


def checkpoint_resume_state(path):
    import torch

    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"resume checkpoint is not a mapping: {path}")
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"resume checkpoint has invalid epoch: {path}")
    if checkpoint.get("model") is None and checkpoint.get("ema") is None:
        raise ValueError(f"resume checkpoint has no model or EMA weights: {path}")
    required = {
        "ema": checkpoint.get("ema"),
        "optimizer": checkpoint.get("optimizer"),
        "scaler": checkpoint.get("scaler"),
        "scheduler_state": checkpoint.get("scheduler_state"),
        "train_args": checkpoint.get("train_args"),
        "updates": checkpoint.get("updates"),
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(f"resume checkpoint missing state {missing}: {path}")
    if not isinstance(checkpoint["scheduler_state"], dict):
        raise ValueError(f"resume checkpoint scheduler state invalid: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "epoch_zero_based": epoch,
        "completed_epoch_one_based": epoch + 1,
        "next_epoch_one_based": epoch + 2,
        "updates": checkpoint["updates"],
        "model_weights_present": True,
        "optimizer_present": True,
        "ema_present": True,
        "scaler_present": True,
        "scheduler_present": True,
        "train_args_present": True,
    }


def write_resumable_checkpoint(source_path, resume_path, trainer):
    import torch

    source_path = Path(source_path)
    resume_path = Path(resume_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    fsync_file(source_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"training checkpoint is not a mapping: {source_path}")
    scheduler = getattr(trainer, "scheduler", None)
    scaler = getattr(trainer, "scaler", None)
    if scheduler is None or scaler is None:
        raise ValueError("trainer scheduler or scaler unavailable for resumable checkpoint")
    checkpoint["scheduler_state"] = scheduler.state_dict()
    checkpoint["scaler"] = scaler.state_dict()
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
    resume_state = checkpoint_resume_state(resume_path)
    return resume_state


def validate_resume(args, run_dir):
    last_path = Path(run_dir) / "weights" / "last_resume.pt"
    if args.resume == "never":
        if last_path.exists():
            raise ValueError(f"existing run cannot start fresh: {run_dir}")
        return None
    if not last_path.exists():
        if args.resume == "always":
            raise FileNotFoundError(last_path)
        return None
    return checkpoint_resume_state(last_path)


def runtime_resume_trainer(data_path, project, run_name, device=None):
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    class RuntimeResumeTrainer(DetectionTrainer):
        def rebind_runtime_paths(self):
            self.args.data = str(data_path)
            self.args.project = str(project)
            self.args.name = run_name
            if device is not None:
                self.args.device = device
            self.save_dir = Path(project) / run_name
            self.args.save_dir = str(self.save_dir)

        def check_resume(self, overrides):
            super().check_resume(overrides)
            self.rebind_runtime_paths()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rebind_runtime_paths()

    return RuntimeResumeTrainer


def read_yaml(path):
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def staging_lock_path(destination):
    destination = Path(destination)
    return destination.parent / f".{destination.name}.lock"


def staging_temporary_prefix(destination):
    destination = Path(destination)
    return f".{destination.name}.tmp-"


def remove_path(path):
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def validate_archive_members(archive):
    members = archive.getmembers()
    names = set()
    for member in members:
        if not member.isfile():
            raise ValueError(f"archive member is not a regular file: {member.name}")
        original_name = member.name
        windows_path = PureWindowsPath(original_name)
        if windows_path.is_absolute() or windows_path.drive or windows_path.root:
            raise ValueError(f"archive has Windows absolute path: {original_name}")
        normalized_name = original_name.replace("\\", "/")
        relative = PurePosixPath(normalized_name)
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"archive has unsafe relative path: {original_name}")
        key = relative.as_posix().casefold()
        if key in names:
            raise ValueError(f"archive has duplicate relative path: {relative.as_posix()}")
        names.add(key)
    return len(members)


def staged_dataset_data(source_data, destination):
    staged = dict(source_data)
    staged["path"] = str(destination)
    return staged


def yaml_has_windows_absolute_path(value):
    if isinstance(value, dict):
        return any(yaml_has_windows_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(yaml_has_windows_absolute_path(item) for item in value)
    if not isinstance(value, str):
        return False
    path = PureWindowsPath(value)
    return path.is_absolute() or bool(path.drive)


def stage_file_paths(root):
    root = Path(root)
    files = []
    keys = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        windows_path = PureWindowsPath(relative.as_posix())
        if windows_path.is_absolute() or windows_path.drive or windows_path.root:
            raise ValueError(f"staged dataset has Windows absolute path: {relative}")
        key = relative.as_posix().casefold()
        if key in keys:
            raise ValueError(f"staged dataset has duplicate relative path: {relative}")
        keys.add(key)
        if path.stat().st_size == 0:
            raise ValueError(f"staged dataset has 0-byte file: {relative}")
        files.append(path)
    return files


def normalize_empty_label_files(root):
    root = Path(root)
    normalized = 0
    for path in (root / "labels").rglob("*.txt"):
        if path.is_file() and path.stat().st_size == 0:
            path.write_text("\n", encoding="utf-8")
            normalized += 1
    return normalized


def validate_staged_dataset(root, source_data, destination, archive_file_count):
    import yaml

    root = Path(root)
    destination = Path(destination)
    files = stage_file_paths(root)
    if len(files) != archive_file_count:
        raise ValueError(f"staged archive file count mismatch: expected {archive_file_count}, found {len(files)}")
    dataset_yaml = root / "dataset.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError("staged dataset.yaml missing")
    expected_data = staged_dataset_data(source_data, destination)
    actual_data = read_yaml(dataset_yaml)
    if actual_data != expected_data:
        raise ValueError("staged dataset.yaml content mismatch")
    if yaml_has_windows_absolute_path(actual_data):
        raise ValueError("staged dataset.yaml has Windows absolute path")
    image_paths = {}
    for split, expected_count in (("train", STAGING_COUNTS["train_images"]), ("val", STAGING_COUNTS["val_images"])):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"staged split directory missing: {split}")
        images = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) != expected_count:
            raise ValueError(f"staged {split} image count mismatch: expected {expected_count}, found {len(images)}")
        image_paths[split] = images
    labels = sorted(path for path in (root / "labels").rglob("*") if path.is_file() and path.suffix.lower() == ".txt")
    if len(labels) != STAGING_COUNTS["labels"]:
        raise ValueError(f"staged label count mismatch: expected {STAGING_COUNTS['labels']}, found {len(labels)}")
    for split, images in image_paths.items():
        label_dir = root / "labels" / split
        for image in images:
            expected_label = label_dir / image.relative_to(root / "images" / split).with_suffix(".txt")
            if not expected_label.is_file():
                raise FileNotFoundError(f"staged label missing: {expected_label.relative_to(root)}")
    total_images = sum(len(paths) for paths in image_paths.values())
    if total_images != STAGING_COUNTS["total_images"]:
        raise ValueError(f"staged total image count mismatch: expected {STAGING_COUNTS['total_images']}, found {total_images}")
    return {
        **STAGING_COUNTS,
        "archive_files": archive_file_count,
    }


def stage_dataset_archive(data_path, archive_path, max_seconds, staging_root=Path("/tmp")):
    import yaml

    archive_path = Path(archive_path).resolve()
    data_path = Path(data_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError("dataset archive missing")
    source_data = read_yaml(data_path) if data_path.is_file() else None
    if source_data is not None and not isinstance(source_data, dict):
        raise ValueError("source dataset.yaml is not a mapping")
    destination = Path(staging_root) / "smoke-fire-pyro-sdis"
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = staging_lock_path(destination)
    try:
        lock_path.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"dataset staging already in progress: {destination}") from error
    started = time.perf_counter()
    temporary_directory = None
    temporary_archive = None
    try:
        for stale_temporary in destination.parent.glob(f"{staging_temporary_prefix(destination)}*"):
            remove_path(stale_temporary)
        if destination.exists():
            remove_path(destination)
        unique = uuid.uuid4().hex
        temporary_directory = destination.parent / f"{staging_temporary_prefix(destination)}{unique}"
        temporary_archive = destination.parent / f".{destination.name}.archive-{unique}.tar"
        temporary_directory.mkdir()
        copied_bytes = 0
        with archive_path.open("rb") as source, temporary_archive.open("xb") as target:
            while True:
                if max_seconds and time.perf_counter() - started >= max_seconds:
                    raise TimeoutError(f"dataset archive staging exceeded {max_seconds}s")
                chunk = source.read(16 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                copied_bytes += len(chunk)
        extract_started = time.perf_counter()
        with tarfile.open(temporary_archive, "r") as archive:
            archive_file_count = validate_archive_members(archive)
            archive.extractall(temporary_directory, filter="data")
        if source_data is None:
            source_data = read_yaml(temporary_directory / "dataset.yaml")
            if not isinstance(source_data, dict):
                raise ValueError("archive dataset.yaml is not a mapping")
        normalized_empty_labels = normalize_empty_label_files(temporary_directory)
        (temporary_directory / "dataset.yaml").write_text(
            yaml.safe_dump(staged_dataset_data(source_data, destination), sort_keys=False),
            encoding="utf-8",
        )
        counts = validate_staged_dataset(temporary_directory, source_data, destination, archive_file_count)
        temporary_directory.replace(destination)
        temporary_directory = None
        return destination / "dataset.yaml", {
            "source": str(archive_path),
            "destination": str(destination),
            "mode": "archive",
            "bytes": copied_bytes,
            "copy_seconds": extract_started - started,
            "extract_seconds": time.perf_counter() - extract_started,
            "counts": counts,
            "normalized_empty_labels": normalized_empty_labels,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception:
        if temporary_directory is not None:
            remove_path(temporary_directory)
        raise
    finally:
        if temporary_archive is not None:
            temporary_archive.unlink(missing_ok=True)
        lock_path.rmdir()


def run_training(args, sync_callback=None):
    enforce_cost_gate(args)
    if not (args.model == "yolo26x.pt" or args.model.endswith(".pt")):
        raise ValueError("only official yolo26x.pt or .pt checkpoints are allowed for step 13.c")
    if args.imgsz != 1280 or args.epochs != 20 or args.patience != 0 or args.seed != 20260707:
        raise ValueError("step 13.c fixed training config mismatch")
    if args.save_period != 1:
        raise ValueError("step 13.c requires save_period=1")
    if args.batch != 4:
        raise ValueError("training batch must equal 4")
    runtime = runtime_metadata()
    if not runtime["cuda_available"]:
        raise RuntimeError("GPU is required for step 13.c")
    identity = verified_dataset_identity(args)
    config = material_config(args)
    run_dir = Path(args.project).resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_data_path, dataset_stage = stage_dataset_archive(
        Path(args.data).resolve(),
        args.dataset_archive,
        args.dataset_stage_max_seconds,
    )
    write_json_atomic(run_dir / "run_metadata.json", {
        "schema": "smoke-fire-training-run-v1",
        "created_at_utc": utc_now(),
        "provider": args.provider,
        "runtime_id": args.runtime_id,
        "runtime_limits": {
            "session_limit_seconds": args.session_limit_seconds or None,
            "storage_limit": args.storage_limit,
        },
        "runtime": runtime,
        "identity": identity,
        "material_config": config,
        "dataset_stage": dataset_stage,
        "cost": cost_metadata(args),
    })

    resume_state = validate_resume(args, run_dir)
    previous_epoch = resume_state["completed_epoch_one_based"] if resume_state else 0
    started = time.monotonic()

    def on_model_save(trainer):
        completed_epoch = int(trainer.epoch) + 1
        resume_path = run_dir / "weights" / "last_resume.pt"
        write_resumable_checkpoint(
            run_dir / "weights" / "last.pt",
            resume_path,
            trainer,
        )
        if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
            trainer.stop = True
        if args.max_epochs_per_invocation and completed_epoch - previous_epoch >= args.max_epochs_per_invocation:
            trainer.stop = True
        if sync_callback:
            sync_callback()

    from ultralytics import YOLO

    last_path = run_dir / "weights" / "last_resume.pt"
    trainer_class = runtime_resume_trainer(runtime_data_path, Path(args.project).resolve(), args.run_name) if resume_state else None
    if resume_state:
        model = YOLO(str(last_path))
        train_args = {"resume": str(last_path), "data": str(runtime_data_path)}

        def restore_scheduler_state(trainer):
            import torch

            checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state"])

        model.add_callback("on_pretrain_routine_end", restore_scheduler_state)
    else:
        model = YOLO(args.model)
        train_args = {
            "data": str(runtime_data_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "seed": args.seed,
            "workers": args.workers,
            "device": args.device,
            "amp": not args.no_amp,
            "save_period": args.save_period,
            "project": str(Path(args.project).resolve()),
            "name": args.run_name,
            "exist_ok": True,
            "pretrained": True,
        }
    if trainer_class is not None:
        train_args["trainer"] = trainer_class
    model.add_callback("on_model_save", on_model_save)
    model.train(**train_args)
    final_state = checkpoint_resume_state(last_path) if last_path.is_file() else None
    completed_epoch = final_state["completed_epoch_one_based"] if final_state else previous_epoch
    if completed_epoch <= previous_epoch or not last_path.is_file():
        raise RuntimeError("training ended without a newly verified resumable checkpoint")
    checkpoint_resume_state(last_path)
    if sync_callback:
        sync_callback()
    return {
        "mode": "train",
        "run_dir": str(run_dir),
        "started_epoch_one_based": previous_epoch + 1,
        "completed_epoch_one_based": completed_epoch,
    }


def main(argv=None):
    result = run_training(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
