import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath


STAGING_SCHEMA_VERSION = "smoke-fire-pyro-sdis-stage-v2"
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
    parser.add_argument("--dataset-snapshot", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--dependency-lock", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--runtime-id", default="unknown")
    parser.add_argument("--session-limit-seconds", type=int, default=0)
    parser.add_argument("--storage-limit", default="unknown")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--resume", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=30)
    parser.add_argument("--smoke-use-archive", action="store_true")
    parser.add_argument("--max-epochs-per-invocation", type=int, default=0)
    parser.add_argument("--max-runtime-seconds", type=int, default=0)
    parser.add_argument("--dataset-stage-max-seconds", type=int, default=600)
    parser.add_argument("--dataset-archive")
    parser.add_argument("--dataset-archive-manifest")
    parser.add_argument("--subset-images", type=int, default=0)
    parser.add_argument("--free-quota", action="store_true")
    parser.add_argument("--paid-spend-usd", type=float)
    parser.add_argument("--paid-cap-usd", type=float, default=20.0)
    parser.add_argument("--paid-stop-threshold-usd", type=float, default=18.0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


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
    snapshot_path = Path(args.dataset_snapshot).resolve()
    audit_path = Path(args.dataset_audit).resolve()
    lock_path = Path(args.dependency_lock).resolve()
    runtime_lock_path = Path(args.runtime_lock).resolve()
    for path in (data_path, snapshot_path, audit_path, lock_path, runtime_lock_path):
        if not path.exists():
            raise FileNotFoundError(path)
    snapshot = read_json(snapshot_path)
    audit = read_json(audit_path)
    snapshot_shards = snapshot.get("shards")
    if not isinstance(snapshot_shards, dict) or not snapshot_shards:
        raise ValueError("dataset snapshot has no shard hashes")
    if audit.get("shards") != snapshot_shards:
        raise ValueError("dataset audit shard hashes do not match snapshot")
    if audit.get("invariant_differences"):
        raise ValueError("dataset audit invariants failed")
    if audit.get("shard_hashes_match_snapshot") is not True:
        raise ValueError("dataset audit did not verify shard snapshot")
    return {
        "dataset_snapshot_sha256": sha256_file(snapshot_path),
        "dataset_audit_sha256": sha256_file(audit_path),
        "dependency_lock_sha256": sha256_file(lock_path),
        "runtime_lock_sha256": sha256_file(runtime_lock_path),
        "training_entry_sha256": sha256_file(Path(__file__).resolve()),
        "dataset": snapshot.get("dataset"),
        "shard_count": len(snapshot_shards),
    }


def material_config(args, identity):
    return {
        "model": args.model,
        "epochs_total": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "seed": args.seed,
        "amp": not args.no_amp,
        "save_period": args.save_period,
        "subset_images": args.subset_images,
        "dataset_snapshot_sha256": identity["dataset_snapshot_sha256"],
        "dataset_audit_sha256": identity["dataset_audit_sha256"],
        "dependency_lock_sha256": identity["dependency_lock_sha256"],
        "runtime_lock_sha256": identity["runtime_lock_sha256"],
        "training_entry_sha256": identity["training_entry_sha256"],
        "code_commit": args.code_commit,
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
    if args.smoke_only or args.free_quota:
        return
    if args.paid_spend_usd is None:
        raise ValueError("full training blocked: paid spend/quota dashboard is unknown")
    if args.paid_spend_usd >= args.paid_cap_usd:
        raise ValueError("full training blocked: paid cap reached")
    if args.paid_spend_usd >= args.paid_stop_threshold_usd and args.max_epochs_per_invocation > 1:
        raise ValueError("full training blocked: paid spend reached long-block stop threshold")


def last_results_row(run_dir):
    path = Path(run_dir) / "results.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def checkpoint_files(run_dir):
    files = {}
    for name in ("last.pt", "last_resume.pt", "best.pt"):
        path = Path(run_dir) / "weights" / name
        files[name] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "bytes": path.stat().st_size if path.exists() else None,
        }
    return files


def resumable_snapshot_files(run_dir):
    weights_dir = Path(run_dir) / "weights"
    snapshots = {}
    for path in sorted(weights_dir.glob("epoch_*_resume.pt")):
        snapshots[path.name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return snapshots


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
        "dataset_identity": checkpoint.get("dataset_identity"),
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(f"resume checkpoint missing state {missing}: {path}")
    if not isinstance(checkpoint["scheduler_state"], dict):
        raise ValueError(f"resume checkpoint scheduler state invalid: {path}")
    if not isinstance(checkpoint["dataset_identity"], dict):
        raise ValueError(f"resume checkpoint dataset identity invalid: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "epoch_zero_based": epoch,
        "next_epoch_one_based": epoch + 2,
        "updates": checkpoint["updates"],
        "model_weights_present": True,
        "optimizer_present": True,
        "ema_present": True,
        "scaler_present": True,
        "scheduler_present": True,
        "train_args_present": True,
        "dataset_identity": checkpoint["dataset_identity"],
    }


def write_resumable_checkpoint(source_path, resume_path, snapshot_path, trainer, identity):
    import torch

    source_path = Path(source_path)
    resume_path = Path(resume_path)
    snapshot_path = Path(snapshot_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if snapshot_path.exists():
        raise FileExistsError(f"immutable resume snapshot already exists: {snapshot_path}")
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
    checkpoint["dataset_identity"] = dict(identity)
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
    snapshot_temporary = snapshot_path.with_name(f".{snapshot_path.name}.{uuid.uuid4().hex}.tmp")
    with resume_path.open("rb") as source, snapshot_temporary.open("xb") as target:
        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    os.replace(snapshot_temporary, snapshot_path)
    fsync_directory(snapshot_path.parent)
    resume_state = checkpoint_resume_state(resume_path)
    snapshot_state = checkpoint_resume_state(snapshot_path)
    if resume_state["sha256"] != snapshot_state["sha256"]:
        raise ValueError("resumable checkpoint and immutable snapshot hashes differ")
    return resume_state


def previous_manifest(run_dir):
    path = Path(run_dir) / "checkpoint_manifest.json"
    return read_json(path) if path.exists() else None


def validate_resume(args, run_dir, config):
    manifest = previous_manifest(run_dir)
    last_path = Path(run_dir) / "weights" / "last_resume.pt"
    if args.resume == "never":
        if last_path.exists() or manifest:
            raise ValueError(f"existing run cannot start fresh: {run_dir}")
        return None
    if not last_path.exists():
        if args.resume == "always":
            raise FileNotFoundError(last_path)
        return None
    if not manifest:
        raise ValueError(f"resume blocked: checkpoint manifest missing for {last_path}")
    if manifest.get("material_config") != config:
        raise ValueError("resume blocked: material config mismatch")
    recorded = manifest.get("checkpoint_files", {}).get("last_resume.pt", {}).get("sha256")
    actual = sha256_file(last_path)
    if not recorded or recorded != actual:
        raise ValueError("resume blocked: last_resume.pt hash mismatch")
    state = checkpoint_resume_state(last_path)
    if state["dataset_identity"] != manifest.get("identity"):
        raise ValueError("resume blocked: checkpoint dataset identity mismatch")
    expected_snapshot = Path(run_dir) / "weights" / f"epoch_{state['epoch_zero_based'] + 1:03d}_resume.pt"
    if not expected_snapshot.is_file() or sha256_file(expected_snapshot) != actual:
        raise ValueError("resume blocked: immutable checkpoint snapshot mismatch")
    return manifest


def write_checkpoint_manifest(run_dir, args, identity, config, runtime, completed_epoch, stop_reason):
    manifest = {
        "schema": "smoke-fire-checkpoint-v1",
        "created_at_utc": utc_now(),
        "run_name": args.run_name,
        "completed_epoch": completed_epoch,
        "epochs_total": args.epochs,
        "stop_reason": stop_reason,
        "material_config": config,
        "identity": identity,
        "runtime": runtime,
        "provider": args.provider,
        "runtime_id": args.runtime_id,
        "cost": cost_metadata(args),
        "checkpoint_files": checkpoint_files(run_dir),
        "resumable_snapshots": resumable_snapshot_files(run_dir),
        "last_metrics": last_results_row(run_dir),
    }
    write_json_atomic(Path(run_dir) / "checkpoint_manifest.json", manifest)
    return manifest


def read_yaml(path):
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def staging_destination(archive_sha256, staging_root=Path("/tmp")):
    return Path(staging_root) / f"smoke-fire-pyro-sdis-{archive_sha256}"


def staging_marker_path(destination):
    return Path(destination) / ".stage.json"


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
        if relative.as_posix() == ".stage.json":
            continue
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
        "dataset_yaml_sha256": sha256_file(dataset_yaml),
    }


def cache_marker_matches(marker, archive_sha256, archive_manifest_sha256, identity, source_data_sha256, counts, dataset_yaml_path):
    if not isinstance(marker, dict):
        return False
    if marker.get("staging_schema_version") != STAGING_SCHEMA_VERSION:
        return False
    if marker.get("archive_sha256") != archive_sha256:
        return False
    if marker.get("archive_manifest_sha256") != archive_manifest_sha256:
        return False
    if marker.get("dataset_snapshot_sha256") != identity["dataset_snapshot_sha256"]:
        return False
    if marker.get("source_dataset_yaml_sha256") != source_data_sha256:
        return False
    if marker.get("counts") != counts:
        return False
    if marker.get("dataset_yaml_sha256") != sha256_file(dataset_yaml_path):
        return False
    return True


def stage_dataset_archive(data_path, identity, archive_path, manifest_path, max_seconds, staging_root=Path("/tmp"), force_rebuild=False):
    import yaml

    archive_path = Path(archive_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    data_path = Path(data_path).resolve()
    if not archive_path.is_file() or not manifest_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("dataset archive, manifest, or dataset.yaml missing")
    archive_manifest = read_json(manifest_path)
    if archive_manifest.get("dataset_snapshot_sha256") != identity["dataset_snapshot_sha256"]:
        raise ValueError("dataset archive snapshot mismatch")
    expected_archive_sha256 = archive_manifest.get("archive_sha256")
    expected_archive_files = archive_manifest.get("files")
    if not isinstance(expected_archive_sha256, str) or len(expected_archive_sha256) != 64:
        raise ValueError("dataset archive manifest has invalid archive hash")
    if not isinstance(expected_archive_files, int) or expected_archive_files <= 0:
        raise ValueError("dataset archive manifest has invalid file count")
    source_archive_sha256 = sha256_file(archive_path)
    if source_archive_sha256 != expected_archive_sha256:
        raise ValueError("dataset archive hash mismatch")
    source_data = read_yaml(data_path)
    if not isinstance(source_data, dict):
        raise ValueError("source dataset.yaml is not a mapping")
    archive_manifest_sha256 = sha256_file(manifest_path)
    source_data_sha256 = sha256_file(data_path)
    destination = staging_destination(expected_archive_sha256, staging_root)
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
            marker_path = staging_marker_path(destination)
            try:
                cache_counts = validate_staged_dataset(destination, source_data, destination, expected_archive_files)
                marker = read_json(marker_path)
                cache_valid = cache_marker_matches(
                    marker,
                    expected_archive_sha256,
                    archive_manifest_sha256,
                    identity,
                    source_data_sha256,
                    cache_counts,
                    destination / "dataset.yaml",
                )
            except Exception:
                cache_valid = False
            if cache_valid and not force_rebuild:
                return destination / "dataset.yaml", {
                    "source": str(archive_path),
                    "manifest": str(manifest_path),
                    "destination": str(destination),
                    "mode": "archive_cached",
                    "counts": cache_counts,
                    "archive_sha256": expected_archive_sha256,
                    "dataset_yaml_sha256": cache_counts["dataset_yaml_sha256"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            remove_path(destination)
        unique = uuid.uuid4().hex
        temporary_directory = destination.parent / f"{staging_temporary_prefix(destination)}{unique}"
        temporary_archive = destination.parent / f".{destination.name}.archive-{unique}.tar"
        temporary_directory.mkdir()
        digest = hashlib.sha256()
        copied_bytes = 0
        with archive_path.open("rb") as source, temporary_archive.open("xb") as target:
            while True:
                if max_seconds and time.perf_counter() - started >= max_seconds:
                    raise TimeoutError(f"dataset archive staging exceeded {max_seconds}s")
                chunk = source.read(16 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                copied_bytes += len(chunk)
        if digest.hexdigest() != expected_archive_sha256:
            raise ValueError("dataset archive hash mismatch while copying to SSD")
        extract_started = time.perf_counter()
        with tarfile.open(temporary_archive, "r") as archive:
            archive_file_count = validate_archive_members(archive)
            if archive_file_count != expected_archive_files:
                raise ValueError(f"archive file count mismatch: expected {expected_archive_files}, found {archive_file_count}")
            archive.extractall(temporary_directory, filter="data")
        normalized_empty_labels = normalize_empty_label_files(temporary_directory)
        (temporary_directory / "dataset.yaml").write_text(
            yaml.safe_dump(staged_dataset_data(source_data, destination), sort_keys=False),
            encoding="utf-8",
        )
        counts = validate_staged_dataset(temporary_directory, source_data, destination, expected_archive_files)
        marker = {
            "staging_schema_version": STAGING_SCHEMA_VERSION,
            "archive_sha256": expected_archive_sha256,
            "archive_manifest_sha256": archive_manifest_sha256,
            "dataset_snapshot_sha256": identity["dataset_snapshot_sha256"],
            "source_dataset_yaml_sha256": source_data_sha256,
            "counts": counts,
            "dataset_yaml_sha256": counts["dataset_yaml_sha256"],
            "created_at_utc": utc_now(),
        }
        write_json_atomic(staging_marker_path(temporary_directory), marker)
        temporary_directory.replace(destination)
        temporary_directory = None
        return destination / "dataset.yaml", {
            "source": str(archive_path),
            "manifest": str(manifest_path),
            "destination": str(destination),
            "mode": "archive",
            "bytes": copied_bytes,
            "copy_seconds": extract_started - started,
            "extract_seconds": time.perf_counter() - extract_started,
            "counts": counts,
            "normalized_empty_labels": normalized_empty_labels,
            "archive_sha256": expected_archive_sha256,
            "dataset_yaml_sha256": counts["dataset_yaml_sha256"],
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


def stage_dataset(data_path, identity, max_seconds, archive_path=None, archive_manifest_path=None):
    if archive_path or archive_manifest_path:
        if not archive_path or not archive_manifest_path:
            raise ValueError("dataset archive path and manifest must be provided together")
        return stage_dataset_archive(data_path, identity, archive_path, archive_manifest_path, max_seconds)
    import yaml

    source_data = read_yaml(data_path)
    source_root = Path(source_data["path"])
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    stage_name = f"smoke-fire-pyro-sdis-{identity['dataset_snapshot_sha256'][:16]}"
    destination = Path("/tmp") / stage_name
    marker_path = destination / ".stage.json"
    started = time.perf_counter()
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker.get("dataset_snapshot_sha256") == identity["dataset_snapshot_sha256"]:
            return destination / "dataset.yaml", {
                "source": str(source_root),
                "destination": str(destination),
                "reused": True,
                "elapsed_seconds": time.perf_counter() - started,
            }
        shutil.rmtree(destination)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    files = 0
    copied_bytes = 0
    try:
        for source_path in source_root.rglob("*"):
            if max_seconds and time.perf_counter() - started >= max_seconds:
                raise TimeoutError(f"dataset staging exceeded {max_seconds}s")
            relative = source_path.relative_to(source_root)
            if source_path.suffix == ".cache":
                continue
            destination_path = temporary / relative
            if source_path.is_dir():
                destination_path.mkdir(parents=True, exist_ok=True)
                continue
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            files += 1
            copied_bytes += source_path.stat().st_size
        staged_data = dict(source_data)
        staged_data["path"] = str(destination)
        (temporary / "dataset.yaml").write_text(yaml.safe_dump(staged_data, sort_keys=False), encoding="utf-8")
        marker = {
            "dataset_snapshot_sha256": identity["dataset_snapshot_sha256"],
            "source": str(source_root),
            "created_at_utc": utc_now(),
        }
        write_json_atomic(temporary / ".stage.json", marker)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / "dataset.yaml", {
        "source": str(source_root),
        "destination": str(destination),
        "reused": False,
        "files": files,
        "bytes": copied_bytes,
        "elapsed_seconds": time.perf_counter() - started,
    }


def stage_smoke_dataset(data_path, identity, run_name, smoke_batches, max_seconds):
    import yaml

    source_data = read_yaml(data_path)
    source_root = Path(source_data["path"])
    source_images = source_root / source_data["train"]
    source_labels = source_root / "labels" / "train"
    images = sorted(path for path in source_images.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) < smoke_batches:
        raise ValueError(f"smoke dataset needs {smoke_batches} images, found {len(images)}")
    destination = Path("/tmp") / f"smoke-fire-smoke-{identity['dataset_snapshot_sha256'][:12]}-{run_name}"
    if destination.exists():
        shutil.rmtree(destination)
    started = time.perf_counter()
    selected = images[:smoke_batches]
    files = 0
    copied_bytes = 0
    try:
        for image_path in selected:
            if max_seconds and time.perf_counter() - started >= max_seconds:
                raise TimeoutError(f"smoke dataset staging exceeded {max_seconds}s")
            image_target = destination / "images" / "train" / image_path.name
            image_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, image_target)
            files += 1
            copied_bytes += image_path.stat().st_size
            label_path = source_labels / f"{image_path.stem}.txt"
            if label_path.exists():
                label_target = destination / "labels" / "train" / label_path.name
                label_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label_path, label_target)
                files += 1
                copied_bytes += label_path.stat().st_size
        staged_data = {
            "path": str(destination),
            "train": "images/train",
            "val": "images/train",
            "names": source_data["names"],
        }
        dataset_path = destination / "dataset.yaml"
        dataset_path.write_text(yaml.safe_dump(staged_data, sort_keys=False), encoding="utf-8")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return dataset_path, {
        "source": str(source_root),
        "destination": str(destination),
        "mode": "smoke_subset",
        "files": files,
        "bytes": copied_bytes,
        "elapsed_seconds": time.perf_counter() - started,
    }


def prepare_smoke_dataset(data_path, run_dir, smoke_batches):
    import yaml

    source = read_yaml(data_path)
    root = Path(source["path"])
    train_value = Path(source["train"])
    train_dir = train_value if train_value.is_absolute() else root / train_value
    images = sorted(path for path in train_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) < smoke_batches:
        raise ValueError(f"smoke dataset needs {smoke_batches} images, found {len(images)}")
    smoke_dir = Path(run_dir) / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    list_path = smoke_dir / "train.txt"
    list_path.write_text("\n".join(str(path) for path in images[:smoke_batches]) + "\n", encoding="utf-8")
    smoke_data = {"path": str(root), "train": str(list_path), "val": str(list_path), "names": source["names"]}
    yaml_path = smoke_dir / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(smoke_data, sort_keys=False), encoding="utf-8")
    return yaml_path


def prepare_resume_gate_subset(data_path, run_dir, subset_images, seed):
    import yaml

    if not 256 <= subset_images <= 512:
        raise ValueError("resume gate subset must contain 256..512 images")
    source = read_yaml(data_path)
    root = Path(source["path"])
    train_value = Path(source["train"])
    train_dir = train_value if train_value.is_absolute() else root / train_value
    label_dir = root / "labels" / "train"
    images = sorted(path for path in train_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    empty = []
    positive = []
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        if label_path.read_text(encoding="utf-8").strip():
            positive.append(image_path)
        else:
            empty.append(image_path)
    empty_count = subset_images // 2
    positive_count = subset_images - empty_count
    if len(empty) < empty_count or len(positive) < positive_count:
        raise ValueError("resume gate subset lacks positive or empty-label images")
    generator = random.Random(seed)
    selected = sorted(generator.sample(empty, empty_count) + generator.sample(positive, positive_count))
    subset_dir = Path(run_dir) / "resume_gate_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    list_path = subset_dir / "train.txt"
    list_path.write_text("\n".join(str(path) for path in selected) + "\n", encoding="utf-8")
    subset_hash = sha256_file(list_path)
    dataset = {
        "path": str(root),
        "train": str(list_path),
        "val": str(list_path),
        "names": source["names"],
    }
    yaml_path = subset_dir / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    return yaml_path, {
        "schema": "smoke-fire-resume-gate-subset-v1",
        "images": subset_images,
        "positive_images": positive_count,
        "empty_label_images": empty_count,
        "seed": seed,
        "selection_sha256": subset_hash,
        "list_path": str(list_path),
    }


def run_smoke(args, run_dir, runtime, data_path, dataset_stage):
    import torch
    from ultralytics import YOLO

    smoke_data = prepare_smoke_dataset(data_path, run_dir, args.smoke_batches)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = YOLO(args.model)
    model.train(
        data=str(smoke_data),
        epochs=1,
        imgsz=args.imgsz,
        batch=1,
        patience=0,
        seed=args.seed,
        workers=args.workers,
        device=args.device,
        amp=not args.no_amp,
        save=False,
        val=False,
        plots=False,
        project=str(Path(run_dir) / "smoke"),
        name="run",
        exist_ok=True,
    )
    elapsed = time.perf_counter() - started
    report = {
        "schema": "smoke-fire-training-smoke-v1",
        "created_at_utc": utc_now(),
        "requested_batches": args.smoke_batches,
        "batch": 1,
        "imgsz": args.imgsz,
        "elapsed_seconds": elapsed,
        "throughput_images_per_second": args.smoke_batches / elapsed if elapsed else None,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "dataset_stage": dataset_stage,
        "runtime": runtime,
    }
    write_json_atomic(Path(run_dir) / "smoke_report.json", report)
    return report


def run_training(args, sync_callback=None):
    enforce_cost_gate(args)
    if not (args.model == "yolo26x.pt" or args.model.endswith(".pt")):
        raise ValueError("only official yolo26x.pt or .pt checkpoints are allowed for step 13.c")
    if args.imgsz != 1280 or args.epochs != 20 or args.patience != 5 or args.seed != 20260707:
        raise ValueError("step 13.c fixed training config mismatch")
    if args.save_period != 1:
        raise ValueError("step 13.c requires save_period=1")
    if not args.smoke_only and args.batch != 4:
        raise ValueError("production and resume-gate batch must equal 4")
    runtime = runtime_metadata()
    if not runtime["cuda_available"]:
        raise RuntimeError("GPU is required for step 13.c")
    identity = verified_dataset_identity(args)
    config = material_config(args, identity)
    run_dir = Path(args.project).resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke_only and not args.smoke_use_archive:
        runtime_data_path, dataset_stage = stage_smoke_dataset(
            Path(args.data).resolve(),
            identity,
            args.run_name,
            args.smoke_batches,
            args.dataset_stage_max_seconds,
        )
    else:
        runtime_data_path, dataset_stage = stage_dataset(
            Path(args.data).resolve(),
            identity,
            args.dataset_stage_max_seconds,
            args.dataset_archive,
            args.dataset_archive_manifest,
        )
    resume_gate_subset = None
    if args.subset_images:
        if args.smoke_only:
            raise ValueError("resume gate subset cannot combine with smoke mode")
        runtime_data_path, resume_gate_subset = prepare_resume_gate_subset(
            runtime_data_path,
            run_dir,
            args.subset_images,
            args.seed,
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
        "resume_gate_subset": resume_gate_subset,
        "cost": cost_metadata(args),
    })
    if args.smoke_only:
        report = run_smoke(args, run_dir, runtime, runtime_data_path, dataset_stage)
        if sync_callback:
            sync_callback()
        return {"mode": "smoke", "run_dir": str(run_dir), "report": report}

    resume_manifest = validate_resume(args, run_dir, config)
    previous_epoch = int(resume_manifest.get("completed_epoch", 0)) if resume_manifest else 0
    started = time.monotonic()
    stop_reason = "completed"

    def on_model_save(trainer):
        nonlocal stop_reason
        completed_epoch = int(trainer.epoch) + 1
        resume_path = run_dir / "weights" / "last_resume.pt"
        snapshot_path = run_dir / "weights" / f"epoch_{completed_epoch:03d}_resume.pt"
        write_resumable_checkpoint(
            run_dir / "weights" / "last.pt",
            resume_path,
            snapshot_path,
            trainer,
            identity,
        )
        if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
            trainer.stop = True
            stop_reason = "max_runtime_seconds"
        if args.max_epochs_per_invocation and completed_epoch - previous_epoch >= args.max_epochs_per_invocation:
            trainer.stop = True
            stop_reason = "max_epochs_per_invocation"
        write_checkpoint_manifest(run_dir, args, identity, config, runtime, completed_epoch, stop_reason)
        if sync_callback:
            sync_callback()

    from ultralytics import YOLO

    last_path = run_dir / "weights" / "last_resume.pt"
    trainer_class = None
    if args.subset_images:
        from ultralytics.models.yolo.detect.train import DetectionTrainer

        class ResumeGateTrainer(DetectionTrainer):
            def validate(self):
                return {}, 0.0

            def final_eval(self):
                return

        trainer_class = ResumeGateTrainer
    if resume_manifest:
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
    manifest = previous_manifest(run_dir)
    completed_epoch = int(manifest.get("completed_epoch", previous_epoch)) if manifest else previous_epoch
    if completed_epoch <= previous_epoch or not last_path.is_file():
        raise RuntimeError("training ended without a newly verified resumable checkpoint")
    checkpoint_resume_state(last_path)
    final = write_checkpoint_manifest(run_dir, args, identity, config, runtime, completed_epoch, stop_reason)
    if sync_callback:
        sync_callback()
    return {
        "mode": "train",
        "run_dir": str(run_dir),
        "started_epoch_one_based": previous_epoch + 1,
        "manifest": final,
    }


def main(argv=None):
    result = run_training(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
