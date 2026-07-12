import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path


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
    parser.add_argument("--batch", type=int, default=-1)
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
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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
        raise ValueError("resume blocked: last.pt hash mismatch")
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
        "last_metrics": last_results_row(run_dir),
    }
    write_json_atomic(Path(run_dir) / "checkpoint_manifest.json", manifest)
    return manifest


def read_yaml(path):
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stage_dataset_archive(data_path, identity, archive_path, manifest_path, max_seconds):
    import yaml

    archive_path = Path(archive_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not archive_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("dataset archive or manifest missing")
    archive_manifest = read_json(manifest_path)
    if archive_manifest.get("dataset_snapshot_sha256") != identity["dataset_snapshot_sha256"]:
        raise ValueError("dataset archive snapshot mismatch")
    expected_archive_sha256 = archive_manifest.get("archive_sha256")
    if not expected_archive_sha256:
        raise ValueError("dataset archive manifest has no archive hash")
    source_data = read_yaml(data_path)
    destination = Path("/tmp") / f"smoke-fire-pyro-sdis-{identity['dataset_snapshot_sha256'][:16]}"
    temporary_archive = destination.with_suffix(".tar.tmp")
    if destination.exists():
        shutil.rmtree(destination)
    temporary_archive.unlink(missing_ok=True)
    started = time.perf_counter()
    digest = hashlib.sha256()
    copied_bytes = 0
    try:
        with archive_path.open("rb") as source, temporary_archive.open("wb") as target:
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
            raise ValueError("dataset archive hash mismatch")
        extract_started = time.perf_counter()
        destination.mkdir(parents=True)
        with tarfile.open(temporary_archive, "r") as archive:
            destination_root = destination.resolve()
            for member in archive.getmembers():
                if not (destination_root / member.name).resolve().is_relative_to(destination_root):
                    raise ValueError("unsafe dataset archive member")
            archive.extractall(destination, filter="data")
        staged_data = dict(source_data)
        staged_data["path"] = str(destination)
        (destination / "dataset.yaml").write_text(yaml.safe_dump(staged_data, sort_keys=False), encoding="utf-8")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        temporary_archive.unlink(missing_ok=True)
    return destination / "dataset.yaml", {
        "source": str(archive_path),
        "manifest": str(manifest_path),
        "destination": str(destination),
        "mode": "archive",
        "bytes": copied_bytes,
        "copy_seconds": extract_started - started if "extract_started" in locals() else None,
        "extract_seconds": time.perf_counter() - extract_started if "extract_started" in locals() else None,
    }


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
    if args.model != "yolo26x.pt":
        raise ValueError("only official yolo26x.pt is allowed for step 13.c")
    if args.imgsz != 1280 or args.epochs != 20 or args.patience != 5 or args.seed != 20260707:
        raise ValueError("step 13.c fixed training config mismatch")
    if args.save_period != 1:
        raise ValueError("step 13.c requires save_period=1")
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
        temporary_resume_path = resume_path.with_suffix(".pt.tmp")
        shutil.copy2(run_dir / "weights" / "last.pt", temporary_resume_path)
        temporary_resume_path.replace(resume_path)
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
    if resume_manifest:
        model = YOLO(str(last_path))
        train_args = {"resume": str(last_path), "data": str(runtime_data_path)}
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
    model.add_callback("on_model_save", on_model_save)
    model.train(**train_args)
    manifest = previous_manifest(run_dir)
    completed_epoch = int(manifest.get("completed_epoch", previous_epoch)) if manifest else previous_epoch
    final = write_checkpoint_manifest(run_dir, args, identity, config, runtime, completed_epoch, stop_reason)
    if sync_callback:
        sync_callback()
    return {"mode": "train", "run_dir": str(run_dir), "manifest": final}


def main(argv=None):
    result = run_training(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
