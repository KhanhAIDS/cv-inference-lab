import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_dfire_yolo26x as dfire

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIM_DATASET = REPO_ROOT / "datasets" / "smoke_fire_detection" / "dfire-relabeled-rfdetr"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "smoke_fire_detection" / "runs" / "rfdetr_large_dfire_relabeled_gb10"


def parse_args():
    parser = argparse.ArgumentParser(description="RF-DETR-L fine-tune on relabeled D-Fire (companion run, not track 13f)")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", default=None, help="path to checkpoint_<N>.ckpt to resume from")
    return parser.parse_args()


def validate_protocol(args):
    if args.resolution != 640 or args.epochs != 20 or args.patience != 0 or args.seed != 20260707:
        raise ValueError("D-Fire RF-DETR protocol is locked: resolution=640, epochs=20, patience=0, seed=20260707")
    if args.batch != 4 or args.grad_accum != 4:
        raise ValueError("D-Fire RF-DETR protocol is locked: batch=4, grad_accum=4 (effective 16)")
    if args.val_ratio != 0.1:
        raise ValueError("D-Fire RF-DETR protocol is locked: val_ratio=0.1")


def link_split(name, images):
    images_dir = SHIM_DATASET / name / "images"
    labels_dir = SHIM_DATASET / name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_source_dir = dfire.DATA_ROOT / ("train" if name != "test" else "test") / "labels"
    for image_path in images:
        image_link = images_dir / image_path.name
        if not image_link.exists():
            image_link.symlink_to(image_path)
        label_source = labels_source_dir / f"{image_path.stem}.txt"
        label_link = labels_dir / f"{image_path.stem}.txt"
        if not label_link.exists():
            if label_source.exists():
                label_link.symlink_to(label_source)
            else:
                label_link.write_text("\n")


def ensure_shim_dataset(seed, val_ratio):
    dfire.build_split(seed, val_ratio)
    train_images = [Path(line) for line in (dfire.SPLIT_DIR / "train.txt").read_text().splitlines() if line]
    val_images = [Path(line) for line in (dfire.SPLIT_DIR / "val.txt").read_text().splitlines() if line]
    test_images = [Path(line) for line in (dfire.SPLIT_DIR / "test.txt").read_text().splitlines() if line]
    link_split("train", train_images)
    link_split("valid", val_images)
    link_split("test", test_images)
    data_yaml = SHIM_DATASET / "data.yaml"
    content = "nc: 2\nnames:\n  0: smoke\n  1: fire\n"
    if not data_yaml.exists() or data_yaml.read_text() != content:
        data_yaml.write_text(content)


def verify_resume_checkpoint(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    missing = [key for key in ("state_dict", "optimizer_states", "lr_schedulers", "epoch", "global_step") if key not in checkpoint]
    if missing:
        raise ValueError(f"resume checkpoint missing keys {missing}: {path}")
    print(f"resume checkpoint verified: epoch={checkpoint['epoch']} global_step={checkpoint['global_step']}")


def main():
    args = parse_args()
    validate_protocol(args)
    ensure_shim_dataset(args.seed, args.val_ratio)
    if args.resume:
        verify_resume_checkpoint(args.resume)

    from rfdetr import RFDETRLarge

    model = RFDETRLarge(resolution=args.resolution)
    model.train(
        dataset_dir=str(SHIM_DATASET),
        dataset_file="yolo",
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        num_workers=args.num_workers,
        checkpoint_interval=1,
        seed=args.seed,
        early_stopping=False,
        tensorboard=False,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    print("TRAIN_FINISHED")


if __name__ == "__main__":
    main()
