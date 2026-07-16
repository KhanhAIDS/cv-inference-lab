import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET = REPO_ROOT / "datasets" / "smoke_fire_detection" / "pyro-sdis-yolo"
SHIM_DATASET = REPO_ROOT / "datasets" / "smoke_fire_detection" / "pyro-sdis-yolo-rfdetr"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "smoke_fire_detection" / "runs" / "rfdetr_large_pyro_sdis_gb10"


def parse_args():
    parser = argparse.ArgumentParser(description="RF-DETR-L fine-tune on Pyro-SDIS (step 13f)")
    parser.add_argument("--resolution", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", default=None, help="path to checkpoint_<N>.ckpt to resume from")
    return parser.parse_args()


def validate_protocol(args):
    if args.resolution != 1280 or args.epochs != 20 or args.patience != 0 or args.seed != 20260707:
        raise ValueError("step 13f protocol is locked: resolution=1280, epochs=20, patience=0, seed=20260707")
    if args.batch != 4 or args.grad_accum != 4:
        raise ValueError("step 13f protocol is locked: batch=4, grad_accum=4 (effective 16)")


def ensure_shim_dataset():
    if not SOURCE_DATASET.is_dir():
        raise FileNotFoundError(f"source dataset missing: {SOURCE_DATASET}")
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            src = SOURCE_DATASET / sub / split
            if not src.is_dir():
                raise FileNotFoundError(f"source split missing: {src}")
            dst = SHIM_DATASET / split / sub
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                if dst.resolve() != src.resolve():
                    raise ValueError(f"shim symlink points elsewhere: {dst} -> {dst.resolve()}")
            elif dst.exists():
                raise ValueError(f"shim path exists and is not a symlink: {dst}")
            else:
                dst.symlink_to(src, target_is_directory=True)
    data_yaml = SHIM_DATASET / "data.yaml"
    content = "nc: 1\nnames:\n  0: smoke\n"
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
    ensure_shim_dataset()
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
