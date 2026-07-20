import json
import re
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent / "demo_results"
FIGLIB_FRAME_PATTERN = re.compile(r"^(\d+)_([+-]\d+)$")
STATE = {"device": "cuda" if torch.cuda.is_available() else "cpu"}


def is_oom_error(exc):
    return "out of memory" in str(exc).lower()

MODELS = [
    {
        "name": "dfire_yolo26n_baseline",
        "family": "yolo",
        "weights": REPO_ROOT / "artifacts/smoke_fire_detection/runs/dfire_yolo26n_baseline_full_vram/weights/best.pt",
        "imgsz": 640,
        "training_status": "hoan tat (E0 baseline) - D-Fire test mAP50-95=0.404, mAP50=0.684",
    },
    {
        "name": "yolo26x_pyro_sdis",
        "family": "yolo",
        "weights": REPO_ROOT / "artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis/weights/best.pt",
        "imgsz": 1280,
        "training_status": "hoan tat 20/20 - Pyro-SDIS val best mAP50-95=0.4949",
    },
    {
        "name": "rfdetr_large_pyro_sdis",
        "family": "rfdetr",
        "weights": REPO_ROOT
        / "artifacts/smoke_fire_detection/runs/rfdetr_large_pyro_sdis/checkpoint_best_total.pth",
        "imgsz": 1280,
        "training_status": "hoan tat 20/20 - RF-DETR final best-total",
    },
]

BOX_ANNOTATOR = sv.BoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


def figlib_sample_frames(count=3, target_offset=600):
    root = REPO_ROOT / "datasets/smoke_fire_detection/FIgLib"
    picked = []
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = []
        for image_path in seq_dir.glob("*.jpg"):
            match = FIGLIB_FRAME_PATTERN.match(image_path.stem)
            if match:
                frames.append((int(match.group(2)), image_path))
        if not frames:
            continue
        frames.sort(key=lambda item: abs(item[0] - target_offset))
        picked.append(frames[0][1])
        if len(picked) >= count:
            break
    return picked


def sample_images():
    pyro_val = sorted((REPO_ROOT / "datasets/smoke_fire_detection/pyro-sdis-yolo/images/val").glob("*.jpg"))
    dfire_test = sorted((REPO_ROOT / "datasets/smoke_fire_detection/D-Fire/test/images").glob("*.jpg"))
    return {
        "pyro_sdis_val": pyro_val[:3],
        "dfire_test": dfire_test[:3],
        "figlib_sample": figlib_sample_frames(3),
    }


def load_model(spec):
    if spec["family"] == "yolo":
        return YOLO(str(spec["weights"]))
    from rfdetr import RFDETRLarge

    return RFDETRLarge(
        resolution=spec["imgsz"], num_classes=1, pretrain_weights=str(spec["weights"]), device=STATE["device"]
    )


def predict_yolo(model, image_path, imgsz):
    result = model.predict(str(image_path), imgsz=imgsz, conf=0.15, device=STATE["device"], verbose=False)[0]
    annotated = result.plot()
    boxes = result.boxes
    detections = []
    if boxes is not None:
        for class_id, confidence in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            detections.append({"class_name": result.names[int(class_id)], "confidence": float(confidence)})
    return annotated, detections


def predict_rfdetr(model, image_path):
    image = Image.open(image_path).convert("RGB")
    det = model.predict(image, threshold=0.15)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    class_names = list(det.data.get("class_name", []))
    confidences = list(det.confidence) if det.confidence is not None else []
    labels = [f"{name} {conf:.2f}" for name, conf in zip(class_names, confidences)]
    annotated = BOX_ANNOTATOR.annotate(frame.copy(), det)
    annotated = LABEL_ANNOTATOR.annotate(annotated, det, labels=labels)
    detections = [{"class_name": name, "confidence": float(conf)} for name, conf in zip(class_names, confidences)]
    return annotated, detections


def process_model(spec, images, summary):
    print(f"loading {spec['name']} ({spec['weights']}) on {STATE['device']}")
    model = load_model(spec)
    summary["models"][spec["name"]] = {
        "weights": str(spec["weights"].relative_to(REPO_ROOT)),
        "training_status": spec["training_status"],
        "device": STATE["device"],
    }
    for dataset_name, paths in images.items():
        for image_path in paths:
            if spec["family"] == "yolo":
                annotated, detections = predict_yolo(model, image_path, spec["imgsz"])
            else:
                annotated, detections = predict_rfdetr(model, image_path)
            out_path = OUT_DIR / dataset_name / spec["name"] / f"{image_path.stem}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), annotated)
            summary["datasets"].setdefault(dataset_name, {}).setdefault(spec["name"], {})[image_path.name] = detections
            print(f"  {dataset_name}/{image_path.name}: {len(detections)} detection(s)")
    del model
    if STATE["device"] == "cuda":
        torch.cuda.empty_cache()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = sample_images()
    summary = {"models": {}, "datasets": {}}

    for spec in MODELS:
        if not spec["weights"].exists():
            print(f"skip {spec['name']}: weights missing at {spec['weights']}")
            continue
        try:
            process_model(spec, images, summary)
        except RuntimeError as exc:
            if not is_oom_error(exc) or STATE["device"] == "cpu":
                raise
            print(f"  GPU out of memory (shared server contention) - falling back to CPU for {spec['name']}")
            torch.cuda.empty_cache()
            STATE["device"] = "cpu"
            process_model(spec, images, summary)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done: {OUT_DIR}")


if __name__ == "__main__":
    main()
