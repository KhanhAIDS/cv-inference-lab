# D-Fire Baseline Report

## 1. Dataset Audit

*This section will be populated after running `audit_dfire.py`*

**Summary of D-Fire Local Data:**
- Training Images: 17221
- Training Labels: 17221 (7835 empty/negative)
- Test Images: 4306
- Test Labels: 4306 (2005 empty/negative)
- Class Distribution: `0: smoke` (11942 total), `1: fire` (14645 total)

## 2. Train Configuration

**Command used:**
```bash
python -m smoke_fire.train.train_yolo --data data/dfire/dfire.yaml --model yolov8n.pt --run-name dfire_yolo_baseline
```

**Key Parameters:**
- **Model:** YOLOv8n (default)
- **Image Size:** 640
- **Epochs:** 100 (with early stopping patience=20)
- **Batch Size:** Auto
- **Seed:** 20260707
- **Pretrained:** True

## 3. Evaluation Metrics

*This section will be populated after running `eval_yolo.py` on the test split.*

**Overall Box Metrics:**
- **mAP50:** TBD
- **mAP50-95:** TBD
- **Precision:** TBD
- **Recall:** TBD

**Per-Class Metrics:**
- **Smoke:** TBD
- **Fire:** TBD

**Frame-Level Proxies (Derived):**
- Image FP: TBD
- Image FN: TBD

## 4. Latency Profiling

*This section will be populated after running `profile_yolo.py`*

**Hardware Target:** TBD (Local CPU / Modal GPU L4)

- **Warmup Frames:** 50
- **Measured Frames:** 500
- **Mean Latency:** TBD ms
- **p50 Latency:** TBD ms
- **p95 Latency:** TBD ms
- **FPS:** TBD

## 5. Failure Cases (Hard Negatives)

*Top false positives extracted from test negatives using `extract_hard_negatives.py`*

- **Total False Positive Images:** TBD
- **Top Confidence FP Example Path:** `artifacts/dfire/hard_negative_crops/fp_000_...`
- *(Embed sample crops here when available)*

## 6. Next Steps

- **Temporal Verifier Preparation:** Tải và chuẩn hóa dataset video (PyroNear / FIgLib) cho M2 temporal verifier.
- **Note:** Không tải dataset video trước khi report này hoàn thiện 100% (audit, metrics, latency thực tế). `false alarms/hour` và `time-to-detection` sẽ bắt đầu đo ở M2.
