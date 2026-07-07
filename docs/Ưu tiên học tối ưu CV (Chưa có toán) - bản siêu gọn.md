# Ưu tiên học tối ưu CV — bản siêu gọn

- Scope: Computer Vision inference / deployment optimization.
- Không gồm toán tối ưu CV.
- Mục tiêu: học đúng thứ tự, tránh lan man.

---

## 0 — Nguyên tắc

- Benchmark trước, tối ưu sau.
- Accuracy parity trước, quantization sau.
- Biết bottleneck trước, đổi model sau.
- Không tin FPS trung bình; đo p50 / p95 / p99.
- Đo end-to-end, không chỉ model forward.
- Benchmark trên target device thật.
- Không học custom CUDA / compiler sâu quá sớm.

---

## 1 — Thứ tự học 80/20

1. Target constraints
2. Benchmark / profiling
3. Metric contract + golden test set
4. Export + parity test
5. Runtime target
6. FP16
7. INT8 PTQ nếu cần
8. Preprocess / postprocess optimization
9. Architecture selection nếu baseline không đạt
10. Dataset/training fix nếu accuracy yếu
11. Video/system optimization nếu có video
12. Serving/monitoring/rollback
13. Compression nâng cao nếu vẫn không đạt
14. Hardware-specific tuning nếu device rất gắt

---

## 2 — Target contract

### Cần chốt

- task: classification / detection / segmentation / pose / tracking / OCR / video analytics
- device: NVIDIA GPU / Jetson / Intel CPU-iGPU-NPU / Android / iOS-macOS / CPU-only / browser
- runtime: ONNX Runtime / TensorRT / OpenVINO / Core ML / LiteRT / ExecuTorch / Triton
- budget: resolution / batch / p95 latency / throughput / memory / power / thermal / model size / accuracy floor

### Output

```yaml
task:
device:
runtime:
resolution:
batch:
latency_budget_p95:
throughput_budget:
memory_budget:
accuracy_metric:
accuracy_floor:
deployment_mode:
```

---

## 3 — Benchmark contract

### Đo

- end-to-end latency
- model-only latency
- preprocess / postprocess latency
- p50 / p95 / p99
- throughput
- memory peak
- CPU/GPU utilization
- H2D/D2H transfer
- power/thermal
- accuracy parity

### Tách stage

- decode
- resize / normalize / layout convert
- model forward
- NMS / mask decode / tracking
- render / serialization nếu production cần

### Tool

- PyTorch Profiler
- Nsight Systems / Compute
- TensorRT `trtexec`
- ONNX Runtime profiling
- OpenVINO `benchmark_app`
- Triton Perf Analyzer
- `nvidia-smi dmon` / `tegrastats`
- Android Profiler / Xcode Instruments

### Protocol

- warmup: 50–100
- measure: 300–1000
- fixed batch / resolution
- realistic input
- eval mode
- no debug mode
- same target device

---

## 4 — Reproducibility + cold start

### Ghi lại

- OS / driver / CUDA / cuDNN
- TensorRT / PyTorch / ONNX opset / ONNX Runtime / OpenVINO
- JetPack / Core ML Tools / LiteRT nếu liên quan
- device SKU / power mode
- Docker image / lockfile
- model hash / artifact checksum
- exact benchmark command

### Cold-start metric

- model load time
- engine build / deserialize time
- first inference latency
- warm inference latency
- cache hit/miss
- memory after load

---

## 5 — Metric + golden set

### Metric

- classification: top-1 / top-5 / F1 / confusion matrix
- detection: mAP50 / mAP50-95 / precision / recall / small-medium-large AP
- segmentation: mIoU / Dice / boundary quality
- tracking: IDF1 / MOTA / ID switch / latency per stream
- OCR: CER / WER / exact match / field-level accuracy

### Golden set

- normal cases
- hard negatives
- blur / low-light / occlusion
- small objects / crowded scenes
- domain shift
- rare classes
- production-like samples

### Tolerance

- classification: top-k match / logit tolerance
- detection: class match / bbox IoU tolerance / score tolerance
- segmentation: mask IoU / boundary tolerance
- pose: keypoint pixel tolerance / OKS
- tracking: ID switch tolerance
- OCR: text diff / field diff

---

## 6 — Architecture selection

### Rule

- Edge/mobile: small / conv-friendly / static shape / few unsupported ops / simple postprocess
- Server GPU: TensorRT-friendly / batch tốt / throughput cao / dynamic batching
- CPU-only: model nhỏ / INT8 tốt / memory layout tốt / tránh attention nặng nếu không cần
- Video realtime: detector nhẹ / tracker / keyframe inference / frame skipping

### Candidate

- classification: MobileNet / EfficientNet / ConvNeXt / ViT / Swin / FastViT / MobileViT
- detection: YOLO family / RT-DETR / SSD / MobileDet / detector+tracker
- segmentation: UNet / DeepLab / SegFormer / Mask2Former / YOLO-seg

---

## 7 — Export + parity

### Path

- PyTorch → ONNX
- ONNX → TensorRT
- ONNX → OpenVINO
- ONNX Runtime EP
- Core ML
- LiteRT / TFLite
- ExecuTorch

### ONNX checklist

- `model.eval()`
- correct sample input / dtype / device
- `dynamo=True`
- `dynamic_shapes` nếu cần dynamic input
- `verify=True`
- `report=True` nếu debug
- `profile=True` nếu debug export
- `dump_exported_program=True` nếu cần inspect
- `external_data=True` nếu model lớn
- opset theo backend target

### Parity check

- output shape
- class score
- bbox coordinates
- mask values
- keypoints
- NMS result
- top-k prediction
- metric trên golden set

---

## 8 — Runtime optimization

### NVIDIA GPU

- TensorRT
- `trtexec`
- FP16 / INT8
- dynamic shape profile
- tactic / timing cache
- CUDA graph nếu phù hợp
- per-layer profiling

### Server GPU

- Triton
- dynamic batching
- concurrent model execution
- Perf Analyzer / Model Analyzer

### Intel CPU/iGPU/NPU

- OpenVINO
- `benchmark_app`
- latency vs throughput mode
- async infer
- thread / stream tuning
- INT8 / NNCF

### Cross-platform

- ONNX Runtime
- CPU / CUDA / TensorRT / OpenVINO / CoreML / DirectML / NNAPI / XNNPACK / QNN

### Mobile

- Android: LiteRT / TFLite / NNAPI / GPU delegate / QNN
- iOS/macOS: Core ML / ANE / Metal / static shape / packaging

---

## 9 — Fallback audit

### Kiểm tra

- unsupported ops
- provider placement
- CPU fallback
- TensorRT parser error
- runtime graph partition
- hidden CPU↔GPU copy
- I/O Binding / device tensor
- output pre-allocation

### Pass

- Biết op nào chạy CPU/GPU/NPU.
- Không có fallback âm thầm.
- Không có copy thừa lớn.

---

## 10 — Quantization

### Học

- FP32 → FP16
- FP16 → INT8
- PTQ / QAT
- calibration dataset
- per-tensor vs per-channel
- symmetric vs asymmetric
- QDQ vs QOperator
- activation clipping

### Thứ tự

1. FP16 trước.
2. INT8 PTQ sau.
3. Nếu metric tụt: tăng calibration quality / đổi calibration method / exclude sensitive layers / mixed precision.
4. Nếu vẫn tụt: QAT / distillation + QAT.

### Debug

- layer-wise sensitivity
- activation histogram
- outlier check
- mixed precision fallback
- calibration cache validation
- per-class / per-slice metric delta
- threshold retune

### Học sau

- FP8
- INT4 weight-only
- FP4
- block-wise quantization

---

## 11 — Preprocess / postprocess

### Optimize

- bỏ Python loop
- vectorize NumPy/Torch
- OpenCV optimized path
- GPU preprocess / GPU NMS nếu đáng
- async CPU/GPU
- pinned memory
- zero-copy camera buffer
- fuse resize + normalize nếu runtime hỗ trợ
- tránh copy thừa / CPU↔GPU ping-pong

### Input contract

- RGB/BGR
- uint8/float32
- scale 0–1 hay 0–255
- mean/std
- resize algorithm
- aspect ratio
- letterbox padding value
- EXIF orientation
- NCHW/NHWC

### Postprocess contract

- NMS class-aware/class-agnostic
- IoU threshold
- confidence threshold
- top-k before/after NMS
- bbox format: xyxy / xywh / cxcywh
- normalized vs pixel coords
- mask upsample method

---

## 12 — Dataset / training fix

### Đẩy lên sớm nếu

- baseline accuracy thấp
- FP/FN nhiều
- domain shift mạnh
- class imbalance nặng
- production data khác train data
- quantization làm lộ điểm yếu model

### Học

- data cleaning
- label noise
- hard negative mining
- class imbalance
- augmentation
- active learning
- synthetic data
- domain split
- leakage check
- error taxonomy
- retrain loop

---

## 13 — Compression nâng cao

### Học khi cần

- teacher-student distillation
- logits / feature distillation
- structured pruning
- channel pruning
- low-rank compression
- sparsity

### Không ưu tiên sớm

- unstructured pruning
- NAS
- custom CUDA
- TVM sâu
- compiler IR-level optimization

---

## 14 — Video / system

### Học

- frame skipping
- keyframe inference
- detector + tracker
- multi-stream batching
- queue / backpressure
- drop stale frame
- async pipeline
- decode pipeline
- RTSP latency
- stream reconnect

### Decode / timestamp

- OpenCV VideoCapture / FFmpeg / GStreamer / NVDEC / V4L2
- hardware decode yes/no
- capture / decode / inference / output timestamp
- end-to-end age
- dropped / stale frame count

---

## 15 — Hardware-specific tuning

### Học

- CUDA basics / Tensor Cores
- Jetson DLA
- ARM NEON
- AVX2 / AVX512
- Apple ANE
- Android NPU / DSP
- memory bandwidth
- CPU affinity / NUMA
- thermal throttling
- power mode

### Rule

- Cloud/server: học sau P0–P6.
- Jetson/mobile/CPU-only/power-budget gắt: đẩy lên sớm.

---

## 16 — Serving / production

### Học

- Triton hoặc lightweight API
- dynamic batching
- model warmup
- health / readiness / liveness
- timeout / fallback
- canary / rollback
- model versioning
- drift monitoring
- request tracing

### Metric

- p50 / p95 / p99 latency
- throughput
- timeout / crash / OOM rate
- GPU / CPU utilization
- queue time
- confidence drift
- input distribution drift
- accuracy proxy
- rollback count

### Load test

- steady load
- burst load
- overload
- soak test
- model reload during traffic
- bad input flood
- camera reconnect storm nếu video
- OOM recovery

---

## 17 — Release / governance

### Checklist

- model registry
- artifact versioning
- dataset version
- training commit
- eval report hash
- model owner
- rollback artifact
- release note
- image logging policy
- PII/privacy nếu có camera người thật
- dependency scan

---

## 18 — Decision tree

- Cloud NVIDIA GPU: benchmark → ONNX → TensorRT FP16 → INT8 nếu cần → Triton → dynamic batching → monitoring
- Edge GPU / Jetson: device benchmark → fixed power mode → TensorRT FP16/INT8 → model nhỏ hơn nếu cần → zero-copy → thermal profiling
- Mobile: mobile architecture → static shape → Core ML/LiteRT → FP16/INT8 → operator support → real-device battery/thermal test
- CPU-only: OpenVINO/ONNX Runtime → INT8 → thread/stream tuning → memory layout → vectorize pre/postprocess → model nhỏ hơn
- Realtime video: end-to-end benchmark → decode/model/postprocess split → detector+tracker → async queue → drop stale frame → monitor lag/backlog

---

## 19 — Skill priority

### Must-have

- benchmark đúng
- PyTorch profiling
- ONNX export/debug
- ONNX Runtime baseline
- TensorRT hoặc OpenVINO
- FP16
- INT8 PTQ
- preprocess/postprocess optimization
- accuracy parity
- device-real testing
- fallback audit
- reproducibility

### Should-have

- Triton serving
- dynamic batching
- QAT
- distillation
- video async pipeline
- calibration debugging
- hard negative mining
- mobile packaging nếu deploy mobile

### Learn later

- custom CUDA kernel
- TensorRT plugin
- TVM sâu
- compiler IR
- NAS
- FPGA
- distributed inference
- adversarial robustness sâu
- FP8/INT4/FP4 sâu nếu target chưa cần

---

## 20 — Anti-patterns

- Benchmark chỉ FPS trung bình.
- Không đo p95/p99.
- Không warmup.
- Không tách preprocess/model/postprocess.
- Không ghi environment/version.
- Export ONNX xong không test parity.
- Không audit fallback.
- INT8 không calibration set đại diện.
- Chọn architecture theo benchmark public, không theo data thật.
- Tối ưu trên laptop, deploy trên device khác.
- Dùng dynamic shape lung tung làm engine chậm.
- Bỏ qua CPU↔GPU copy.
- Dùng Python loop trong postprocess.
- Sai RGB/BGR/normalize/letterbox nhưng đổ lỗi runtime.
- Custom CUDA quá sớm.
- Serving không load test.
- Không có rollback.
- Không monitor drift.

---

## 21 — Làm ngay

### Làm

- Chọn 1 task duy nhất.
- Chọn 1 device thật.
- Chọn 1 model baseline.
- Tạo 100–500 ảnh golden set.
- Viết benchmark end-to-end.

### Không làm

- Không học custom CUDA.
- Không học TVM.
- Không học pruning.
- Không học NAS.
- Không đổi 5 model cùng lúc.
- Không INT8 khi chưa có FP16 + parity.
