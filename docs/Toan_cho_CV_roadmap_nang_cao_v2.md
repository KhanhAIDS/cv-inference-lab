# Toán cho Computer Vision Model Optimization — Roadmap gọn, nâng cao, thực dụng

## Mục tiêu

- Học toán để tối ưu mô hình CV khi triển khai.
- Không học toán kiểu đại học tuyến tính.
- Không chia roadmap theo classification / detection / segmentation / video.
- Tất cả model CV cuối cùng đều quay về cùng vài lớp toán:
  - dữ liệu / phân phối
  - tensor / layer
  - loss / metric
  - gradient / optimizer
  - architecture
  - precision
  - quantization / compression
  - complexity / runtime

## Nguyên tắc lọc kiến thức

- Chỉ học thứ giúp trả lời câu hỏi tối ưu:
  - model tốt thật hay metric nhiễu?
  - bottleneck nằm ở compute, memory, I/O, preprocessing hay postprocessing?
  - layer nào nặng nhất?
  - shape/layout có đúng không?
  - FP16/BF16/INT8 có làm lệch output không?
  - quantization tụt accuracy vì tensor/layer nào?
  - model lớn hơn có đáng latency/memory không?
  - threshold có đáng tin không?

---

# Roadmap duy nhất

## Stage 0 — Benchmark statistics

### Học

- mean / median
- variance / std
- p50 / p95 / p99
- confidence interval
- bootstrap cơ bản
- sample size
- outlier
- warmup
- train/val/test leakage
- distribution shift

### Dùng để

- biết cải thiện thật hay nhiễu
- so sánh model A/B
- đo latency đúng
- tránh kết luận từ vài lần chạy
- biết validation set quá nhỏ

### Output bắt buộc

- bảng metric: mean, std, CI
- bảng latency: p50, p95, p99
- số samples / số runs / warmup
- quyết định: kết quả đủ tin hay chưa

### Pass

- tự viết bootstrap CI cho metric
- tự giải thích vì sao accuracy tăng 0.3% có thể vô nghĩa
- tự phát hiện benchmark thiếu warmup hoặc sample size quá nhỏ

---

## Stage 1 — Tensor algebra + shape literacy

### Học

- scalar / vector / matrix / tensor
- shape / rank / axis
- broadcasting
- reshape / view / transpose / permute
- contiguous vs non-contiguous
- stride memory
- NCHW / NHWC
- batched matmul
- einsum
- norm / cosine similarity

### Dùng để

- debug model summary
- debug ONNX export
- debug attention tensor
- tính activation memory
- hiểu layout làm chậm

### Output bắt buộc

- shape table từng stage
- activation memory từng stage
- params từng block
- layout: input/output là NCHW hay NHWC

### Pass

- nhìn model summary tính được shape/params
- biết tensor nào chiếm memory lớn nhất
- biết permute/view sai có thể tạo copy hoặc non-contiguous tensor

---

## Stage 2 — Convolution / attention / layer math

### Học

- convolution output size
- kernel / stride / padding / dilation
- pooling
- receptive field
- grouped conv
- depthwise separable conv
- pointwise conv
- transposed conv
- patch embedding
- self-attention
- attention O(N²)
- MLP block
- residual connection
- normalization

### Dùng để

- hiểu CNN, ViT, hybrid model
- hiểu resolution tăng làm cost tăng mạnh
- hiểu patch size trade-off
- hiểu vì sao attention nặng theo token count

### Output bắt buộc

- công thức output size conv
- receptive field sơ bộ
- FLOPs conv block
- attention memory theo số token
- so sánh resolution / patch size / channels

### Pass

- tự tính Conv2D output shape
- tự tính cost conv: `H × W × Cin × Cout × K²`
- tự tính attention cost: `N² × d`
- giải thích vì sao 224→384 không tăng tuyến tính

---

## Stage 3 — Cost model / complexity

### Học

- FLOPs / MACs
- params
- activation memory
- bytes moved
- memory bandwidth
- arithmetic intensity = ops / byte
- compute-bound vs memory-bound
- batch-size effect
- throughput vs latency

### Dùng để

- chọn backbone có lý do
- biết nên giảm resolution, width, depth hay token count
- biết bottleneck là model, memory, copy, preprocess hay postprocess
- tránh tối ưu nhầm chỗ

### Output bắt buộc

- bảng params / FLOPs / activation memory
- bảng latency breakdown: preprocess / model / postprocess / I/O
- hypothesis: compute-bound hay memory-bound
- bằng chứng profiler

### Pass

- tính được memory sơ bộ trước khi chạy
- biết batch tăng throughput nhưng có thể tăng latency
- biết layer elementwise thường memory-bound
- biết convolution/matmul lớn thường dễ tận dụng Tensor Core hơn

---

## Stage 4 — Loss / metric / calibration math

### Học

- likelihood / log-likelihood
- entropy / cross entropy
- KL divergence
- softmax / sigmoid
- BCE / CE
- focal loss
- label smoothing
- class weighting
- confusion matrix
- precision / recall / F1
- ROC-AUC / PR-AUC
- IoU / Dice / mIoU / AP / mAP
- threshold sweep
- calibration curve
- expected calibration error
- temperature scaling

### Dùng để

- chọn metric đúng
- debug loss giảm nhưng metric không tăng
- xử lý class imbalance
- chọn threshold production
- biết confidence có đáng tin không

### Output bắt buộc

- loss curve
- metric curve
- threshold sweep
- PR curve hoặc ROC curve
- calibration curve nếu output dùng như probability

### Pass

- tự derive softmax + CE gradient
- tự implement BCE/CE/focal bằng NumPy
- biết khi nào accuracy vô dụng
- biết mAP khác accuracy ở đâu
- biết confidence 0.9 không đồng nghĩa xác suất đúng 90%

---

## Stage 5 — Gradient / optimizer / training stability

### Học

- derivative / partial derivative
- chain rule
- Jacobian
- vector-Jacobian product
- autograd graph
- gradient checking
- SGD / momentum / Nesterov
- RMSProp / Adam / AdamW
- weight decay
- learning-rate schedule
- warmup
- gradient clipping
- overfit / underfit
- regularization

### Dùng để

- fine-tune model ổn định
- debug exploding/vanishing gradient
- debug custom loss
- biết freeze/unfreeze layer hợp lý
- cải thiện model khi deployment trick không cứu được accuracy

### Output bắt buộc

- gradient norm log
- LR schedule
- overfit test trên subset nhỏ
- ablation optimizer / weight decay / augmentation

### Pass

- tự viết SGD + momentum + Adam mini
- tự gradient-check custom loss
- biết weight decay khác L2 trong AdamW ở đâu ở mức trực giác
- biết khi nào cần lower LR cho backbone

---

## Stage 6 — Architecture math for CV optimization

### Học

- stem / backbone / neck / head
- residual block
- bottleneck block
- inverted residual
- depthwise separable conv
- squeeze-excitation
- compound scaling
- feature pyramid
- anchor / anchor-free idea
- box regression
- NMS
- Hungarian matching
- mask loss
- skip connection
- upsampling
- transformer encoder block
- positional encoding

### Dùng để

- biết đổi backbone hay head
- biết layer nào đáng nén
- biết resolution/width/depth trade-off
- biết detector/segmenter chỉ khác head + loss + metric, không cần roadmap riêng

### Output bắt buộc

- architecture summary 1 trang
- bottleneck stage
- accuracy/latency/memory trade-off table
- quyết định: giữ model, đổi backbone, đổi head, hay fine-tune

### Pass

- giải thích được ResNet / EfficientNet / ViT bằng shape + cost
- biết DETR cần Hungarian matching vì set prediction
- biết segmentation tốn memory vì feature map lớn
- biết detection tốn postprocess nếu NMS/box decode nặng

---

## Stage 7 — Numerical precision / parity

### Học

- FP32 / TF32 / FP16 / BF16 / FP8
- rounding
- overflow / underflow
- accumulation error
- absolute error
- relative error
- ULP trực giác
- tolerance
- mixed precision
- output parity test

### Dùng để

- chuyển FP32 → FP16/BF16/FP8
- debug PyTorch vs ONNX vs TensorRT output lệch
- đặt tolerance hợp lý
- biết metric delta có chấp nhận được không

### Output bắt buộc

- max_abs_diff
- mean_abs_diff
- max_rel_diff
- cosine similarity nếu embedding
- metric delta FP32 vs lower precision
- pass/fail threshold

### Pass

- biết không cần bitwise-identical
- biết FP16 dễ underflow hơn BF16
- biết accumulation precision quan trọng hơn storage precision trong nhiều layer
- biết threshold production phải test lại sau precision change

---

## Stage 8 — Quantization math

### Học

- affine quantization
- symmetric / asymmetric quantization
- scale
- zero-point
- clipping
- calibration
- min-max / entropy / percentile calibration
- per-tensor / per-channel / block-wise
- Q/DQ graph
- fake quantization
- PTQ
- QAT
- quantization noise
- sensitive layer

### Dùng để

- INT8 / INT4 deployment
- giảm memory / bandwidth
- tăng inference speed nếu hardware hỗ trợ
- debug accuracy drop
- quyết định FP16 đủ hay cần INT8

### Output bắt buộc

- calibration dataset spec
- FP32 vs INT8 metric delta
- tensor/layer mismatch report
- list layer exclude nếu cần
- quyết định: PTQ pass, QAT cần, hoặc bỏ INT8

### Pass

- tự quantize 1 tensor bằng NumPy
- hiểu `x_fp32 ≈ scale × (x_int - zero_point)`
- biết calibration data không đại diện → accuracy tụt
- biết INT8 không tự nhanh nếu hardware/runtime không support tốt

---

## Stage 9 — Compression / distillation / pruning

### Học

- pruning
- sparsity
- structured vs unstructured sparsity
- low-rank factorization
- knowledge distillation
- temperature scaling trong distillation
- teacher-student KL
- smaller architecture search thủ công

### Dùng để

- giảm params
- giảm latency thật nếu sparsity/runtime hỗ trợ
- distill model lớn sang model nhỏ
- giữ metric khi model nhỏ hơn

### Output bắt buộc

- baseline teacher/student
- metric delta
- latency/memory delta
- sparsity pattern
- decision: compression có đáng không

### Pass

- biết pruning không chắc nhanh nếu runtime không tận dụng sparsity
- biết distillation cần teacher tốt và student đủ capacity
- biết low-rank giảm params nhưng có thể không giảm latency nếu kernel không phù hợp

---

## Stage 10 — Runtime / GPU performance model

### Học

- kernel launch overhead
- CPU-GPU sync
- H2D / D2H copy
- CUDA stream
- async pipeline
- occupancy trực giác
- warp / block / thread
- memory coalescing
- cache / shared memory trực giác
- Tensor Core shape alignment
- graph optimization / fusion
- dynamic shape cost

### Dùng để

- model nhanh nhưng end-to-end vẫn chậm
- GPU utilization thấp
- preprocessing CPU nghẽn
- batch=1 latency xấu
- TensorRT/ONNX/OpenVINO engine không tối ưu vì dynamic shape hoặc unsupported op

### Output bắt buộc

- profiler trace
- kernel time
- CPU time
- H2D/D2H time
- GPU utilization
- memory bandwidth estimate
- unsupported op list

### Pass

- biết profiler quan trọng hơn đoán
- biết operation nhỏ dễ latency-bound
- biết elementwise/fusion ảnh hưởng lớn
- biết Tensor Core không tự được dùng nếu shape/dtype/layout không hợp

---

# Thứ tự học ngắn nhất

1. Benchmark statistics
2. Tensor shape + layout
3. Conv / attention math
4. Cost model
5. Loss / metric / calibration
6. Gradient / optimizer
7. Architecture math
8. Numerical precision
9. Quantization
10. Compression
11. Runtime/GPU model

---

# Bài tập bắt buộc

## Bài 1 — Shape/cost audit

- Chọn 1 model CV bất kỳ.
- Ghi shape từng stage.
- Tính params từng block.
- Tính activation memory từng stage.
- Tính FLOPs thô.

## Bài 2 — Metric/threshold audit

- Tính confusion matrix.
- Vẽ precision/recall theo threshold.
- Vẽ PR curve.
- Chọn threshold theo business constraint.
- Không dùng accuracy một mình.

## Bài 3 — Profiler audit

- Chạy profiler.
- Tách preprocess / model / postprocess / I/O.
- Ghi top 10 operator theo time.
- Ghi top 10 operator theo memory.
- Đưa hypothesis bottleneck.

## Bài 4 — Precision parity

- So FP32 vs FP16/BF16/ONNX/TensorRT.
- Tính max_abs_diff / mean_abs_diff / metric delta.
- Đặt pass/fail threshold.

## Bài 5 — Quantization audit

- Quantize 1 tensor bằng NumPy.
- Chạy INT8 PTQ.
- So FP32 vs INT8.
- Tìm layer/tensor lệch nhất.
- Quyết định PTQ pass hay cần QAT.

---

# Không học sớm

- measure theory
- proof-heavy linear algebra
- convex optimization đầy đủ
- information theory sâu
- SVD sâu
- projective geometry sâu nếu không làm 3D/camera
- custom CUDA kernel trước khi biết profiler/cost model
- compiler theory sâu trước khi biết runtime bottleneck

---

# Nguồn học — chọn theo module, không đọc tuần tự

## Core books / notes

| Nguồn | Loại | Dùng cho |
|---|---|---|
| Mathematics for Machine Learning | book | linear algebra, calculus, probability, optimization nền |
| Deep Learning Book | book | numerical computation, optimization, CNN, regularization |
| Dive into Deep Learning | book + code | học bằng implementation, CNN, ViT, optimization, CV |
| CS231n Notes | course notes | CNN, backprop, loss, training, CV architecture |
| Szeliski — Computer Vision: Algorithms and Applications | book | signal/image geometry/classic CV khi cần |
| NVIDIA Deep Learning Performance Guide | docs | GPU cost model, Tensor Core, arithmetic intensity |

## Videos / courses / playlists

| Nguồn | Loại | Dùng cho |
|---|---|---|
| Stanford CS231n lecture videos | video course | CV deep learning chuẩn nhất |
| MIT 6.S191 | video course | deep learning fast overview + CV lecture |
| Full Stack Deep Learning | video course | deployment, testing, monitoring, ML systems |
| PyTorch official tutorials | tutorials | profiler, AMP, export, implementation |
| NVIDIA developer / DLI materials | videos/docs | TensorRT, CUDA, inference optimization |

## Papers cần đọc có chọn lọc

| Paper | Vì sao cần |
|---|---|
| Deep Residual Learning for Image Recognition | residual connection, backbone math |
| A guide to convolution arithmetic for deep learning | conv/pooling/transposed conv shape |
| EfficientNet | depth/width/resolution scaling |
| An Image is Worth 16x16 Words | ViT, patch/token/attention cost |
| FlashAttention | attention không chỉ FLOPs, còn IO/memory |
| DETR | object detection = set prediction + Hungarian matching |
| U-Net | segmentation memory, skip connection, upsampling |
| Focal Loss | class imbalance, dense detection |
| On Calibration of Modern Neural Networks | calibration, temperature scaling |
| Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference | integer quantization nền |
| Distilling the Knowledge in a Neural Network | teacher-student compression |
| Learning both Weights and Connections for Efficient Neural Networks | pruning/compression |

## Docs sát deployment

| Docs | Dùng cho |
|---|---|
| PyTorch Profiler | operator time + memory |
| PyTorch AMP | mixed precision |
| PyTorch Quantization | PTQ/QAT concepts |
| ONNX Runtime Quantization | static/dynamic quantization, QDQ, debugging |
| TensorRT Quantized Types | INT8/FP8/INT4, PTQ/QAT, calibration |
| CUDA C Programming Guide | GPU execution/memory/precision reference |
| NVIDIA GPU Performance Background | arithmetic intensity, compute-bound/memory-bound |
| ONNX export docs | export/parity/debug graph |

---

# Source links

## Books / courses

- Mathematics for Machine Learning: https://mml-book.github.io/
- Deep Learning Book: https://www.deeplearningbook.org/
- Dive into Deep Learning: https://d2l.ai/
- CS231n course: https://cs231n.stanford.edu/
- CS231n notes: https://cs231n.github.io/
- Szeliski book: https://szeliski.org/Book/
- MIT 6.S191: https://introtodeeplearning.com/
- Full Stack Deep Learning: https://fullstackdeeplearning.com/course/2022/

## Papers

- ResNet: https://arxiv.org/abs/1512.03385
- Convolution arithmetic: https://arxiv.org/abs/1603.07285
- EfficientNet: https://arxiv.org/abs/1905.11946
- ViT: https://arxiv.org/abs/2010.11929
- FlashAttention: https://arxiv.org/abs/2205.14135
- DETR: https://arxiv.org/abs/2005.12872
- U-Net: https://arxiv.org/abs/1505.04597
- Focal Loss: https://arxiv.org/abs/1708.02002
- Calibration: https://arxiv.org/abs/1706.04599
- Integer-only quantization: https://arxiv.org/abs/1712.05877
- Distillation: https://arxiv.org/abs/1503.02531
- Pruning: https://arxiv.org/abs/1506.02626

## Docs

- PyTorch Profiler: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- PyTorch AMP: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
- PyTorch Quantization in Practice: https://pytorch.org/blog/quantization-in-practice/
- ONNX Runtime Quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- TensorRT Quantized Types: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
- NVIDIA GPU Performance Background: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
- CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
