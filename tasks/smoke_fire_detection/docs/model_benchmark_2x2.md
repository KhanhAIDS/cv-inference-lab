# Benchmark 2×2 — YOLO26x/RF-DETR-L × Pyro-SDIS/D-Fire trên FIgLib (dev)

Trạng thái: **dev-stage xong, winner khóa. Final camera-held-out (`--split test`) CHƯA chạy — chờ duyệt.**

## 1. Protocol

- Ma trận chính (4 candidate sạch, tranh winner): `yolo26x_pyro_sdis`, `rfdetr_large_pyro_sdis`, `yolo26x_dfire`, `rfdetr_large_dfire`. Control/reference (không tranh winner): `yolo26n_dfire_control` (D-Fire label gốc), `pyronear_yolov8s_reference` (leakage-caveat, không rõ tập train).
- Train: Pyro-SDIS ở resolution 1280, D-Fire ở resolution 800; cả 2 dataset 20 epoch, seed `20260707`, batch/optimizer/scheduler/augmentation native theo từng framework (không ép iso-config giữa YOLO26x và RF-DETR-L). D-Fire dùng label đã relabel cho `yolo26x_dfire`/`rfdetr_large_dfire`; `yolo26n_dfire_control` dùng label gốc — không so trực tiếp giữa 2 loại label để suy diễn data-quality effect.
- Eval trên FIgLib dev split: camera-disjoint, seed `20260707`, 359 sequence / 108 camera (0 overlap với final split 152 sequence / 35 camera). Resolution inference chính: **1280 cho cả 4 candidate sạch + control**, kể cả 2 model D-Fire train@800 (quyết định user 2026-07-20, xem mục 6 — đã verify không phải premature). Pyronear: 1024 (số chính, khớp resolution train xác nhận qua ONNX metadata).
- Metric chọn winner: `AUROC(max_smoke_confidence)`, nhãn dương khi offset ≥ 180s sau ignition, nhãn âm khi offset < 0 (pre-ignition), band `[0,180s)` loại bỏ khỏi tính toán. Bootstrap 1000 mẫu, seed `20260707`, 2 cấp: event (cluster theo `sequence_id`) và camera (cluster theo `camera_id`). **Không multiple-comparison correction** (rule pre-registered, ghi caveat).
- Seed caveat: mỗi candidate chỉ train 1 seed. CI bootstrap ở đây đo nhiễu *đánh giá* (event/camera resample), không đo nhiễu *huấn luyện* (seed). Khoảng cách nhỏ giữa 2 candidate gần nhau không phân biệt được với may rủi seed.
- Nguồn: `artifacts/smoke_fire_detection/figlib_dev_compare_matrix.json` (2×2 matrix), `figlib_dev_temporal_rfdetr_large_dfire.json` (operating point winner), `figlib_dev_winner_lock.json` (khóa winner), `gate_g0_auroc_*.json` (AUROC control/reference/resolution phụ).

## 2. Source-domain metrics (train/val/test gốc, không phải FIgLib)

| Candidate | Dataset | Split | mAP50-95 | mAP50 | Ghi chú |
|---|---|---|---|---|---|
| `yolo26x_pyro_sdis` | Pyro-SDIS | val | 0.4949 | 0.7450 | epoch 19/20 best |
| `rfdetr_large_pyro_sdis` | Pyro-SDIS | val | 0.4627 | 0.7502 | COCO-mAP, `imgsz=1280` |
| `yolo26x_dfire` | D-Fire (relabel) | test | 0.40957 | 0.7287 | smoke 0.458/0.778, fire 0.361/0.679 |
| `rfdetr_large_dfire` (**winner**) | D-Fire (relabel) | test | **0.4801** | **0.8194** | smoke 0.5548/0.8664, fire 0.4054/0.7724, `imgsz=800` |
| `yolo26n_dfire_control` | D-Fire (label gốc) | test | 0.404 | — | control, label khác 2 dòng trên |

Không so trực tiếp D-Fire vs Pyro-SDIS (khác class composition). Không so trực tiếp `yolo26n_dfire_control` với 2 model D-Fire fresh để suy diễn "data-quality effect" (label khác nhau, xem mục 1).

## 3. Dev 2×2 matrix — AUROC(smoke) trên FIgLib, resolution 1280

| Candidate | Kiến trúc | Dataset | AUROC | CI95 event | CI95 camera |
|---|---|---|---|---|---|
| `yolo26x_pyro_sdis` | YOLO26x | Pyro-SDIS | 0.7839 | [0.7617, 0.8040] | [0.7604, 0.8070] |
| `rfdetr_large_pyro_sdis` | RF-DETR-L | Pyro-SDIS | 0.7871 | [0.7651, 0.8088] | [0.7618, 0.8094] |
| `yolo26x_dfire` | YOLO26x | D-Fire | 0.7605 | [0.7382, 0.7812] | [0.7392, 0.7821] |
| **`rfdetr_large_dfire`** | RF-DETR-L | D-Fire | **0.8317** | **[0.8103, 0.8498]** | **[0.8097, 0.8518]** |
| `yolo26n_dfire_control` (control) | YOLO26n | D-Fire (label gốc) | 0.7380 | [0.7151, 0.7607] | [0.7147, 0.7591] |
| `pyronear_yolov8s_reference` (leakage-caveat) | YOLOv8s | không rõ | 0.8078 | [0.7872, 0.8282] | [0.7869, 0.8277] |

27,263/28,360 frame dùng (1,097 frame rơi vào ignore-band `[0,180s)`, 358 sequence, 108 camera).

**Control/reference không vượt winner**: `yolo26n_dfire_control` (0.7380) và `pyronear_yolov8s_reference` (0.8078, leakage-caveat) đều thấp hơn `rfdetr_large_dfire` (0.8317) — không rơi vào case "control thắng tuyệt đối" (mục Giả định khóa của plan).

## 4. Pairwise, architecture effect, dataset effect, interaction

**Pairwise (Δ = cột − hàng, dương nghĩa là cột tốt hơn):**

| | `yolo26x_pyro_sdis` | `rfdetr_large_pyro_sdis` | `yolo26x_dfire` | `rfdetr_large_dfire` |
|---|---|---|---|---|
| `yolo26x_pyro_sdis` | — | +0.0033 [-0.0107,0.0173] | -0.0234 [-0.0406,-0.0058] | +0.0479 [0.0307,0.0659] |
| `rfdetr_large_pyro_sdis` | | — | -0.0266 [-0.0434,-0.0100] | +0.0446 [0.0290,0.0620] |
| `yolo26x_dfire` | | | — | +0.0712 [0.0559,0.0862] |
| `rfdetr_large_dfire` | | | | — |

(CI là event-bootstrap; camera-bootstrap cho cùng dấu và độ lớn tương tự, xem `figlib_dev_compare_matrix.json`.)

**Architecture effect (RF-DETR − YOLO), theo dataset:**
- D-Fire: **+0.0712** [0.0559, 0.0862] event / [0.0559, 0.0876] camera — RF-DETR-L rõ ràng tốt hơn YOLO26x trên D-Fire (CI không chứa 0).
- Pyro-SDIS: +0.0033 [-0.0107, 0.0173] event / [-0.0116, 0.0179] camera — **không có khác biệt kiến trúc đáng kể** trên Pyro-SDIS (CI chứa 0).

**Dataset effect (Pyro − D-Fire), theo kiến trúc:**
- YOLO26x: +0.0234 [0.0058, 0.0406] — YOLO26x làm tốt hơn khi train trên Pyro-SDIS.
- RF-DETR-L: -0.0446 [-0.0620, -0.0290] — RF-DETR-L làm tốt hơn khi train trên D-Fire (dấu ngược YOLO).

**Interaction:** -0.0680 [-0.0886, -0.0477] — CI không chứa 0, xác nhận **crossover interaction thật**: kiến trúc nào "thắng" phụ thuộc dataset train, không có kiến trúc thắng tuyệt đối trên cả 2 dataset. Đây là finding chính của benchmark 2×2, không phải nhiễu thống kê.

## 5. Winner rule và kết quả

- Rule (pre-registered): leader = AUROC điểm cao nhất trong 4 candidate sạch; leader thắng 1 đối thủ khi event-bootstrap lower CI của ΔAUROC>0 VÀ camera-bootstrap median delta cùng chiều >0; thắng hết 3 đối thủ → winner rõ.
- **Leader = winner = `rfdetr_large_dfire`**, thắng cả 3 đối thủ (`yolo26x_pyro_sdis`, `rfdetr_large_pyro_sdis`, `yolo26x_dfire`) theo đúng rule — không cần tie-break bằng latency/cost.
- Gate AUROC≥0.80: **PASS** (0.8317 ≥ 0.80).
- Artifact khóa: `artifacts/smoke_fire_detection/figlib_dev_winner_lock.json`.

## 6. Resolution sensitivity — câu hỏi bắt buộc @1280 vs @800 cho 2 model D-Fire

Cả `yolo26x_dfire` và `rfdetr_large_dfire` train ở D-Fire resolution 800, nhưng winner rule dùng AUROC đo ở 1280 (ép lên, không phải resolution train gốc). Đã cache thêm `@800` cho cả 2 để kiểm tra trực tiếp:

| Candidate | AUROC@1280 | AUROC@800 | CI95 event @800 |
|---|---|---|---|
| `yolo26x_dfire` | 0.7605 | 0.7308 | [0.7091, 0.7520] |
| `rfdetr_large_dfire` (winner) | 0.8317 | 0.8256 | [0.8041, 0.8450] |

**Kết luận: `@800` KHÔNG vượt `@1280` cho cả 2 model** — `yolo26x_dfire` thấp hơn rõ ở `@800` (0.7308 vs 0.7605), `rfdetr_large_dfire` (winner) chỉ nhỉnh hơn không đáng kể ở `@1280` (0.8317 vs 0.8256, CI chồng lấn gần như hoàn toàn). Quyết định khóa resolution 1280 cho cả 4 candidate sạch (chốt 2026-07-20, trước khi có số `@800`) **không phải là hấp tấp/premature** — nếu `@800` vượt rõ thì đây sẽ là giới hạn thật của winner rule, nhưng dữ liệu thực tế cho thấy ngược lại. Không có oversight cần ghi nhận, không cần đổi winner.

## 7. Latency / VRAM (Modal L4, batch 1, 30 warmup + 300 measured × 3)

| Candidate | Resolution | Mean latency | FPS | Peak VRAM |
|---|---|---|---|---|
| `yolo26x_pyro_sdis` | 1280 | 82.26 ms | 12.81 | 798.7 MB |
| `rfdetr_large_pyro_sdis` | 1280 | 141.18 ms | 7.08 | 461.5 MB |
| `yolo26x_dfire` | 1280 | 77.26 ms | 12.94 | 798.5 MB |
| `rfdetr_large_dfire` (winner) | 1280 | 136.05 ms | 7.35 | 461.5 MB |
| `yolo26n_dfire_control` | 1280 | 41.73 ms | 23.97 | 142.6 MB |
| `pyronear_yolov8s_reference` | 1024 | 34.82 ms | 28.72 | 141.4 MB |

Winner (`rfdetr_large_dfire`) chậm hơn YOLO26x cùng resolution ~1.76x (136ms vs 77-82ms), nhưng VRAM thấp hơn (461.5MB vs 798.5MB). Đây là chi phí phải trả cho +0.07-0.09 AUROC so với 3 candidate sạch còn lại — không tie-break bằng latency vì winner đã thắng rõ theo AUROC (mục 5), không rơi vào top-set tie. `$/camera-tháng` chưa tính (chỉ cần khi tie-break thật, không phát sinh ở đây).

## 8. Operating point khóa (winner `rfdetr_large_dfire`, N-of-M/EMA trên FIgLib dev)

| Budget | Rule | Recall | Precision | FA/giờ (CI95) | TTD median | TTD mean (CI95) |
|---|---|---|---|---|---|---|
| FA≤1/camera/ngày | EMA α=0.5, threshold=0.6 | 0.690 | 0.969 | 0.0347 [0.0130, 0.0608] | 598s (~10.0 phút) | 714.6s [655.2, 776.2] |
| FA≤1/camera/tuần | EMA α=0.1, threshold=0.8 | 0.067 | 0.960 | 0.00434 [0, 0.0130] | 2010s (~33.5 phút) | 1943s [1797.3, 2072.4] |

Trade-off rõ: ngân sách FA lỏng (1/ngày) cho recall 69% với TTD ~10 phút; ngân sách chặt (1/tuần) chỉ còn recall 6.7% (phần lớn event bị bỏ sót để giữ false-alarm cực thấp) với TTD ~33.5 phút. Đây là 2 operating point sẽ dùng khi mở final (Giai đoạn 5), không quét lại threshold ở đó.

## 9. So sánh SOTA (research 2026-07-20, WebSearch + xác minh — xem caveat protocol)

- **SmokeyNet gốc (Dewangan et al., arXiv 2112.08598, Remote Sensing 2022):** metric Acc/Prec/Recall/F1/TTD, **không có AUROC** — 2-frame: Acc 83.49%, F1 82.59%, TTD 3.12 phút; human baseline Acc 78.5%/F1 82.8%. Split by-fire 144/64/62 (315 sequence, 24,800 ảnh tile 224×224) — **khác hẳn** split camera-disjoint 510-sequence/dev-final hiện tại và khác metric. Không so trực tiếp số, chỉ dùng làm neo định tính: winner benchmark này (AUROC 0.83) và SmokeyNet gốc (Acc/F1 ~83%) nằm cùng bậc "khá tốt nhưng chưa gần hoàn hảo" trên FIgLib, dù không thể quy đổi trực tiếp.
- **Paper cùng split SmokeyNet (arXiv 2311.10116):** Acc 84.67%/F1 84.05%. **Baldota et al. (arXiv 2212.14143)** tự reproduce SmokeyNet ra thấp hơn bản gốc (Acc 80.12/F1 77.52, split gần giống 131/63/61) — cho thấy biến thiên reproduction thật giữa các nhóm, không riêng gì benchmark này.
- **Pyronear/pyro-sdis:** model card `pyronear/yolov8s` **không công bố mAP/P/R/F1 nào**; không có số FIgLib/temporal công bố cho chính checkpoint này. Paper khác cùng tổ chức (PYRONEAR-2025, arXiv 2402.05349, **khác checkpoint/dataset**) báo P/R/F1/TTD (single-frame P0.805/R0.775/F1 0.790/TTD 1.76 phút) — không cùng protocol, chỉ tham chiếu gián tiếp. Số benchmark này đo được cho chính checkpoint `pyronear/yolov8s` trên đúng split camera-disjoint: AUROC 0.8078 (leakage-caveat, xem mục 3) — đây là số **tự đo, không lấy từ paper**.
- **D-Fire baseline literature:** mọi số công khai tìm được đều là mAP50 (60.6-80.9%), **không có mAP50-95 công khai** để đối chiếu trực tiếp. Số của benchmark này (mục 2): YOLO26x mAP50 test 0.7287 (trong range), RF-DETR-L mAP50 test 0.8194 (nhỉnh hơn biên trên, hợp lý — RF-DETR-L cũng là kiến trúc mạnh hơn baseline YOLOv5/v8 dùng trong các nguồn đó).
- **Latency GPU:** không tìm được số latency wildfire-smoke cụ thể trên T4/L4/A10 để so trực tiếp. Benchmark chính hãng RF-DETR-L @704px T4 TensorRT10.4 FP16 batch1 = 6.8ms; YOLO26-X @640px cùng điều kiện = 9.6ms (Roboflow/Ultralytics docs). Số đo ở mục 7 (L4, 1280px, PyTorch eager, không TensorRT) cao hơn ~8-20x — giải thích hợp lý bằng resolution cao hơn (~3-4x pixel) + thiếu tối ưu TensorRT/FP16 (chưa gọi `model.optimize_for_inference`), **không phải do L4 yếu hơn T4**. Không so trực tiếp số latency giữa 2 nguồn vì khác điều kiện đo.

## 10. Giới hạn / caveat

- **Weak label:** FIgLib không có bbox, nhãn event-level theo `ignition_offset_seconds`; AUROC đo trên nhãn frame suy ra từ event, không phải nhãn frame-level thật.
- **Ignore band:** `[0,180s)` loại khỏi tính toán để tránh nhãn nhập nhằng ngay lúc ignition — 1,097/28,360 frame (3.9%) bị loại mỗi candidate.
- **Negative denominator (FA/giờ):** tính trên tổng giờ cửa sổ âm toàn dev set (230.6 giờ), không phải theo từng camera riêng — giả định implicit là false-alarm rate đồng nhất giữa camera.
- **Corrupt JPEG:** 1 frame lỗi (rỗng) trong index, loại khỏi mọi cache, đã log ở `.errors.json` từng candidate.
- **Pyronear leakage-caveat:** không xác minh được `pyronear/yolov8s` có thấy ảnh FIgLib/HPWREN lúc train hay không (audit lineage 2026-07-14, xem `research_plan.md` mục 8 bước 12) — số của nó (mục 3) có thể lạc quan hơn thực tế zero-shot thật.
- **Seed caveat:** 1 seed/candidate (mục 1) — gap nhỏ giữa candidate gần nhau (vd `yolo26x_pyro_sdis` vs `rfdetr_large_pyro_sdis`, Δ=0.0033, CI chứa 0) không phân biệt được với nhiễu seed.
- **No multiple-comparison correction:** 6 pairwise + 2 architecture-effect + 2 dataset-effect + 1 interaction đều dùng CI95 riêng lẻ, không điều chỉnh family-wise error — rule pre-registered trước khi thấy số, chấp nhận theo plan gốc.
- **Resolution-confound đã kiểm tra và loại trừ** cho winner (mục 6) — không còn treo.

## 11. Việc CHƯA làm — để lại khi có thêm thời gian (không phải bỏ sót)

Theo ràng buộc thời gian, các phần sau trong plan gốc **cố ý chưa làm**, không tự ý làm thêm hay bỏ qua âm thầm:

- **Human-visibility stratification** (visible/fire_only/ambiguous/not_visible) — `human_review_labels.json` đã bị dọn khỏi artifacts, cần restore từ git history nếu muốn dùng.
- **Spatial-persistence overlay** cho tier FA≤1/tuần + zero-FA — subcommand đã bị trim khỏi `temporal_eval.py`, chưa quyết định restore hay hủy cam kết.
- **Lát cắt lỗi L0** trên winner (bbox area/short-side, visibility, offset, camera, FP category) — P1 sau khi khóa winner, chưa chạy.
- **Roadmap P1-P3** (PYRONEAR-2025 data-lever, SmokeyNet reference reproduction, tối ưu suy luận cấp xuất xưởng, RQ5 near-field) — parked, sequenced sau khi user duyệt final.
- **$/camera-tháng** — không cần vì không có tie-break thật ở winner rule.

## 12. Bước tiếp theo

Giai đoạn 5 (final camera-held-out, `eval.py detector-cache --split test`) là thao tác **one-shot, không lặp lại được** — chỉ chạy `rfdetr_large_dfire` (winner), `yolo26n_dfire_control`, `pyronear_yolov8s_reference` (allowlist trong `figlib_dev_winner_lock.json`). **Chưa chạy, chờ user duyệt tường minh.**
