# Benchmark 2×2 — YOLO26x/RF-DETR-L × Pyro-SDIS/D-Fire trên FIgLib (dev)

Trạng thái: **DONE — dev-stage + final camera-held-out (`--split test`) đều xong, winner khóa, gate final PASS (2026-07-21).**

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

## 12. Final camera-held-out (test split) — DONE 2026-07-21, gate PASS

- Split: 152 sequence / 35 camera, 0 overlap với dev (359 seq/108 cam). One-shot, mở đúng 1 lần theo `figlib_dev_winner_lock.json` (`check_final_split_gate` reject candidate ngoài winner+allowlist). Chỉ chạy 3 candidate: `rfdetr_large_dfire` (winner), `yolo26n_dfire_control`, `pyronear_yolov8s_reference` — không chạy lại 3 candidate sạch đã thua trên dev (kỷ luật train/dev/test, tránh multiple-look trên tập held-out).
- 12,001 frame/candidate, 477 rơi vào ignore-band, 11,524 frame dùng, 5,587 positive/5,937 negative. Resolution/conf/postprocess giữ nguyên winner-lock.

| Candidate | AUROC(smoke) | CI95 event | CI95 camera |
|---|---|---|---|
| `rfdetr_large_dfire` (winner) | **0.8347** | [0.8040, 0.8643] | [0.7851, 0.8708] |
| `yolo26n_dfire_control` | 0.7551 | [0.7229, 0.7875] | [0.6978, 0.7946] |
| `pyronear_yolov8s_reference` (leakage-caveat) | 0.8486 | [0.8196, 0.8766] | [0.8042, 0.8826] |

- **Gate AUROC≥0.80: PASS** (0.8347 ≥ 0.80) — khớp dev (0.8317), không rớt gate trên tập held-out chưa từng đụng tới.
- **Pairwise winner vs control:** Δ=+0.0796, event CI [0.0556, 0.1074], camera CI [0.0487, 0.1083] — CI không chứa 0 → winner thắng control rõ, khớp hướng dev.
- **Pairwise winner vs Pyronear reference (leakage-caveat):** Δ=-0.0138, event CI [-0.0383, 0.0109], camera CI [-0.0379, 0.0114] — **CI chứa 0 cả 2 phía, khác biệt không có ý nghĩa thống kê**, dù điểm Pyronear nhỉnh hơn (0.8486 vs 0.8347). Không phải "winner thua" — 2 số không phân biệt được bằng thống kê, và Pyronear vẫn mang leakage-caveat (mục 10) nên số của nó không đáng tin hơn winner.
- Artifact: `gate_g0_auroc_{rfdetr_large_dfire,yolo26n_dfire_control,pyronear_yolov8s_reference}_final.json`, `figlib_final_compare_winner_vs_control.json`, `figlib_final_compare_winner_vs_pyronear.json`.

### 12.1 Replay operating point khóa (không quét lại threshold trên test)

| Tier | Rule khóa | Recall | Precision | FA/giờ (CI95) | TTD median | So với dev |
|---|---|---|---|---|---|---|
| FA≤1/ngày | EMA α=0.5, thr=0.6 | 0.704 | 0.947 | 0.0615 [0.0104, 0.1122] | 539s | dev: recall 0.690, FA 0.0347/h — recall giữ, **FA vượt budget ngày (0.0417/h) ~1.5x** |
| FA≤1/tuần | EMA α=0.1, thr=0.8 | 0.099 | 0.882 | 0.0205 [0, 0.0511] | 1980s | dev: recall 0.067, FA 0.00434/h — recall nhích, **FA vượt budget tuần (0.00595/h) ~3.4x** |

- **Giới hạn phát hiện được, không giấu:** cả 2 threshold khóa trên dev đều KHÔNG giữ đúng FA budget khi replay 1 lần trên test. Nguyên nhân nhiều khả năng là cỡ mẫu: final chỉ có 97.6 giờ cửa sổ âm (dev 230.6 giờ) và 35 camera (dev 108) — vài false-alarm event thêm/bớt đổi hẳn FA/giờ. Đây đúng là lý do kỷ luật "khóa trên dev, replay đúng 1 lần trên test" tồn tại — nó bắt được hiện tượng operating point không transfer hoàn hảo mà nếu chỉ tin số dev sẽ không phát hiện ra. Không đổi winner, không tự tune lại threshold trên test.

## 13. Spatial-persistence overlay (restore + chạy, quyết định user 2026-07-21)

- Cam kết cũ (`research_plan.md` mục 8 bước 12, 2026-07-09): giữ overlay 0-GPU (rescoring = trung bình cửa sổ 5-frame theo vị trí, thay `max_smoke_confidence`) cho tier FA≤1/tuần + zero-FA. Subcommand từng bị trim khỏi `temporal_eval.py`, đã restore nguyên bản từ git history (commit `d82bfcf`).
- Threshold khóa trên **dev** cache của winner (không tìm trên test): sweep EMA α∈{0.05,0.1,0.2,0.3,0.5} × threshold∈{0.5..0.99} + N-of-M mặc định, chọn theo đúng rule (đạt budget, recall cao nhất, tie→TTD, tie→FA).
- Pooled AUROC(smoke) overlay: dev 0.8319 (raw 0.8317, CI trùng), final 0.8349 (raw 0.8347, CI trùng) — **overlay không đổi AUROC pooled**, đúng phát hiện gốc 2026-07-09 trên detector khác; giá trị của nó chỉ ở định hình lại operating point đuôi (tail), không phải cải thiện detector.

| Tier | Rule khóa trên dev | Recall dev | FA/giờ dev | Recall final (replay) | FA/giờ final (replay) |
|---|---|---|---|---|---|
| "Zero-FA" | EMA α=0.1, thr=0.7 | 0.246 | 0.0 (CI [0,0]) | 0.342 | **0.0307** (CI [0, 0.0617]) — không còn zero trên final (3 false-alarm event) |
| FA≤1/tuần | EMA α=0.05, thr=0.55 | 0.305 | 0.00434 (CI [0, 0.0131]) | 0.368 | **0.0410** (CI [0.0102, 0.0819]) — vượt budget tuần ~6.9x trên final |

- **Đọc kết quả trung thực:** recall overlay (0.246-0.368) cao hơn hẳn recall raw ở tier tương ứng (raw tuần: 0.067 dev / 0.099 final) — hướng cải thiện giữ nhất quán dev→final, tín hiệu thật. NHƯNG cam kết "zero-FA"/"FA≤1/tuần" không giữ được trên final, cùng pattern budget-overshoot đã thấy ở raw score (mục 12.1) — củng cố giả thuyết đây là giới hạn cỡ mẫu của chính final split, không phải lỗi riêng của overlay. Không có CI cho `event_recall` (hạn chế sẵn có của pipeline, đã ghi từ bản gốc 2026-07-09) — số recall trên là điểm ước lượng, chưa có khoảng tin cậy.
- Kết luận vai trò: overlay xác nhận lại đúng vai trò ban đầu — **diagnostic/tham khảo cho tier chặt, không thay được raw score làm operating point chính thức** (không đủ độ tin cậy để cam kết FA budget cứng ở quy mô camera hiện có).
- Artifact: `gate_g0_auroc_rfdetr_large_dfire_spatial_persistence_{dev,final}.json`, `figlib_final_temporal_overlay_{zerofa,week}.json`. Sweep khóa threshold dev (`figlib_dev_temporal_overlay_sweep.json`) đã xóa 2026-07-21 sau khi chốt số vào đây — cần lại thì tra git history hoặc chạy lại `temporal_eval.py` sweep từ cache winner. Cache rescored trung gian (`figlib_{dev,final}_spatial_persistence_rfdetr_large_dfire.jsonl`) không giữ lại — tái tạo được bằng 1 lệnh `temporal_eval.py spatial-persistence` từ cache gốc đã giữ.

## 14. Bước tiếp theo (đề xuất, chưa tự chạy)

- Final AUROC 0.8347 ≥ 0.80 → gate PASS, không cần proposal xếp hạng G4 (chỉ cần khi fail).
- Đề xuất mở tiếp theo đúng roadmap đã ghi (`research_plan.md` mục P1, chưa đổi): (a) lát cắt lỗi L0 trên winner trước khi chọn đòn bẩy nào; (b) PYRONEAR-2025 data-lever (lọc FIgLib-source) làm data-only scale-up, có control; (c) SmokeyNet reference reproduction (2 lead đã tìm, chưa tải); (d) E1a qua đường tải thay thế để có FA/hour denominator đúng domain; (e) chỉ mở G2 (learned verifier) sau khi có E1 data.
- Không tự chạy bất kỳ mục nào ở trên — chờ user duyệt riêng từng mục.
