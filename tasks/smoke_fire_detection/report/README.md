# Báo cáo tổng hợp — Early Smoke/Fire Detection (lần báo cáo #1)

- Phạm vi: toàn bộ lịch sử dự án `smoke_fire_detection` (2026-07-07 → nay), vì đây là lần demo/báo cáo đầu tiên.
- Vai trò file này: gom bằng chứng + kết quả để chuẩn bị slide/demo. Methodology đầy đủ, quyết định, con số chốt: `../docs/research_plan.md` (static, không xóa). Timeline command chi tiết: git log (`git log --oneline`) + `CHANGELOG.md` (volatile, có thể đã bị user xóa trắng).
- Bằng chứng chạy thật (ảnh annotate + số liệu detection) nằm ở `demo_results/`, sinh bởi `demo_inference.py` (mục 6).

## 1. Tóm tắt 1 phút

- Bài toán: phát hiện sớm khói/lửa từ camera cố định trên tháp canh (early wildfire smoke detection). Mục tiêu là **time-to-detect (TTD)** và **false-alarm/giờ**, không phải mAP đơn thuần.
- Trạng thái: đã đi hết **một vòng nghiên cứu đầy đủ** — baseline domain khác → đo gate transfer → chẩn đoán lỗi → thử hướng sửa → **một thử nghiệm thất bại có đo lường (tiling)** → quyết định hướng đi bằng số liệu → hiện đang fine-tune 2 candidate detector đúng domain (YOLO26x, RF-DETR-L), **cả hai đều CHƯA xong** (18/20 và 5-7/20 epoch).
- Đóng góp lớn nhất tính đến nay **không phải "model đã xong"** mà là: (a) hạ tầng đo lường chặt (AMOC curve, bootstrap CI theo event, split camera-disjoint, audit leakage/lineage), (b) định lượng được domain gap thật (khói ở target camera nhỏ hơn ~20x so với ảnh train), (c) một thất bại thực nghiệm (tiling) được đo, phân tích, và đóng lại đúng quy trình thay vì lờ đi, (d) hạ tầng train/resume portable qua 3 nền tảng (Modal, server GPU cục bộ, Kaggle) với checkpoint kill-resume đã kiểm chứng.

## 2. Timeline thành quả (mốc chính)

| Ngày | Việc | Kết quả |
|---|---|---|
| 2026-07-07 | Scaffold repo | — |
| 2026-07-08 | E0 — D-Fire YOLO26n baseline | test: P=0.764 R=0.714 mAP50=0.684 mAP50-95=0.404; smoke mAP50-95=0.482, fire mAP50-95=0.326; latency 21.6ms mean / 46.4 fps (GPU GB10, imgsz 640) |
| 2026-07-08/09 | FIgLib index + audit | 40,362 frame hợp lệ / 511 folder / 510 sequence / 143 camera; anomaly data đã xác minh (archive rỗng, filename convention cũ) |
| 2026-07-09 | Dedup audit D-Fire (perceptual hash) | Bác bỏ giả thuyết cũ "val bị leak từ train nhiều hơn test" — tỷ lệ near-duplicate 2 phía gần bằng nhau (~24%) |
| 2026-07-09 | **Gate G0 — transfer check** | AUROC(smoke, imgsz640) = 0.7017 → **marginal** (0.60–0.80), chưa đủ mở thẳng G1 |
| 2026-07-09 | Diagnose lỗi (offset band, per-sequence) | AUROC tăng đơn điệu theo thời gian sau ignition (0.617→0.703→0.730); per-sequence bimodal: 26% AUROC≥0.9, 41% AUROC<0.6 |
| 2026-07-09 | Domain-gap định lượng | Bbox smoke phát hiện được ở FIgLib nhỏ hơn **~20 lần** median bbox lúc train (D-Fire) |
| 2026-07-09 | Thử imgsz 1280 | AUROC(smoke) = 0.7425 (+0.041) — vẫn marginal |
| 2026-07-09 | **Tiling 2×2 — THẤT BẠI có đo lường** | Xem mục 3 |
| 2026-07-09 | **Gate G1 — temporal gain** | N-of-M/EMA chỉ có lợi ở ngân sách chặt (FA≤1/tuần): TTD≈1122s @ recall 21%; ở FA≤1/ngày **không có gain rõ** so với single-frame (TTD≈825-840s @ recall ~47-49%) |
| 2026-07-09 | Phase-0 probe: spatial persistence | AUROC không đổi nhưng mở operating point mới: FA=0 tuyệt đối tại recall=0.135 |
| 2026-07-09 | Phase-0 probe: frame-differencing (motion) | AUROC=0.5276 (~random) → **đóng vĩnh viễn hướng motion-as-input** |
| 2026-07-09 | Human review 40 sequence (blind) | visible_smoke=23, fire_only=5, ambiguous=5, not_visible=7 |
| 2026-07-14 | Audit lineage `pyronear/yolov8s` | Không nguồn nào loại trừ leakage với FIgLib/HPWREN — mọi số của model này mang cờ leakage-caveat, không dùng làm winner chính thức |
| 2026-07-14 | Chốt candidate mới: RF-DETR-L | Quyết định thêm 1 model-family ngoài họ YOLO, protocol khóa (1280/20ep/batch4×acc4/seed cố định); kill-resume test PASS trước khi train dài |
| 2026-07-14 → nay | **Train YOLO26x/Pyro-SDIS** | Modal L4 → dừng tự nhiên ở epoch 18/20 (ephemeral run hết hạn, không phải lỗi) → đang resume nốt epoch 19-20 trên Kaggle T4 |
| 2026-07-14 → nay | **Train RF-DETR-L/Pyro-SDIS** | Server GB10 cục bộ, đang chạy liên tục, tới epoch 7/20 tại thời điểm viết báo cáo |

## 3. Bằng chứng thất bại — Tiling 2×2 (đúng như yêu cầu báo cáo)

- **Hypothesis:** chia ảnh 2×2 tile, detect trên từng tile độ phân giải gốc, sẽ phục hồi tín hiệu cho 131 sequence FIgLib đang "im lặng" (detector không trigger dù có khói).
- **Kết quả (pilot 10,345 frame, không chạy full 40K để tiết kiệm chi phí trước khi cam kết):**
  - `39/131` sequence thoát nhóm AUROC thấp (tín hiệu thật phục hồi được).
  - Nhưng: `zero_detection_rate_negative` giảm từ `0.956 → 0.787` — tức tỷ lệ frame negative không bị trigger giảm mạnh, nghĩa là **false alarm tăng**. `18/131` sequence chuyển từ "im lặng" sang "trigger sai".
  - Kết luận: **hỗn hợp, không phải thắng lợi** — mâu thuẫn trực tiếp với mục tiêu RQ1 ("false alarm thấp").
- **Vá thêm — refine ≥2-tile-agreement (chỉ giữ detection được ≥2 tile chồng lấn xác nhận):** dập được false-trigger gần về baseline (`0.787→0.955`) nhưng dập theo luôn phần lớn true-positive (`0.724→0.900`) — số sequence thoát nhóm thấp giảm từ `39/131` xuống `16/131`. Root cause đo được: true-positive vốn đã yếu ở cấp tile (case cụ thể: confidence global 0.633 nhưng per-tile chỉ 0.207/0.087) — giả định "tín hiệu thật nhất quán qua tile, nhiễu thì không" **sai trên data này**.
- **Quyết định (2026-07-09):** bỏ hướng tiling, dùng thẳng imgsz=1280 (AUROC 0.7425) làm nền, mở gate G1 bằng judgment call thay vì chờ G0 qua ngưỡng 0.80.
- **Trạng thái code:** đã xóa khỏi `eval.py`/`temporal_eval.py` theo quyết định dọn dẹp minh bạch (commit `0916eec`, 2026-07-14) — không giữ code chết cho một hướng đã đóng. Bằng chứng số liệu ở trên trích từ `docs/research_plan.md` mục 8 bước 9 (đã verify lại khi viết báo cáo này); chi tiết implementation nằm trong git history trước commit đó.

## 4. Hạ tầng kỹ thuật (đóng góp ít hiển nhiên nhưng tốn nhiều effort nhất)

- **Pipeline dữ liệu:** converter Pyro-SDIS (parquet HuggingFace → YOLO layout, gate audit `invariant_differences={}`), FIgLib indexer + split camera-disjoint (seed cố định, manifest commit git, không cần hash lock).
- **Harness đo lường:** AMOC curve (TTD vs FA/hour), bootstrap CI theo event (≥1000 resample), N-of-M/EMA temporal aggregator, tất cả offline-replay CPU trên detector cache (GPU chỉ chạy 1 lần/frame).
- **Đa nền tảng train + resume:**
  - Modal (L4/A100/H100 on-demand) — chạy `yolo26x_pyro_sdis_budget9`, dừng tự nhiên do ephemeral run hết hạn (không phải lỗi/crash), checkpoint đủ full-state để resume ở máy khác.
  - Server GB10 cục bộ (DGX Spark, unified memory 121GB) — train `rfdetr_large_pyro_sdis_gb10` liên tục, đo throughput thật bằng `nvidia-smi pmon` khi bị chia GPU với job khác, quyết định ưu tiên job chính dựa trên số đo (không đoán).
  - Kaggle (T4×2) — 3 notebook tự chứa (không git clone), fix thật từ log lỗi thật của user: path layout Kaggle, P100 không đủ (torch bỏ sm_60), OOM batch=4→ultralytics tự retry batch=2 (deviation đã ghi nhận, phân tích impact LR/EMA/BN không đổi kết luận đáng kể).
  - **Checkpoint contract kiểm chứng bằng thực nghiệm kill-resume** (SIGKILL giữa epoch, resume đúng optimizer/scheduler/EMA/step) cho cả YOLO26x và RF-DETR-L trước khi chạy dài không giám sát — không phải giả định suông.
- **Rigor phụ:** lineage/leakage audit cho checkpoint zero-shot bên thứ 3 (`pyronear/yolov8s`) bằng cách đọc ONNX metadata thật thay vì đoán resolution train.

## 5. Trạng thái model hiện có (để demo)

| Model | Vai trò | Trạng thái | Số liệu train-time |
|---|---|---|---|
| `dfire_yolo26n_baseline_full_vram` (YOLO26n / D-Fire) | E0 control, đã xong | **DONE** | test mAP50-95=0.404, mAP50=0.684 |
| `yolo26x_pyro_sdis_budget9` (YOLO26x / Pyro-SDIS) | Candidate chính | **ĐANG TRAIN** — checkpoint epoch 18/20 | Pyro val mAP50-95=0.492 (chưa final) |
| `rfdetr_large_pyro_sdis_gb10` (RF-DETR-L / Pyro-SDIS) | Candidate DETR-family | **ĐANG TRAIN** — checkpoint epoch 7/20 (best EMA tại epoch 5) | Pyro val mAP50-95=0.445, EMA mAP50-95=0.463 |

- Chưa có model nào chạy qua gate 13d (chọn candidate chính thức trên FIgLib) — cả hai còn đang train, số liệu FIgLib đầy đủ cho 2 candidate mới **chưa tồn tại**. Số trong bảng trên là số Pyro-SDIS val (nội bộ), không phải benchmark cuối cùng.

## 6. Demo inference bằng weight hiện tại (chưa train xong)

- Script: `demo_inference.py` — load 3 checkpoint hiện có (kể cả 2 checkpoint giữa chừng ở trên), chạy trên ảnh mẫu có sẵn của **3 dataset**: Pyro-SDIS val, D-Fire test, FIgLib (offset +600s ~ 10 phút sau ignition, đúng band mà `diagnose` đã đo là AUROC bắt đầu tốt hơn). Ảnh annotate + `summary.json` (số detection/confidence từng ảnh) lưu ở `demo_results/`.
- Chạy thật lúc viết báo cáo này (server `ai2`, GPU GB10, `.venv` project — script tự fallback CPU nếu GPU shared server bị OOM do job khác, đã xảy ra 1 lần trong lúc test và fallback hoạt động đúng).
- **Vài kết quả đáng chú ý (xem ảnh thật trong `demo_results/`):**
  - `demo_results/pyro_sdis_val/yolo26x_pyro_sdis/force-06_cabanelle-125_2024-02-24T09-07-27.jpg`: YOLO26x bắt đúng một cột khói rất nhỏ ở xa, confidence 0.70 — đúng use-case mục tiêu (khói nhỏ, camera tháp xa).
  - `demo_results/figlib_sample/rfdetr_large_pyro_sdis/1466360053_+00600.jpg`: RF-DETR-L (mới train 5-7/20 epoch) vẫn bắt được một vệt khói mờ rất nhỏ ở đường chân trời trên ảnh FIgLib thật (domain mục tiêu cuối cùng) — tín hiệu tích cực dù chưa train xong.
  - `demo_results/dfire_test/*`: cả 3 model đều **0 detection** trên cả 3 ảnh D-Fire test lấy mẫu — cỡ mẫu quá nhỏ (3 ảnh) để kết luận gì, chỉ nêu trung thực khi demo, không phải benchmark chính thức (số D-Fire chính thức đã có ở mục 5, đo trên toàn bộ test set 4,306 ảnh).
- **Giới hạn phải nói rõ khi demo/báo cáo:** đây là 3 ảnh/dataset để minh họa trực quan, KHÔNG phải eval chính thức; 2/3 model dùng checkpoint giữa chừng, số sẽ đổi khi train xong; không suy rộng từ demo này thành kết luận benchmark.

## 7. Việc tiếp theo

- Train xong YOLO26x (2 epoch cuối, Kaggle) + RF-DETR-L (13 epoch còn lại, GB10) → đồng bộ checkpoint cuối về `artifacts/`.
- Chạy `eval.py detector-cache` cho cả 2 candidate trên FIgLib đầy đủ → gate 13d (chọn candidate chính thức bằng paired bootstrap event + cluster bootstrap camera).
- Mở final camera-held-out sau khi khóa winner (chỉ 1 lần, không chạy nhiều lần).
- Chi tiết đầy đủ next-step: `../docs/research_plan.md` mục 8 bước 13-14.
