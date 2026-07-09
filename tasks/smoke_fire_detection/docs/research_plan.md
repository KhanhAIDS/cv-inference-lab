# Research Plan — Early Fire Detection (v2)

- Updated: `2026-07-08`
- Thay thế bản v1 (dạng chat monologue: không có decision rule, không có cost, dẫn chiếu PYRONEAR-2025 như thể đã có local, citation hỏng).
- Cách dùng: file này chốt **câu hỏi, gate, thứ tự**. Kỹ thuật cụ thể tra `research_toolbox.md`. State volatile tra `agent_context.md` (root).

## 1. Research question

- **RQ1 (chính):** early fire detection từ camera cố định, false alarm thấp — temporal confirmation cải thiện single-frame RGB bao nhiêu, đo event-level.
- **RQ2 (định danh của lab):** cost frontier — tại mỗi mức chi phí inference (`$/camera-tháng` đo thực trên Modal), TTD tốt nhất đạt được ở FA-budget cố định là bao nhiêu.
- **RQ3 (parked):** thermal/RGB-T gain; distill RGB-T teacher sang RGB student.

## 2. Fact đã kiểm chứng (2026-07-08) — nền của mọi quyết định

- FIgLib local: `512` sequence, `136` unique camera-string, `40362` frame hợp lệ (`figlib_index.py`, verified 2026-07-08 sau khi merge 476 sequence mới từ `tempo/`), cadence `60s`, offset `-2400..+2400s`, ảnh `2048x1536`, không có bbox → chỉ có weak label theo dấu offset. Folder pattern `{YYYYMMDD}_{FIRE|fire_name}_{camera_id}`, parse bằng `split("_", 2)` (đã tổng quát hoá, verified với hàng trăm tên đám cháy khác nhau, không chỉ literal `FIRE`).
  - **Data quality anomaly (verified 2026-07-08):** `20250123_GilmanFire_tdllns-mobo-c` là archive rỗng thật sự (chỉ có thư mục, 0 file — lỗi từ nguồn tải, không phải lỗi giải nén). `20200831_FIRE_wc-n-mobo-c` (180 frame) dùng convention tên file cũ `{timestamp}.jpg` (không có suffix offset) → bị `figlib_index.py` loại hoàn toàn khỏi index (0 frame hợp lệ). `20250801_BernardoFire_bl-n-mobo-c` và `20250804_CoolFire_bi-w-mobo-c` trộn lẫn cả hai convention (một phần frame bị loại: 41 và 40 frame tương ứng). Tổng `bad_filename_frames` = 261.
  - **Quan trọng — sửa lại giả định cũ:** đây KHÔNG còn là "36/315 sequence của bộ FIgLib cố định". Các sequence mới trải dài tới `2026-06` (vd `20260629_JunctionFire_*`, `20250107_EatonFire_*`) — HPWREN duy trì FIgLib như archive sống, liên tục thêm fire mới sau khi paper gốc công bố, không phải một tập tĩnh 315 sequence. Không còn mẫu số cố định để nói "đã tải X/Y%".
- FIgLib gốc theo paper: ~24.8k ảnh, 315 sequence, 101 camera (SmokeyNet, [arXiv 2112.08598](https://arxiv.org/abs/2112.08598)) — đây là snapshot học thuật đóng băng lúc paper publish (2021), không phải giới hạn trên của archive hiện tại.
- HPWREN archive công khai, duyệt theo camera + ngày, có tool bulk download, chỉ yêu cầu attribution ([hpwren.ucsd.edu](https://www.hpwren.ucsd.edu/news/20210318/)) → nguồn negative-day gần như vô hạn, **đúng chính các camera trong FIgLib**.
- PYRONEAR-2025 tồn tại thật ([arXiv 2402.05349](https://arxiv.org/abs/2402.05349)): ~50k ảnh, ~150k bbox annotation, 640 fire, có video sequence cho sequential model. **Chưa có local.**
- Cadence use case: `1 frame/phút/camera` → latency budget mỗi frame là hàng chục giây. Bottleneck thật khi scale là `$/camera-tháng` và GPU sharing, không phải ms/frame.

## 3. Ba lỗi của v1 mà v2 sửa

1. **Thiếu power thống kê (v1 dựa trên 36 event — cập nhật 2026-07-08 sau khi merge FIgLib lên 512 event).** Lượng tử hóa event recall giờ ~0.2%/event thay vì ~2.8%/event — vế "số event" của vấn đề coi như đã giải quyết phần lớn bởi merge dữ liệu, không phải bởi thiết kế thí nghiệm. **Nhưng vế thứ hai không tự hết:** cửa sổ negative của mỗi sequence vẫn chỉ là ~40 phút ngay trước ignition của chính camera đó (offset `-2400..0s`) — dù có 512 event, toàn bộ negative sample vẫn tương quan cao (sát giờ cháy, cùng camera, thiếu đa dạng mây/sương/hoàng hôn/mùa) → `FA/hour` đo trên riêng FIgLib vẫn không đại diện, bất kể có bao nhiêu event. E1a (negative-day harvesting từ HPWREN) vẫn cần thiết cho mẫu số FA/hour, dù không còn là bottleneck số 1 cho power của G0/G1. **Khuyến nghị:** có thể chạy G0 (AUROC pre/post ignition) và G1 (temporal gain N-of-M/EMA) ngay với power thật trên 512 event trước khi chờ negative-day harvesting; giữ E1a cho riêng bài toán FA/hour denominator. → Vẫn dùng bootstrap CI theo event bắt buộc.
2. **So sánh confounded.** N-of-M/EMA ăn scalar confidence; LSTM/Transformer ăn input giàu hơn — nếu LSTM thắng, không biết do model class hay do input. → Sửa bằng thiết kế factorized (E2).
3. **Không có operating-point protocol.** FA/h vs TTD là đường cong; so ở một threshold là cherry-picking. → Sửa bằng AMOC protocol (mục 4).

## 4. Nguyên tắc thí nghiệm bắt buộc

- **AMOC protocol:** quét threshold → curve `TTD` vs `FA/hour`. Điểm báo cáo chuẩn: `TTD @ FA ≤ 1 alarm/camera/ngày` và `@ ≤ 1 alarm/camera/tuần`.
- **Bootstrap CI theo event** (≥1000 resample). Hai method chỉ được kết luận khác nhau khi CI không giao nhau.
- **Detector cache là hạ tầng trung tâm:** detector chạy đúng 1 lần/frame trên GPU, cache JSONL; mọi temporal experiment là offline replay CPU — iterate verifier không tốn GPU.
- **Weak-label ignore band:** frame offset `0..+180s` loại khỏi train/eval frame-level (khói có thể chưa visible ngay sau ignition); TTD event-level vẫn đo từ offset 0.
- **Split kép:** mọi kết quả FIgLib report cả event-split lẫn camera-split (camera lặp giữa sequence: 136 camera-string / 512 seq local).
- **Alarm semantics:** alarm = lần đầu verifier vượt threshold trong sequence; alarm tại offset < 0 → false alarm event; latching (không đếm thêm sau alarm đầu); TTD chỉ đo khi alarm đầu tại offset ≥ 0.
- Mỗi experiment khai báo trước: hypothesis, decision rule, cost, next-if-fail.

## 5. Gates — cây quyết định

- **G0 — transfer check (làm NGAY khi E0 xong):** AUROC của `max_smoke_confidence` (D-Fire detector zero-shot) phân biệt frame pre/post ignition trên FIgLib.
  - `≥ 0.80` → detector đủ tín hiệu → mở G1.
  - `0.60–0.80` → thử imgsz lớn/tiling trước (kéo một phần E4 lên sớm), đo lại; vẫn yếu → nhánh dưới.
  - `< 0.60` → D-Fire không transfer sang khói xa/nhỏ: pivot sang tile-classifier (SmokeyNet-style) hoặc fine-tune trên PYRONEAR-2025. **Không xây temporal trên nền detector mù.**
- **G1 — temporal gain:** N-of-M + EMA vs single-frame trên AMOC. Nếu simple rule không cải thiện TTD@FA → nghi detector trước, đừng đổ lỗi verifier.
- **G2 — learned verifier:** chỉ mở khi G1 pass VÀ có data đủ power (E1). LSTM/tiny-Transformer so trong khung factorized E2.
- **G3 — thermal:** chỉ mở khi RGB temporal track có kết quả ổn định. Lưu ý: FLAME 3 là UAV cận cảnh — khác domain tower; kết luận không transfer trực tiếp sang RQ1, giá trị chính là học multimodal fusion.

## 6. Experiments

- **E0 — D-Fire YOLO baseline** (đang chạy). Vai trò: control, học pipeline, profiling target, proposal generator. Sau khi train xong bổ sung: (a) eval + latency chuẩn; (b) dedup audit train/test bằng perceptual hash — D-Fire gom từ web, nghi near-duplicate inflate mAP; (c) đọc per-class metric có ý thức về imbalance (`fire_only` ~5% train).
- **E0.5 — Transfer check (gate G0).** Detector cache 40362 frame FIgLib (512 sequence, verified 2026-07-08) → AUROC pre/post + AMOC single-frame. Cost: 1 lượt GPU ngắn + CPU. **Experiment quyết định hướng — ưu tiên cao nhất sau E0.**
- **E1 — Data acquisition (P0, nút cổ chai thống kê của toàn bộ plan):**
  - **E1a — negative-day harvesting:** tải N ngày không cháy từ HPWREN archive cho chính các camera local (trộn trời quang / mây / sương / hoàng hôn) → hàng trăm giờ negative đúng domain làm mẫu số FA/hour. Rẻ, không cần label.
  - **E1b — PYRONEAR-2025:** tải khi mở G2 (learned verifier cần data train); FIgLib giữ vai trò test transfer.
- **E2 — Temporal verifier, thiết kế factorized:** `{input: scalar confidence | chuỗi box+conf | ROI embedding} × {aggregator: N-of-M | EMA | logistic-window | LSTM | tiny Transformer}`. Bắt buộc chạy trước: `scalar × {N-of-M, EMA}`. Tất cả trên cùng detector cache. Câu hỏi: gain đến từ model class hay từ input giàu hơn.
- **E3 — Hard-negative flywheel event-level:** chạy pipeline trên negative days (E1a) → lấy FP persistence cao → human review → thêm vào train → retrain → đo lại AMOC. `extract_hard_negatives.py` hiện là frame-level, cần mở rộng sequence-level.
- **E4 — Cost frontier (RQ2):** trục x = `$/camera-tháng` đo thực trên Modal; trục y = `TTD @ FA-budget`. Các điểm: model size (n/s/m), imgsz (640 / 1280 / tile full-res), frame rate (1/min vs 1/2min), cascade bật/tắt. Hypothesis cần kiểm: ở cadence 1/min, model to hơn 10–50× vẫn rẻ → "bắt buộc nano" là dogma nhập từ real-time 30fps.
- **E5 — Task formulation** (bbox vs tile-classifier vs segmentation): kích hoạt khi G0 fail, hoặc sau G2. Cùng split, cùng event metric.
- **E6 — Synthetic smoke ramp (P2, optional):** composite khói tham số hóa (size/contrast tăng dần) lên frame negative của đúng camera → ground-truth onset chính xác tuyệt đối; đo sensitivity của TTD theo kích thước/độ tương phản khói.
- **Parked:** E7 thermal ablation (FLAME 3), E8 RGB-T→RGB distillation, E9 VLM runtime verifier. VLM dùng được NGAY ở vai trò khác: annotation assistant / triage hard-negative (offline, không tính vào inference cost).

## 7. Anchor literature (đối chiếu — không tin số chưa tự reproduce)

- SmokeyNet + FIgLib ([arXiv 2112.08598](https://arxiv.org/abs/2112.08598)): baseline mạnh nhất trên FIgLib, "rivals human performance"; điền số F1/TTD chính xác khi implement so sánh.
- PYRONEAR-2025 ([arXiv 2402.05349](https://arxiv.org/abs/2402.05349)): sequential model tăng recall so với single-frame, precision tương đương — cùng hypothesis với RQ1.

## 8. Thứ tự việc ngay

1. ✅ E0 (`dfire_yolo26n_baseline_full_vram`) — weights final đã đồng bộ + verify (2026-07-08), đã chạy `eval.py accuracy` chính thức trên `test` split (không phải auto-validation `val`): **all P=0.764 R=0.714 mAP50=0.684 mAP50-95=0.404**; `smoke P=0.822 R=0.791 mAP50=0.765 mAP50-95=0.482`; `fire P=0.707 R=0.638 mAP50=0.604 mAP50-95=0.326`; latency mean `21.6ms` p95 `26.7ms` fps `46.4` (GPU `NVIDIA GB10`, imgsz 640, batch 1). **Lưu ý:** mAP50 trên `test` (0.684) thấp hơn rõ rệt so với auto-validation trên `val` (0.76) — cùng model, khác split, cách nhau ~0.08 mAP50; đáng nghi ngờ thêm cho giả thuyết near-duplicate D-Fire train/val đã nêu ở mục 6 (nếu `val` lẫn near-duplicate của `train` thì mAP `val` bị inflate so với `test` thật). Đã chạy `eval.py hard-negatives` trên `test` split song song.
2. ✅ FIgLib index + audit (`dataset.py figlib`, đã gộp từ `figlib_index.py` cũ — xem `agent_context.md` mục "Current code state") — done, re-verified 2026-07-08 sau khi merge `tempo/` + xóa 1 archive rỗng: 40362 frame hợp lệ, 511 sequence, 135 camera (xem mục 2 cho anomaly còn lại: 1 sequence dùng filename convention cũ, không có frame hợp lệ).
3. ✅ Detector cache subcommand (`eval.py detector-cache`, đã gộp từ `figlib_detector_cache.py` cũ) — script sẵn sàng, server `ai2` giờ đã có `.venv` + `ultralytics`/`torch` (CUDA verified hoạt động trên GPU GB10) — có thể chạy ngay, chưa chạy trong lượt này (ưu tiên E0 test-split trước).
4. ✅ AMOC harness + N-of-M + EMA + bootstrap CI (`temporal_eval.py temporal`, giữ tách riêng khỏi `eval.py` — không phụ thuộc `ultralytics`) — done, logic verify bằng synthetic data, chờ detector cache thật.
5. ✅ **G0 AUROC — DONE 2026-07-09.** `eval.py detector-cache` chạy thật trên toàn bộ FIgLib (`40361/40362` frame, 1 frame lỗi file rỗng đã skip có log, xem `agent_context.md`), viết mới `gate_g0_auroc.py` (Mann-Whitney U rank-sum bằng numpy, tie-corrected, không thêm dependency `scipy`/`scikit-learn`; ignore-band `180s`; bootstrap CI 1000 resample cả event-split và camera-split). **Đã merge vào `temporal_eval.py` subcommand `g0` (2026-07-09, gọn codebase, cùng profile dependency numpy/rich, không torch/ultralytics) — file `gate_g0_auroc.py` cũ đã xóa, output số học verify identical.** Kết quả thật:
   - `AUROC(max_smoke_confidence) = 0.7017` — event-split CI95 `[0.6846, 0.7189]`, camera-split CI95 `[0.6830, 0.7195]` (hai CI gần trùng, đều nằm chắc trong khoảng `0.60-0.80`, không giáp biên).
   - `AUROC(max_fire_confidence) = 0.509` (~random, đúng như dự đoán — fire D-Fire là cận cảnh, không transfer sang camera tháp xa).
   - `AUROC(max_any_confidence) = 0.705` (gần bằng smoke vì smoke chiếm ưu thế tín hiệu).
   - **Gate decision: `marginal` (0.60-0.80)** → theo cây quyết định mục 5: thử `imgsz` lớn hơn hoặc tiling trước, đo lại G0 — **chưa kết luận pass/fail cuối cùng**, cần chạy lại bước tiếp theo trước khi quyết định mở G1 hay pivot.
   - Report đầy đủ: `artifacts/smoke_fire_detection/gate_g0_auroc.json`.
6. Tiếp nhánh "marginal" (chưa chạy, chưa confirm process): `eval.py detector-cache` với `imgsz` lớn hơn (1280) trên FIgLib → `temporal_eval.py g0` lại → so sánh AUROC trước/sau.
7. Song song: quyết định cách tải negative days từ HPWREN (E1a) — vẫn cần user quyết định cách tải cụ thể, chưa làm.
8. Chỉ khi mở G2: tải PYRONEAR-2025, chạy factorized E2.

## 9. Dataset khảo sát thêm (2026-07-08) — không đi thu thập toàn bộ dataset liên quan

Nguyên tắc: chỉ tải dataset khi một gate/experiment cụ thể ở trên gọi tên nó và sắp chạy tới — không tải "phòng khi có ích". Đã khảo sát 5 dataset user đề xuất (chi tiết số liệu ở `agent_context.md` mục "Candidate dataset khác"):

- **`pyro-sdis`** (HuggingFace, Apache-2.0, không gate, 3.28GB): duy nhất đáng tải sớm — nhẹ, cùng domain camera cố định như FIgLib. **Đã tải 2026-07-08** theo yêu cầu trực tiếp của user (`datasets/smoke_fire_detection/pyro-sdis/`) — nhưng format thật là parquet (ảnh+annotation embedded), KHÔNG phải folder `images/`+`labels/` như `data.yaml` mô tả, và `class_id` trong annotation là `1` chứ không phải `0` (mâu thuẫn với khai báo `nc=1`). Cần converter riêng trước khi dùng được — xem chi tiết schema ở `agent_context.md`. Vai trò dự kiến (chưa kích hoạt, chưa gate nào gọi tên): nguồn mở rộng smoke-class hoặc detector thứ hai để đối chiếu G0.
- **`PyroNear-2024`**: gần như tập con của PyroNear-2025 đã có trong plan — không tải riêng.
- **`DetectiumFire`**: giá trị khác biệt là vision-language pairs, map với P3 (VLM verifier, parked) — không phục vụ P0 hiện tại. Nếu sau này cần: tải thẳng trên server Linux (có internet trực tiếp) bằng `kaggle` CLI, không cần qua máy Windows.
- **`FLAME 3` (full 6-burn)**: đã parked theo G3 (UAV thermal, khác domain tower camera).
- **`FireSentry`**: task là spread forecasting, không phải presence detection — lệch RQ hiện tại; chưa xác nhận có dataset public.
- **`GWFP`**: tác giả công bố "sẽ public khi paper accept" — hiện chưa có link tải, không phải vấn đề gate.

Kết luận: không có dataset nào trong nhóm trên đủ lý do vượt priority của việc đang bị block (Step 1 + Step 5 ở mục 8).
