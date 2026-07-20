# Plan hậu-train: benchmark 2×2 — bản thực thi (chốt)

- Base: plan gốc agent khác (2026-07-20) + phản biện/amend cùng phiên. Bản này tự chứa, không cần đọc lại 2 bản trước.
- Đọc trước khi làm: `agent_context.md` → `research_plan.md` mục 2, 8 (bước 13), 13d → `research_toolbox.md` khi cần failure-mode menu.
- Platform: tự phát hiện OS/GPU bằng command (`uname -a`, `nvidia-smi -L`), không giả định. Server Linux: kích hoạt `.venv` trước mọi `python`/`pip`.

## Mục tiêu

- Ma trận chính: `YOLO26x×Pyro-SDIS`, `RF-DETR-L×Pyro-SDIS`, `YOLO26x×D-Fire`, `RF-DETR-L×D-Fire`.
- Control (không tranh winner): `YOLO26n×D-Fire`, `pyronear/yolov8s` (leakage-caveat).
- Output: winner sạch trên FIgLib dev + temporal operating point khóa + final camera-held-out mở đúng 1 lần + report tĩnh.
- Ngoài scope: không train lại, không dataset mới, không near-field RQ5, không architecture ablation/tiling/learned verifier/PYRONEAR-2025/E1a, không hash/checksum.

## Giai đoạn 0 — audit + mốc

- Commit git trạng thái artifact hiện tại làm mốc trước khi chạy gì (repo đang có file xóa-trong-index nhưng còn trên disk — dọn theo git status thật, không tự `reset --hard`).
- Audit checkpoint:
  - YOLO26x/Pyro-SDIS: `runs/yolo26x_pyro_sdis/weights/best.pt`.
  - YOLO26x/D-Fire: `runs/yolo26x_dfire/weights/best.pt`.
  - RF-DETR-L/Pyro-SDIS, RF-DETR-L/D-Fire: `checkpoint_best_total.pth` tương ứng — dùng đúng file này, không tự chọn lại regular/EMA.
  - `pyronear/yolov8s`: chưa có local — tải về (HF hub hoặc `modal_app.py:download_pyronear`, hoặc thẳng server có internet), pin revision, `imgsz=1024` đã verify qua ONNX metadata trước đó (tra git history nếu cần lại cách làm).
  - YOLO26n control: `runs/dfire_yolo26n_baseline_full_vram/weights/best.pt`.
- Audit split: `figlib_split_manifest.json`, seed `20260707`, dev 359 seq/108 camera, final 152 seq/35 camera, overlap 0. Expected frame lấy từ index, không hard-code.
- Audit cache cũ `figlib_detector_cache_yolo26x_pyro_sdis_dev.jsonl`:
  - `model_weights`/`candidate_revision` trỏ tên cũ `yolo26x_pyro_sdis_budget9` (dir đã đổi tên `yolo26x_pyro_sdis`) — KHÔNG rewrite file 29MB này, `frame_path_index` đã tương đối và đủ dùng. Chỉ ghi mapping `budget9→yolo26x_pyro_sdis` vào sidecar `.meta.json` mới.
  - Đã audit đủ (28,360/28,361, 1 lỗi JPEG rỗng có log) → tái dùng, không chạy lại GPU.
- Candidate ID chốt: `yolo26x_pyro_sdis`, `rfdetr_large_pyro_sdis`, `yolo26x_dfire`, `rfdetr_large_dfire`, `yolo26n_dfire_control`, `pyronear_yolov8s_reference`.
- D-Fire label: `yolo26n_dfire_control` = label gốc. `yolo26x_dfire`, `rfdetr_large_dfire` = label đã relabel (khác control). Không relabel thêm (user quyết 2026-07-20).

## Giai đoạn 1 — mở rộng inference/cache đa backend

- Runtime chính: server Linux `.venv`, GPU local (đã có `rfdetr==1.8.3` + `ultralytics 8.4.90` + torch cài sẵn; cache dev cũ cũng chạy ở đây, không phải Modal). Modal L4 CHỈ dùng cho profiling latency/cost chuẩn hóa (330 frame/candidate) — không phải runtime cache chính (bản gốc giả định ngược, sai: tốn công upload dev subset không cần thiết khi GPU local đã chạy được).
- Mở rộng `eval.py detector-cache`: thêm `--backend {yolo,rfdetr}` (default `yolo`, giữ CLI cũ), `--candidate-id`, `--class-names smoke` hoặc `smoke,fire` map theo index, `--imgsz` dùng chung.
- Backend RF-DETR: `RFDETRLarge.from_checkpoint(...)`, batch bằng list path, `include_source_image=False`, `threshold=0.05`, không NMS ngoài (metadata `iou=null`, `postprocess=native`). Pilot phải verify: box xyxy ở hệ quy chiếu ảnh gốc, không phải input đã resize.
- Schema chung: `class_id`, `class_name`, `confidence`, `xyxy`, `max_smoke_confidence`, `max_fire_confidence`, `max_any_confidence`. Empty detection → 3 max-score = `0.0`. Batch lỗi → retry từng frame; frame lỗi → skip + errors sidecar merge theo frame key, resume không duplicate.
- Sidecar `<cache>.meta.json` mỗi cache: candidate ID, backend, checkpoint path tương đối, resolution/conf/iou hoặc postprocess=native, framework/torch/cuda version, provider+GPU+VRAM+batch, command chạy, index/written/failed rows, elapsed. Không hash.
- Eval COCO-mAP cho RF-DETR (thiếu, viết mới): package có `rfdetr/evaluation/coco_eval.py` nội bộ — đọc source trước khi gọi, không đoán API eval-only (vd `train(epochs=0,...)`) mà chưa verify trong `rfdetr/main.py`/trainer. Dùng để chạy test D-Fire + val Pyro-SDIS cho RF-DETR (hiện chỉ YOLO có `eval.py accuracy`).
- Modal wrapper: thêm image RF-DETR riêng (pin `rfdetr==1.8.3`), function profiling dùng chung L4 cho cả 2 backend. Modal chết → fallback toàn bộ profiling trên GB10, không trộn hardware giữa candidate.

## Giai đoạn 2 — source-domain eval + pilot

- Tái dùng: YOLO26x/D-Fire (`test_metrics.json`), YOLO26x/Pyro-SDIS (`results.csv`), RF-DETR-L/D-Fire (`metrics.csv` per-epoch, lấy epoch best). RF-DETR-L/Pyro-SDIS: chưa có, chạy val COCO-mAP mới.
- Không so mAP trực tiếp D-Fire vs Pyro-SDIS (khác class composition).
- Resolution confound RF-DETR/D-Fire (train@800, muốn cache FIgLib@1280 đồng nhất) — xử lý trước khi tin kết quả:
  - Pilot bắt buộc: load checkpoint train-800, predict ở 1280, xác nhận box sane (không NaN, không degenerate toàn-frame).
  - Tính lại architecture-effect cột D-Fire ở CẢ 1280 và 800 (800 = 2 model D-Fire cùng resolution, sạch hơn cho contrast).
  - Dấu/độ lớn effect đổi giữa 1280 và 800 → gắn cờ `resolution-confounded` lên interaction/dataset-effect liên quan, không diễn giải như kiến trúc thắng thật.
  - Pilot fail (box hỏng ở 1280) → primary matrix chuyển hẳn về native-resolution (Pyro-SDIS@1280, D-Fire@800) cho toàn bộ 4 candidate sạch, ghi rõ lý do.
- Pilot 30 frame/checkpoint trước full cache: nhiều camera, pre/near/post-ignition, check class map, batch alignment, OOM, resume, error handling.
- Profiling chuẩn (Modal L4, batch 1, 30 warmup + 300 measured × 3 lượt): mean/p50/p95, throughput, peak VRAM. `$/camera-tháng` chỉ dùng tie-break, ghi ngày/nguồn giá Modal.

## Giai đoạn 3 — cache FIgLib dev

- Resolution chính `1280` cho 4 candidate sạch + YOLO26n control; Pyronear `1024` (main) + `1280` (phụ). Resolution phụ D-Fire: `800` cho cả YOLO26x và RF-DETR-L (xem Giai đoạn 2).
- **User confirm 2026-07-20: giữ `1280` làm resolution winner cho cả 4 candidate sạch** (kể cả 2 model D-Fire train@800). Cache `@800` D-Fire vẫn chỉ diagnostic, KHÔNG vào winner rule. Việc cần làm ở Giai đoạn 4: so trực tiếp AUROC@1280 vs AUROC@800 của 2 model D-Fire — nếu @800 vượt @1280 rõ rệt, phải nêu rõ trong report là 1 giới hạn của winner rule (không tự đổi rule ngược, không giấu).
- Sau mỗi cache: alignment với dev index, missing = lỗi đã log, unique frame key, metadata đồng nhất, không lẫn final row, không path tuyệt đối (file MỚI — cache cũ giữ nguyên theo Giai đoạn 0).
- Không chạy lại nếu resume đã đủ.

## Giai đoạn 4 — phân tích 2×2 + chọn winner

- Mở rộng `temporal_eval.py`: thêm `compare-matrix` nhận nhiều `candidate_id=cache_path`, giữ `compare-candidates` cũ, output 1 JSON tổng.
- Metric chính: `AUROC(max_smoke_confidence)`, ignore band `0..+180s`, bootstrap 1000 seed `20260707`, paired theo event + cluster theo camera.
- Bắt buộc: AUROC+CI mỗi candidate; 6 pairwise trong 4 candidate sạch; architecture effect (RF−YOLO) mỗi dataset; dataset effect (Pyro−D-Fire) mỗi kiến trúc; interaction; controls báo riêng không vào winner pool.
- Leader/winner rule (chốt, thay câu mơ hồ bản gốc):
  - Leader = AUROC point cao nhất trong 4 candidate sạch.
  - Leader thắng 1 đối thủ khi: event-bootstrap lower CI của `ΔAUROC>0` VÀ median camera-bootstrap delta cùng chiều `>0`.
  - Leader thắng hết 3 đối thủ → winner rõ. Không thắng hết → "top set" = leader + mọi candidate không thua leader theo rule trên → tie-break bằng latency/cost S0 (Modal L4, batch 1, resolution chính) trong top set.
  - Không multiple-comparison correction (rule pre-registered) — ghi caveat này trong report.
  - Control đứng đầu tuyệt đối (yolo26n hoặc Pyronear cao hơn cả 4 sạch) → vẫn khóa winner trong nhóm sạch cho final; report thẳng "fine-tune không vượt control/reference" như 1 finding, không giấu.
- Gate: AUROC≥0.80 → detector gate pass; <0.80 → vẫn chọn best để final unbiased, không gọi pass.
- Diagnose bổ sung (gần free, CPU-only): chạy `temporal_eval.py diagnose` trên từng primary cache, so silent/confused-sequence-count với mốc lịch sử D-Fire@1280 (166 seq≥0.9 / 131 silent). Câu hỏi giá trị nhất: train đúng domain có cứu nhóm silent không.
- Spatial-persistence overlay: `research_plan.md` mục 8 bước 12 đã cam kết giữ overlay cho tier FA≤1/tuần + zero-FA, nhưng subcommand đã bị trim khỏi `temporal_eval.py`. Chọn 1: (a) restore từ git history, chạy cho winner; (b) chính thức hủy cam kết, sửa `research_plan.md`. Không được im lặng bỏ qua.
- Human-visibility stratification (13.0: visible=23/fire_only=5/ambiguous=5/not_visible=7): `human_review_labels.json` đã bị dọn khỏi artifacts — restore từ git history nếu dùng phân tầng phụ, hoặc bỏ và sửa dangling reference ở `research_plan.md`.
- Dev temporal: single-frame, N-of-M, EMA trên 6 primary cache; 2 budget (≤1 FA/camera/ngày, ≤1/tuần); operating point = đạt FA budget, recall cao nhất, tie→TTD thấp nhất, tie tiếp→FA thấp nhất.
- Tạo `figlib_dev_winner_lock.json`: winner ID, decision rule+metrics, resolution/conf/postprocess, 2 operating point khóa, allowlist baseline/Pyronear được chạy final. Không hash.

## Giai đoạn 5 — final camera-held-out, một lần

- Naming: "final" trong plan = `eval.py --split test` trong code (không phải `dev`/`all`).
- Chỉ chạy: winner sạch, YOLO26n control, Pyronear reference. Không chạy 3 losing clean candidate.
- Gate an toàn tối thiểu (layer mới — failure mode cụ thể: đốt final one-shot bằng candidate sai; chấp nhận khi duyệt plan này): cache command đọc `figlib_dev_winner_lock.json` khi `--split test`, reject candidate không nằm winner+allowlist trước khi inference. Vài dòng check, không thêm manifest phụ.
- Final analysis: AUROC+CI event/camera, pairwise winner vs baseline, Pyronear leakage-caveat, replay đúng operating point khóa (không quét lại threshold), TTD/recall/precision/FA per hour-window+CI.
- Quyết định: final winner≥0.80 → gate pass, viết proposal bước sau (E1/G2), không tự chạy; <0.80 → gate fail, viết G4 proposal xếp hạng (data-only scale, P2/stride-4, P2↔P5 compute-match, adaptive ROI, E5), không tự chạy.

## Báo cáo, artifact, cleanup

- Static report mới `tasks/smoke_fire_detection/docs/model_benchmark_2x2.md`: protocol, source-domain metrics, dev 2×2 matrix, pairwise/factorial, winner rule, resolution sensitivity, latency/VRAM/S0, final metrics, operating points, failure slices, caveat (weak label, ignore band, negative denominator, corrupt JPEG, Pyronear leakage, resolution-confound nếu có).
- Update: `research_plan.md` (13d/13e done, số chốt inline, next gate); `agent_context.md` (winner, artifact sống, blocker, next; caveman); `external_assets.md` (checkpoint/cache/report giữ); `CHANGELOG.md` (append GMT+7, command đầy đủ, file trực tiếp/gián tiếp, không phục hồi phần user đã xóa).
- Retention: giữ 4 primary dev cache sạch, 2 control dev cache, winner/baseline/Pyronear final cache, source eval, comparison, winner-lock, temporal, report. Xóa pilot, cache lỗi dở, temp shim, profiling raw thừa, secondary-resolution cache sau khi số đã chốt. Giữ errors sidecar khi lỗi thật còn tồn tại. Không xóa checkpoint final.
- Không comment code, không `__init__.py`/scaffold/manifest phụ/cost ledger.

## Kiểm thử

- 1 file `unittest` stdlib: schema YOLO/RF-DETR giống nhau, class map 1/2 class, empty detection, batch alignment mismatch + fallback từng frame, resume không duplicate, error merge, path portability, AUROC tie handling, event/camera bootstrap, factorial contrast, winner/tie/cost rule, final-lock reject losing candidate.
- GPU integration: 4 checkpoint load được, pilot 30 frame pass, RF-DETR batch list đúng số kết quả, resume lần 2 ghi 0 frame mới.
- Data acceptance: dev/final cache keyset khớp index trừ lỗi đã log, camera overlap dev/final=0, không NaN/Inf/duplicate/absolute path (file mới).
- Statistical acceptance: cùng ignore band/seed/bootstrap; controls không thể thành winner; final không tái-tune threshold; report khớp JSON.
- Repo acceptance: `git diff` chỉ gồm thay đổi cần thiết; không đụng root `docs/`; không cache/scaffold rác; CHANGELOG đầy đủ.

## Giả định khóa

- Metric chọn winner: raw smoke AUROC; temporal/cost là tie-break.
- Main resolution 1280 cho 4 candidate sạch (trừ khi Giai đoạn 2 pilot fail → chuyển native-resolution).
- RF-DETR: dùng checkpoint winner có sẵn, không tự chọn lại EMA/regular.
- Pyronear: reference-only, leakage-caveat, không tranh winner.
- Runtime cache: server GB10 local; Modal: chỉ profiling.
- Near-field RQ5: parked, không đụng.
