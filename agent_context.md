- Vai trò: bộ nhớ ngắn hạn agent; dữ kiện sống; không command log.
- Repo: lab CV inference; core portable Windows/Linux; Modal wrapper tách core; không hard-code path/OS/GPU/user.
- Platform: Windows dùng `py`; Linux server dùng `.venv/bin/python`/pip.
- Tracking: append `CHANGELOG.md`; GMT+7; đủ command; file trực tiếp/gián tiếp.
- Không hash/checksum/fingerprint dataset, split, checkpoint, artifact, file list nếu user không yêu cầu rõ.
- Root `docs/`: không đọc/sửa/xóa nếu user không yêu cầu rõ.
- Scope: `tasks/smoke_fire_detection/`; dataset `datasets/smoke_fire_detection/`; artifact `artifacts/smoke_fire_detection/`.
- Plan đang chạy: `tasks/smoke_fire_detection/docs/post_train_benchmark_plan.md` (2×2 benchmark YOLO26x/RF-DETR-L × Pyro-SDIS/D-Fire, eval trên FIgLib). Đọc file này trước khi làm tiếp Giai đoạn 4+.

## Dataset/checkpoint

- D-Fire: train 15,500; valid 1,721; test 4,306; class smoke=0, fire=1.
- Pyro-SDIS YOLO: train 29,537; val 4,099; class smoke=0; nguồn 1280×720.
- FIgLib: 40,362 frame hợp lệ; 511 folder; 143 camera; split camera-disjoint seed 20260707; dev 359 seq/108 cam, final 152 seq/35 cam, overlap 0.
- Protocol fair: Pyro-SDIS resolution 1280, D-Fire resolution 800, cả 2 đều 20 epoch seed 20260707. Batch/optimizer/scheduler/augmentation native từng framework, không ép iso-config. Kiến trúc YOLO26x/RF-DETR-L gốc, chỉ đổi train/runtime config + class count + resolution.
- D-Fire label: `yolo26n_dfire_control` = label gốc. `yolo26x_dfire` + `rfdetr_large_dfire` = label đã relabel (khác control). Không relabel thêm (user quyết 2026-07-20) — không so trực tiếp mAP domain-gốc giữa control và 2 model fresh để suy diễn data-quality effect.
- 6 candidate ID: `yolo26x_pyro_sdis`, `rfdetr_large_pyro_sdis`, `yolo26x_dfire`, `rfdetr_large_dfire` (4 candidate sạch, tranh winner); `yolo26n_dfire_control`, `pyronear_yolov8s_reference` (control/reference, KHÔNG tranh winner).
- Weight giữ:
  - YOLO26x/Pyro-SDIS: `runs/yolo26x_pyro_sdis/weights/best.pt`; val mAP50-95 `0.4949`.
  - RF-DETR-L/Pyro-SDIS: `runs/rfdetr_large_pyro_sdis/checkpoint_best_total.pth`; val mAP50-95 `0.4627`.
  - YOLO26x/D-Fire: `runs/yolo26x_dfire/weights/best.pt`; test mAP50-95 `0.40957`.
  - RF-DETR-L/D-Fire: `runs/rfdetr_large_dfire/checkpoint_best_total.pth`; test mAP50-95 `0.4801` (smoke 0.5548, fire 0.4054).
  - YOLO26n/D-Fire control: `runs/dfire_yolo26n_baseline_full_vram/weights/best.pt`; test mAP50-95 `0.404`.
  - Pyronear yolov8s reference: `artifacts/smoke_fire_detection/pyronear_yolov8s.pt` (pin sha256 có sẵn, revision `cd075ce`); leakage-caveat, không rõ tập train.
- Notebook giữ: `colab_train_dfire_yolo26x.ipynb`, `kaggle_train_dfire_rfdetr.ipynb`, `kaggle_train_pyro_sdis_rfdetr.ipynb`.
- Code giữ: `dataset.py`, `eval.py`, `temporal_eval.py`, `modal_app.py`, `report/demo_inference.py`, `test_eval.py`, `test_temporal_eval.py`.

## Giai đoạn 0-2 — DONE 2026-07-20

- G0: audit checkpoint/split/cache cũ khớp plan; tải pyronear_yolov8s.pt; chốt 6 candidate ID.
- G1: `eval.py` thêm backend RF-DETR (`detector-cache`, `rfdetr-accuracy`, `profile`); `modal_app.py` thêm image+function RF-DETR; `temporal_eval.py` thêm `compare-matrix`. Test: `test_eval.py`(22) + `test_temporal_eval.py`(9) = 31, tất cả pass.
- G2: Pilot resolution-confound RF-DETR/D-Fire (train@800, predict@1280 trên FIgLib) — PASS, không NaN/degenerate.
- **Winner resolution — user confirm 2026-07-20: `1280` cho cả 4 candidate sạch** (kể cả 2 model D-Fire train@800). Cache `@800` D-Fire chỉ diagnostic (architecture-effect cross-check + cờ resolution-confound), KHÔNG vào winner rule. Việc mở cần làm ở G4: so AUROC@1280 vs @800 của 2 model D-Fire — @800 vượt rõ rệt thì phải nêu thẳng trong report là giới hạn của winner rule, không giấu, không tự đổi rule.
- Gate an toàn G5 (`eval.py detector-cache --split test`): đọc `figlib_dev_winner_lock.json`, reject candidate không phải winner/allowlist. Test `FinalSplitGateTests` (4 case) đã pass.
- Profiling Modal L4 (batch 1, 30 warmup + 300 measured×3, resolution chính) — `artifacts/smoke_fire_detection/profile_<candidate_id>.json`:
  - `yolo26x_pyro_sdis`(1280): mean 82.26ms fps 12.81 VRAM 798.7MB.
  - `rfdetr_large_pyro_sdis`(1280): mean 141.18ms fps 7.08 VRAM 461.5MB.
  - `yolo26x_dfire`(1280): mean 77.26ms fps 12.94 VRAM 798.5MB.
  - `rfdetr_large_dfire`(1280): mean 136.05ms fps 7.35 VRAM 461.5MB.
  - `yolo26n_dfire_control`(1280): mean 41.73ms fps 23.97 VRAM 142.6MB.
  - `pyronear_yolov8s_reference`(1024): mean 34.82ms fps 28.72 VRAM 141.4MB.
  - RF-DETR-L ~1.7-1.9x chậm hơn YOLO26x cùng resolution, VRAM thấp hơn (461MB vs 798MB). $/camera-tháng chưa tính, chỉ tính khi cần tie-break thật.
- Modal volume còn run-dir cũ trước dọn local (`yolo26x_pyro_sdis_budget9`, `_smoke_locked`, `_archive_benchmark`, `_warmstart`, `_resume_gate`, `resume_gate`) — chưa xóa, ngoài scope plan, cần user xác nhận riêng.

## Giai đoạn 3 — cache FIgLib dev (đang chạy, đã crash+relaunch 1 lần)

- Job chạy nền trên GB10 qua systemd scope (linger bật cho tts01) — sống độc lập SSH/laptop. Script `/tmp/run_giai_doan3_cache.sh` (scratch, không thuộc repo), log `/tmp/giai_doan3_cache.log`, `--resume` an toàn (ghi theo batch, flush ngay, không mất dữ liệu khi crash).
- **CRASH 2026-07-20 ~18:55 GMT+7:** unit `giai-doan3-cache.scope` chết âm thầm giữa chừng `rfdetr_large_dfire` (dừng ở 24,383/28,360 dòng, dòng cuối JSON hợp lệ, không hỏng). Nguyên nhân KHÔNG xác định chắc chắn — không có quyền đọc `dmesg`/kernel log (không có NOPASSWD sudo), không thấy log lỗi user-level, không loại trừ OOM-kill kernel. Phụ: lúc start unit có lỗi `Failed to add ... inotify watch ... No space left on device` (cạn inotify instance, KHÔNG phải cạn disk — disk còn 2.9T) → nghi ngờ systemd mất theo dõi cgroup, khiến `systemctl status` báo `active (running)` dù `Tasks: 0` (tiến trình con đã chết thật) — **bài học: đừng tin `systemctl status` một mình, phải cross-check `pgrep`/mtime file cache.**
- **Relaunch 2026-07-20 19:23 GMT+7:** stop unit cũ (`systemctl --user stop giai-doan3-cache.scope`), start unit mới **`giai-doan3-cache-r2.scope`** cùng script (đã sửa thứ tự, xem dưới). Resume không mất dữ liệu (24,383 dòng cũ giữ nguyên).
- **Đổi thứ tự script (user duyệt 2026-07-20 19:15 GMT+7):** do ràng buộc thời gian, 2 cache D-Fire `@800` (`yolo26x_dfire`, `rfdetr_large_dfire` — cần cho câu hỏi bắt buộc so AUROC@1280 vs @800) đã dời lên **trước** `yolo26n_dfire_control` + `pyronear_yolov8s_reference` (không cần cho winner/câu hỏi mở). Thứ tự run mới: `rfdetr_large_pyro_sdis`(resume no-op) → `yolo26x_dfire`(resume no-op) → `rfdetr_large_dfire`@1280(tiếp tục từ 24,383) → `yolo26x_dfire`@800 → `rfdetr_large_dfire`@800 → `yolo26n_dfire_control`@1280 → `pyronear_yolov8s_reference`@1024 → `pyronear_yolov8s_reference`@1280.
- Trạng thái dòng cache (tổng 28,360/candidate = 28,361 index - 1 frame lỗi), tính tới 2026-07-20 19:23 GMT+7:
  - `yolo26x_pyro_sdis`, `rfdetr_large_pyro_sdis`, `yolo26x_dfire`(@1280): DONE (28,360 mỗi file).
  - `rfdetr_large_dfire`@1280: ĐANG CHẠY LẠI (resume từ 24,383/28,360, ~86%).
  - D-Fire `@800` (2 file), `yolo26n_dfire_control`, `pyronear_yolov8s_reference`(1024+1280): CHƯA CHẠY, theo thứ tự mới ở trên.
- 4 candidate sạch @1280 (đủ để tính winner) hoàn tất ngay khi `rfdetr_large_dfire`@1280 xong — không cần chờ D-Fire@800/control/reference.
- **User quyết định 2026-07-20 19:20 GMT+7 (QUAN TRỌNG, áp dụng khi viết report):** nếu AUROC@800 của `yolo26x_dfire`/`rfdetr_large_dfire` vượt rõ AUROC@1280 — phải viết thẳng đây là **oversight** trong quy trình (winner đã khóa bằng @1280 TRƯỚC KHI có số @800 để so), không diễn giải giảm nhẹ, KHÔNG tự đổi ngược winner đã chọn.

## Quyết định user 2026-07-20 khác (chưa thực thi, không chặn Giai đoạn 4)

- RQ5 near-field: duyệt mở, sequenced sau khi khóa winner RQ1 — `research_plan.md` mục 10.
- SmokeyNet reference: ưu tiên tìm checkpoint public, 2 lead: `gitlab.nrp-nautilus.io/anshumand/pytorch-lightning-smoke-detection`, `github.com/iperezx/sage-smoke-detection` — chưa tự tải/verify.
- Kiến trúc triển khai đa phong bì (routing tĩnh theo camera + specialist + bộ xác nhận chung) — chốt `research_plan.md` mục 2.3.
- Roadmap ưu tiên P0-P3 — `research_plan.md` mục 8.1.

## SOTA facts đã research 2026-07-20 (dùng cho report, không cần research lại)

- Pyronear/pyro-sdis: model card `pyronear/yolov8s` KHÔNG công bố mAP/P/R/F1 nào (verify trực tiếp). Không có số FIgLib/temporal nào công bố cho model này. Paper khác (PYRONEAR-2025, arXiv 2402.05349, KHÁC checkpoint/dataset) báo P/R/F1/TTD: single-frame P0.805/R0.775/F1 0.790/TTD 1.76min; sequential CNN-LSTM P0.793/R0.853/F1 0.822/TTD 1.17min — không đồng nhất protocol, chỉ tham chiếu gián tiếp.
- FIgLib gốc (Dewangan/SmokeyNet, arXiv 2112.08598, Remote Sensing 2022 14(4):1007): dùng Acc/Prec/Recall/F1/TTD, KHÔNG có AUROC. Số chính (2-frame): Acc 83.49% F1 82.59% Prec 89.84% Rec 76.45% TTD 3.12min; human baseline Acc 78.5%/F1 82.8%. Split by-fire 144/64/62 train/val/test (315 sequence tổng, 24,800 ảnh 224×224 tile) — KHÁC hẳn split camera-disjoint 510-sequence/dev-final hiện tại và khác metric (không AUROC) → không so trực tiếp số, chỉ nêu như neo định tính.
- Paper khác cùng split SmokeyNet (arXiv 2311.10116): Acc 84.67%/F1 84.05%. Baldota et al. (arXiv 2212.14143, split gần giống 131/63/61) tự reproduce SmokeyNet ra THẤP hơn paper gốc (Acc 80.12/F1 77.52) — bằng chứng biến thiên reproduction thật, không phải lỗi.
- D-Fire baseline literature: mọi số tìm được đều là mAP50 (60.6-80.9%), KHÔNG có mAP50-95 công khai nào để đối chiếu trực tiếp. Số của mình: YOLO26x mAP50 test `0.7287` (trong range); RF-DETR-L mAP50 test `0.8194` (nhỉnh hơn biên trên range, hợp lý). YOLO26x/Pyro-SDIS val mAP50 `0.7450`; RF-DETR-L/Pyro-SDIS val mAP50 `0.7502` — 2 kiến trúc gần bằng nhau ở source-domain Pyro-SDIS.
- Latency GPU: không tìm được số latency wildfire-smoke cụ thể trên T4/L4/A10. Benchmark chính hãng RF-DETR-L @704px T4 TensorRT10.4 FP16 batch1 = 6.8ms; YOLO26-X @640px cùng điều kiện = 9.6ms (Roboflow/Ultralytics docs). Số của mình (L4, 1280px, PyTorch eager không TensorRT): RF-DETR-L 136-141ms, YOLO26x 77-82ms — cao hơn ~20x/~8-9x, giải thích hợp lý bằng resolution cao hơn (~3-4x pixel) + thiếu TensorRT/FP16 optimize (chưa gọi `model.optimize_for_inference`), KHÔNG phải do L4 yếu hơn T4. Ghi caveat này trong report, không so trực tiếp số latency.

## Làm việc từ local clone (không GPU, không SSH server) — bổ sung 2026-07-20 20:51 GMT+7

- Bối cảnh: server `ai2` (GB10) chỉ truy cập được qua mạng nội bộ công ty, không VPN. User mang laptop về nhà — mất kết nối server tới khi quay lại mạng công ty. Đã commit+push tiến độ lên GitHub (`github.com/KhanhAIDS/cv-inference-lab`, remote `origin`) để làm tiếp được từ local clone.
- **Có sau khi `git pull`:** toàn bộ code, docs, `agent_context.md`/`CHANGELOG.md`, VÀ cache detection JSONL đã track trong git (`artifacts/smoke_fire_detection/figlib_detector_cache_*.jsonl` + `.meta.json`/`.errors.json`, `figlib_index.jsonl`, `figlib_split_manifest.json`, `profile_*.json`, `runs/*/results.csv`/`test_metrics.json`/`training_config.json` — KHÔNG gồm checkpoint nhị phân).
- **KHÔNG có sau khi `git pull`** (gitignore: `datasets/`, `*.pt`/`*.pth`/`*.onnx`/`*.ckpt`/`*.engine`, `.venv/`): dataset D-Fire/Pyro-SDIS/FIgLib gốc, mọi checkpoint (`best.pt`, `checkpoint_best_total.pth`, `pyronear_yolov8s.pt`), virtualenv. Đây là setup cố ý (xem `external_assets.md`) — không phải thiếu sót.
- **Hệ quả quan trọng:** `temporal_eval.py` (`compare-matrix`, `temporal`, `g0`, `diagnose`) **không import torch/ultralytics/rfdetr — chỉ cần `numpy`+`rich`** (verify: đầu file chỉ có `argparse,json,random,statistics,collections,pathlib,numpy,rich`). Nghĩa là **toàn bộ phân tích Giai đoạn 4 (bước 2-7 ở mục "Next" dưới) chạy được 100% trên máy local (Windows, không GPU)**, miễn có file cache JSONL cần thiết (đã có trong git) — KHÔNG cần server/SSH/GPU cho các bước này.
- **KHÔNG chạy được ở local** (thiếu checkpoint + dataset + GPU + `.venv` với torch/ultralytics/rfdetr): `eval.py detector-cache` để sinh cache MỚI — tức 2 cache D-Fire@800 còn dở, `yolo26n_dfire_control`, `pyronear_yolov8s_reference` (xem Giai đoạn 3) VẪN phải chờ server. Cache 4 candidate sạch @1280 đã xong VÀ đã push — đủ để làm winner-lock ở local ngay.
- **Setup local (Windows, theo CLAUDE.md — dùng `py` không dùng `python`):** `py -m pip install numpy rich` (2 dependency duy nhất của `temporal_eval.py`). KHÔNG cần cài `torch`/`ultralytics`/`rfdetr` cho các lệnh `temporal_eval.py` (chỉ cần nếu muốn chạy `eval.py`, mà `eval.py` cũng không chạy được vì thiếu checkpoint/dataset local).
- Việc thực thi được ngay ở local: **bước 2 (compare-matrix, rerun nếu file `figlib_dev_compare_matrix.json` chưa kịp push xong từ server), bước 3-5 (đọc winner, tính operating point, viết `figlib_dev_winner_lock.json`), bước 7 (viết report tĩnh — dùng SOTA facts + source-domain metrics đã có sẵn trong file này)**.
- Việc PHẢI chờ quay lại server: **bước 6 (so AUROC@800 vs @1280 — cache `@800` D-Fire chưa xong)**, và toàn bộ Giai đoạn 5 (final test split, one-shot, cần checkpoint+dataset+GPU+user duyệt tường minh).
- Khi quay lại server: `git pull` để lấy report/winner-lock đã làm ở local (nếu đã commit), rồi tiếp tục bước 6 + dọn artifact theo Giai đoạn 5.

## Next — Giai đoạn 4, trạng thái thực thi 2026-07-20 20:24 GMT+7

- Lý do bàn giao: user phải rời server (mang máy về) trước khi xong toàn bộ, ràng buộc thời gian. Mọi job nền đều persistent (systemd + linger `tts01`), không cần agent đứng cạnh.
- **Bước 1 (cache 4 candidate @1280): DONE.** Cả 4 file `figlib_detector_cache_{yolo26x_pyro_sdis,rfdetr_large_pyro_sdis,yolo26x_dfire,rfdetr_large_dfire}_dev.jsonl` đều 28,360 dòng, verify JSON dòng cuối hợp lệ.
- **Bước 2 (compare-matrix): ĐANG CHẠY nền, unit systemd `giai-doan4-compare-matrix.scope`** (log `/tmp/giai_doan4_compare_matrix.log`, script `/tmp/run_giai_doan4_compare_matrix.sh` — scratch, không thuộc repo). Bootstrap 1000 mẫu, thuần Python/CPU, ước ~20-40 phút (không có progress log giữa chừng, chỉ ghi file ở bước cuối). **Việc đầu tiên của agent tiếp theo: kiểm tra `artifacts/smoke_fire_detection/figlib_dev_compare_matrix.json` đã tồn tại chưa** — nếu có rồi thì dùng luôn, ĐỪNG chạy lại (tốn thêm 20-40 phút vô ích). Nếu chưa có VÀ không còn process `pgrep -f "temporal_eval compare-matrix"` nào sống → job đã chết, chạy lại lệnh dưới đây (copy nguyên, đây chính là nội dung `/tmp/run_giai_doan4_compare_matrix.sh`):

```
.venv/bin/python -m tasks.smoke_fire_detection.temporal_eval compare-matrix \
  --candidate yolo26x_pyro_sdis=artifacts/smoke_fire_detection/figlib_detector_cache_yolo26x_pyro_sdis_dev.jsonl \
  --candidate rfdetr_large_pyro_sdis=artifacts/smoke_fire_detection/figlib_detector_cache_rfdetr_large_pyro_sdis_dev.jsonl \
  --candidate yolo26x_dfire=artifacts/smoke_fire_detection/figlib_detector_cache_yolo26x_dfire_dev.jsonl \
  --candidate rfdetr_large_dfire=artifacts/smoke_fire_detection/figlib_detector_cache_rfdetr_large_dfire_dev.jsonl \
  --clean yolo26x_pyro_sdis,rfdetr_large_pyro_sdis,yolo26x_dfire,rfdetr_large_dfire \
  --architecture yolo26x_pyro_sdis=yolo --architecture rfdetr_large_pyro_sdis=rfdetr \
  --architecture yolo26x_dfire=yolo --architecture rfdetr_large_dfire=rfdetr \
  --dataset yolo26x_pyro_sdis=pyro --dataset rfdetr_large_pyro_sdis=pyro \
  --dataset yolo26x_dfire=dfire --dataset rfdetr_large_dfire=dfire \
  --dataset-order pyro,dfire \
  --latency-ms yolo26x_pyro_sdis=82.26 --latency-ms rfdetr_large_pyro_sdis=141.18 \
  --latency-ms yolo26x_dfire=77.26 --latency-ms rfdetr_large_dfire=136.05 \
  --out artifacts/smoke_fire_detection/figlib_dev_compare_matrix.json
```
  Job chạy nền qua `systemd-run --user --scope --unit=giai-doan4-compare-matrix ...` — nếu cần chạy lại, dùng đúng pattern này (không chạy foreground trần, dễ mất khi mất kết nối; đặt unit name mới nếu `giai-doan4-compare-matrix.scope` còn tồn tại, xem bài học "đừng tin `systemctl status`" ở Giai đoạn 3).
- **Bước 3:** đọc field `winner` trong JSON kết quả (không phải `leader`) — đây là candidate_id thắng cuộc theo rule đã chốt. Vì đã truyền đủ `--latency-ms` cho cả 4 candidate, rule luôn resolve được 1 winner (không rơi vào case `top_set>1` không tie-break được).
- **Bước 4 (2 operating point khóa) — quy trình chính xác, field name thật trong code (`temporal_eval.py:evaluate_rule`):**
  1. Chạy: `.venv/bin/python -m tasks.smoke_fire_detection.temporal_eval temporal --cache artifacts/smoke_fire_detection/figlib_detector_cache_<winner_id>_dev.jsonl --score smoke --out artifacts/smoke_fire_detection/figlib_dev_temporal_<winner_id>.json` (giữ default `--thresholds`/`--nofm`/`--ema-alphas`/`--bootstrap-samples 1000`/`--seed 20260707` — khớp protocol đã dùng ở G1).
  2. Output có field `results`: list các row `{rule, threshold, n/m hoặc alpha, false_alarm_rate_per_hour, event_recall, event_precision, ttd_median_seconds, ttd_mean_seconds, ...}`. **Chú ý: `false_alarm_rate_per_hour` tính theo GIỜ, ngân sách plan tính theo NGÀY/TUẦN** — phải tự quy đổi: FA≤1/camera/ngày ⇔ `false_alarm_rate_per_hour ≤ 1/24 ≈ 0.041667`; FA≤1/camera/tuần ⇔ `false_alarm_rate_per_hour ≤ 1/168 ≈ 0.0059524`.
  3. Với mỗi budget: lọc row thỏa ngưỡng trên, chọn `event_recall` cao nhất; tie → `ttd_median_seconds` thấp nhất; tie tiếp → `false_alarm_rate_per_hour` thấp nhất. Đây chính là 2 operating point khóa (ghi lại `rule`+`threshold`+`n/m` hoặc `alpha` của row được chọn, cùng recall/TTD/FA/precision).
- **Bước 5 — Viết `artifacts/smoke_fire_detection/figlib_dev_winner_lock.json`.** 2 field BẮT BUỘC để gate G5 (`eval.py:allowed_test_candidates`, test `FinalSplitGateTests`) đọc được: `winner_candidate_id` (string), `allowlist_candidate_ids` (list, luôn gồm `["yolo26n_dfire_control", "pyronear_yolov8s_reference"]`). Field khuyến nghị thêm (không bắt buộc cho code, nhưng cần cho report/audit) — template:

```json
{
  "winner_candidate_id": "<candidate_id từ bước 3>",
  "allowlist_candidate_ids": ["yolo26n_dfire_control", "pyronear_yolov8s_reference"],
  "decision_rule": "<winner_rule string từ compare-matrix JSON>",
  "auroc_per_candidate": "<copy per_candidate từ compare-matrix JSON>",
  "resolution": 1280,
  "conf": 0.05,
  "postprocess": {"yolo": "nms iou=0.6", "rfdetr": "native threshold=0.05"},
  "operating_points": {
    "fa_le_1_per_camera_per_day": "<row đã chọn ở bước 4, budget ngày>",
    "fa_le_1_per_camera_per_week": "<row đã chọn ở bước 4, budget tuần>"
  },
  "compare_matrix_source": "artifacts/smoke_fire_detection/figlib_dev_compare_matrix.json",
  "locked_at_gmt7": "<timestamp lúc ghi file>"
}
```
- **Bước 6 (câu hỏi resolution, mở khi 2 cache `@800` xong — xem Giai đoạn 3 cho trạng thái/ETA):** chạy `temporal_eval.py g0 --cache artifacts/smoke_fire_detection/figlib_detector_cache_{yolo26x_dfire,rfdetr_large_dfire}_dev_res800.jsonl` (default ignore-band 180s, bootstrap 1000, seed 20260707 — khớp protocol) để lấy AUROC@800 mỗi model; so với AUROC@1280 (lấy từ `per_candidate.<id>.auroc` trong compare-matrix JSON bước 2, CÙNG ignore-band/seed nên so được thẳng). Nếu @800 vượt rõ @1280 → viết thẳng là **oversight quy trình** trong report (winner đã khóa bằng @1280 TRƯỚC KHI có số @800 — quyết định user 2026-07-20 19:20 GMT+7), KHÔNG đổi ngược winner đã chọn, KHÔNG diễn giải giảm nhẹ.
- **Bước 7 — Report tĩnh `tasks/smoke_fire_detection/docs/model_benchmark_2x2.md`** (cấu trúc đầy đủ đã ghi trong `post_train_benchmark_plan.md` mục "Báo cáo, artifact, cleanup"): dùng số source-domain ở mục "Weight giữ" trên, dùng SOTA facts ở mục trên, dùng winner-lock bước 5, resolution-oversight bước 6 nếu có. Sau report: update `research_plan.md` (13d/13e), `external_assets.md` (thêm 4 cache dev sạch + winner-lock + compare-matrix vào danh sách giữ), `CHANGELOG.md`.
- Ràng buộc vẫn giữ nguyên: chỉ làm phần impactful nhất (winner-lock + report cốt lõi + câu hỏi resolution); spatial-persistence overlay restore/hủy, human-visibility stratification phụ, roadmap P1-P3, RQ5, SmokeyNet reproduction — CHƯA LÀM, ghi rõ "để khi có thêm thời gian" trong report, không tự ý làm thêm.
- Dừng lại sau winner-lock + report nháp — KHÔNG tự chạy Giai đoạn 5 (`--split test`, one-shot). Báo cáo đầy đủ, chờ user duyệt tường minh.
