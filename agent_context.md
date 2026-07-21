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

## Giai đoạn 3 — cache FIgLib dev — DONE 2026-07-21

- 8 cache dev đều 28,360/28,360 dòng: 4 candidate sạch @1280, `yolo26n_dfire_control`@1280, `pyronear_yolov8s_reference`@1024, D-Fire@800×2 (diagnostic, đã xóa raw sau khi chốt số — xem Giai đoạn 4).
- Job từng crash 1 lần giữa chừng (`rfdetr_large_dfire`@1280, ~18:55 GMT+7 2026-07-20, nguyên nhân không xác định chắc — không có quyền đọc kernel log) — resume an toàn, không mất dữ liệu. **Bài học giữ lại: đừng tin `systemctl status` một mình khi debug job nền, cross-check `pgrep`/mtime file thật** (lần đó unit báo `active` dù process con đã chết, do lỗi phụ cạn inotify instance lúc start unit).

## Giai đoạn 4 — phân tích 2×2 + winner — DONE 2026-07-21

- **Winner: `rfdetr_large_dfire`** (RF-DETR-L/D-Fire). AUROC(smoke) FIgLib dev = 0.8317 (CI event [0.8103,0.8498], camera [0.8097,0.8518]) — thắng cả 3 candidate sạch còn lại đúng rule (event lower CI>0 và camera median delta>0), không cần tie-break latency/cost. Control (0.7380) và Pyronear reference leakage-caveat (0.8078) đều thấp hơn winner.
- Interaction kiến trúc×dataset có ý nghĩa (-0.068, CI không chứa 0): RF-DETR-L tốt hơn trên D-Fire, YOLO26x tốt hơn trên Pyro-SDIS — không có kiến trúc thắng tuyệt đối cả 2 dataset.
- Câu hỏi bắt buộc resolution@1280 vs @800: **đã trả lời, không có oversight** — cả 2 model D-Fire, `@1280` bằng hoặc nhỉnh hơn `@800` (yolo26x_dfire 0.760 vs 0.731; winner 0.832 vs 0.826, CI chồng lấn). Quyết định khóa resolution 1280 trước đó là đúng.
- Operating point khóa (winner): FA≤1/ngày → EMA α=0.5 ngưỡng 0.6, recall 0.690, TTD median 598s. FA≤1/tuần → EMA α=0.1 ngưỡng 0.8, recall 0.067, TTD median 2010s.
- Artifact: `figlib_dev_compare_matrix.json`, `figlib_dev_winner_lock.json`, `figlib_dev_temporal_rfdetr_large_dfire.json`, `gate_g0_auroc_{yolo26n_dfire_control,pyronear_yolov8s_reference,yolo26x_dfire_res800,rfdetr_large_dfire_res800}.json`, report tĩnh `tasks/smoke_fire_detection/docs/model_benchmark_2x2.md` (đầy đủ protocol/số liệu/SOTA/caveat, xem file đó thay vì hỏi lại).
- Đã dọn theo retention plan: xóa 2 cache D-Fire@800 raw + pyronear@1280 raw (số đã chốt vào `gate_g0_auroc_*` trước khi xóa).
- CHƯA LÀM (cố ý để dành, không phải bỏ sót — xem mục 11 report): human-visibility stratification, spatial-persistence overlay restore/hủy, lát cắt lỗi L0, roadmap P1-P3, RQ5, SmokeyNet reproduction.

## Quyết định user khác (chưa thực thi, không chặn Giai đoạn 5)

- RQ5 near-field: duyệt mở, sequenced sau khi khóa winner RQ1 — `research_plan.md` mục 10.
- SmokeyNet reference: ưu tiên tìm checkpoint public, 2 lead: `gitlab.nrp-nautilus.io/anshumand/pytorch-lightning-smoke-detection`, `github.com/iperezx/sage-smoke-detection` — chưa tự tải/verify.
- Kiến trúc triển khai đa phong bì (routing tĩnh theo camera + specialist + bộ xác nhận chung) — chốt `research_plan.md` mục 2.3.
- Roadmap ưu tiên P0-P3 — `research_plan.md` mục 8.1.

## Next — Giai đoạn 5 (final camera-held-out, one-shot) — CHƯA CHẠY, chờ user duyệt tường minh

- Naming: "final" trong plan = `eval.py detector-cache --split test`.
- Chỉ chạy 3 candidate (đã trong `figlib_dev_winner_lock.json`): `rfdetr_large_dfire` (winner), `yolo26n_dfire_control`, `pyronear_yolov8s_reference`. Gate an toàn đã có sẵn (`check_final_split_gate`) sẽ tự reject candidate khác.
- KHÔNG tự chạy khi chưa có xác nhận rõ ràng của user trong lượt hiện tại — đây là thao tác không lặp lại được (final split chỉ mở đúng 1 lần).
- Sau khi user duyệt: chạy cache test-split cho 3 candidate trên → `temporal_eval.py` phân tích final (AUROC+CI event/camera, pairwise winner vs baseline, Pyronear leakage-caveat, replay đúng 2 operating point đã khóa — KHÔNG quét lại threshold) → cập nhật report → retention cleanup cuối (giữ winner/baseline/Pyronear final cache + report, xóa cache thừa không nằm allowlist).
