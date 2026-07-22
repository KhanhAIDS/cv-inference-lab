- Vai trò: bộ nhớ ngắn hạn agent; dữ kiện sống; không command log.
- Repo: lab CV inference; core portable Windows/Linux; Modal wrapper tách core; không hard-code path/OS/GPU/user.
- Platform: Windows dùng `py`; Linux server dùng `.venv/bin/python`/pip.
- Tracking: append `CHANGELOG.md`; GMT+7; đủ command; file trực tiếp/gián tiếp.
- Không hash/checksum/fingerprint dataset, split, checkpoint, artifact, file list nếu user không yêu cầu rõ.
- Root `docs/`: không đọc/sửa/xóa nếu user không yêu cầu rõ.
- Scope chính: `tasks/smoke_fire_detection/`; dataset `datasets/smoke_fire_detection/`; artifact `artifacts/smoke_fire_detection/`.
- Task phụ: `tasks/results_dashboard/` (xem mục riêng cuối file) — front-end demo, độc lập hoàn toàn, chỉ đọc artifact của task chính, không đụng ngược lại.

## Dataset/checkpoint

- D-Fire: train 15,500; valid 1,721; test 4,306; class smoke=0, fire=1. Label: `yolo26n_dfire_control` = gốc; `yolo26x_dfire`/`rfdetr_large_dfire` = đã relabel (khác control, không so trực tiếp).
- Pyro-SDIS YOLO: train 29,537; val 4,099; class smoke=0.
- FIgLib: 40,362 frame hợp lệ; 511 folder; 143 camera; split camera-disjoint seed 20260707; dev 359 seq/108 cam, final(test) 152 seq/35 cam, overlap 0.
- 6 candidate: `yolo26x_pyro_sdis`, `rfdetr_large_pyro_sdis`, `yolo26x_dfire`, `rfdetr_large_dfire` (4 sạch, tranh winner); `yolo26n_dfire_control`, `pyronear_yolov8s_reference` (control/reference, không tranh winner).
- Weight/mAP mỗi checkpoint: xem `external_assets.md`.
- Code giữ: `dataset.py`, `eval.py`, `temporal_eval.py`, `modal_app.py`, `test_eval.py`, `test_temporal_eval.py`. `temporal_eval.py` chỉ cần numpy+rich (không torch/ultralytics/rfdetr) — chạy được cả trên Windows local không GPU.
- Dọn 2026-07-21 (tra git history khi cần): xóa 3 notebook train, `report/demo_inference.py`, docs `post_train_benchmark_plan.md` + `research_toolbox.md`; docs/ còn đúng 3 file (`research_plan`, `model_benchmark_2x2`, `external_assets`).
- Artifact 2026-07-21: chỉ giữ cache winner (`rfdetr_large_dfire` dev+test); cache 5 candidate khác + 2 test control/pyronear + `figlib_dev_temporal_overlay_sweep.json` + 2 gate res800 ĐÃ XÓA (tái tạo = chạy lại GPU). Correction 2026-07-22: root `.gitignore` KHÔNG có rule `figlib_detector_cache_*`/`figlib_index.jsonl` (đã verify trực tiếp) — mọi cache nhỏ hiện có (winner, pyronear, smokeynet subsample) đều đang track bình thường trong git; claim cũ ở dòng này sai, đã sửa.

## Benchmark 2×2 — DONE TOÀN BỘ 2026-07-21 (Giai đoạn 0-5)

- **Winner: `rfdetr_large_dfire`.** Dev AUROC(smoke)=0.8317, final(test) AUROC=0.8347 — cả 2 PASS gate ≥0.80. Thắng control rõ trên cả dev+final (CI không chứa 0); vs Pyronear reference (leakage-caveat) khác biệt không có ý nghĩa thống kê trên cả 2 split.
- Resolution winner 1280 cho 4 candidate sạch (kể cả 2 model D-Fire train@800) — đã verify @800 không vượt @1280, không phải oversight.
- Operating point khóa trên dev, replay 1 lần trên final: **cả 2 tier (FA≤1/ngày, FA≤1/tuần) đều vượt budget khi replay** (~1.5x và ~3.4x) — nhiều khả năng do cỡ mẫu final nhỏ hơn dev (97.6h âm vs 230.6h). Đã ghi rõ trong report, không tự tune lại threshold trên test.
- Overlay spatial-persistence (`temporal_eval.py spatial-persistence`, restore từ git history `d82bfcf` 2026-07-21): pooled AUROC không đổi; recall cải thiện thật ở tier chặt (0.25-0.37 vs raw 0.07-0.10) nhưng cũng vượt budget khi replay trên final — chỉ dùng diagnostic, không thay raw score làm operating point chính thức.
- Report tĩnh đầy đủ (mọi số, protocol, caveat, SOTA comparison, final split, overlay): `tasks/smoke_fire_detection/docs/model_benchmark_2x2.md`. Đọc file này trước khi hỏi lại số liệu.
- CHỦ ĐỘNG CHƯA làm (ghi ở report mục 11, không phải bỏ sót): human-visibility stratification, lát cắt lỗi L0 trên winner, roadmap P1-P3, RQ5, SmokeyNet reproduction.

## Quyết định user khác (chưa thực thi)

- RQ5 near-field: duyệt mở, sequenced sau — `research_plan.md` mục 10.
- SmokeyNet reference: 2 lead (`gitlab.nrp-nautilus.io/anshumand/pytorch-lightning-smoke-detection`, `github.com/iperezx/sage-smoke-detection`), chưa tự tải/verify.
- Kiến trúc triển khai đa phong bì (routing tĩnh theo camera + specialist + verifier chung) — chốt `research_plan.md` mục 2.3.
- Modal volume còn run-dir rác cũ (`yolo26x_pyro_sdis_budget9`, `_smoke_locked`, `_archive_benchmark`, `_warmstart`, `_resume_gate`, `resume_gate`) — ngoài scope, cần user xác nhận riêng nếu muốn dọn.

## P1 roadmap — TRẠNG THÁI 2026-07-21 16:16 (user duyệt auto-mode, không hỏi lại việc reversible)

- ✅ (a) L0 error-slice trên winner — DONE. Subcommand mới `temporal_eval.py error-slice` (giữ trong repo). Kết quả: bucket mean-confidence tăng đơn điệu theo bbox area (0.39→0.69 dev, 6 bucket) — xác nhận tiny-object là nguyên nhân miss chính cho winner cụ thể (không chỉ detector cũ). 26% positive frame dev không có detection nào. Per-camera AUROC median 0.876 (107/143 cam), ~10 camera chronic thấp (0.5-0.6) — nghi line-of-sight/vị trí, không phải nhiễu per-fire. Artifact: `l0_error_slice_rfdetr_large_dfire_{dev,final}.json`, `diagnose_rfdetr_large_dfire_dev.json` (chạy lần đầu cho winner RF-DETR, trước đây chỉ có cho detector cũ).
- ✅ (b) PYRONEAR-2025 data-lever — DATA PREP DONE, fine-tune CHỜ USER CHẠY KAGGLE. Giải nén 145,657 file/9.2GB (`datasets/smoke_fire_detection/PYRONEAR-2025/`). Lọc leakage exact-match tên video vs `figlib_split_manifest.json` (511 sequence_id, cả dev+final): **315/592 video (53%) trùng thẳng FIgLib** — loại hết, còn 277 video sạch. Class_id (0/1) KHÔNG phải smoke/fire thật — verify bằng ảnh + thống kê nguồn (mỗi camera-source thuần 1 class_id) → artifact pipeline, remap cả 2 về `smoke`. Output: `datasets/smoke_fire_detection/PYRONEAR-2025-clean/{train,val,test}/{images,labels}` (train 32,784 ảnh/32,159 box, val 8,260/7,316, test 4,267/3,628) + `data.yaml` + `filter_report.json`.
  - User chọn venue **Kaggle thủ công** (không Modal/không GB10 local). Đã viết `tasks/smoke_fire_detection/kaggle_train_pyronear2025_rfdetr_finetune.ipynb`: continue-train `RFDETRLarge(resolution=800, pretrain_weights=checkpoint_best_total.pth)` — **800px khớp training gốc winner, KHÔNG phải 1280 (đó là imgsz lúc eval FIgLib, biến khác)**. Epoch=8 (judgment call chưa validate). Chỉ train trên PYRONEAR-2025-clean (không gộp D-Fire) — data-only lever, control=winner hiện tại.
  - Việc user tự làm (agent không có Kaggle credential/CLI): upload 2 Kaggle Dataset — data (`pyronear2025-clean-kaggle-upload.zip` tại scratchpad, 4.6GB) + checkpoint (`artifacts/smoke_fire_detection/runs/rfdetr_large_dfire/checkpoint_best_total.pth`, 130MB) — rồi tạo notebook Kaggle UI paste nội dung `.ipynb`, Accelerator GPU T4x2, chạy tay.
  - Sau khi có checkpoint mới (việc agent, chưa làm): copy vào `artifacts/smoke_fire_detection/runs/rfdetr_large_dfire_pyronear2025_ft/`, chạy `eval.py detector-cache` + `temporal_eval.py g0`/`compare-candidates` so winner trên FIgLib **dev** (không đụng final — one-shot).
- ✅ (c) SmokeyNet reference — DONE (subsample). Checkpoint ONNX từ `sagecontinuum/sage-smoke-detection` (`s3-west.nrp-nautilus.io/smokeynet/model.onnx`). Adapter mới `tasks/smoke_fire_detection/smokeynet_cache.py` (onnxruntime, tách khỏi eval.py — input 2-frame tile-classifier khác box-detector). CPU-only trên GB10 (~22s/frame, không có onnxruntime-gpu phù hợp aarch64+CUDA13) → full dev bất khả (~7 ngày) → chạy subsample 30/359 sequence (seed 20260707). Kết quả: AUROC=0.8092 (CI event `[0.730,0.895]` rất rộng do n nhỏ). So paired đúng cùng subset với winner (winner tự đo lại trên subset này = 0.7798, khác hẳn AUROC toàn dev 0.8317 — biến thiên mẫu nhỏ thật): Δ=+0.0294, CI chứa 0 → **tie_or_not_proven, không kết luận được**. Giới hạn compute, không phải giới hạn phương pháp. Artifact: `gate_g0_auroc_smokeynet_reference_subsample30.json`, `compare_winner_vs_smokeynet_dev_subsample30.json`.
- ✅ SOTA anchor `pyronear/yolo11s_rapid-raccoon_v8.1.0` (HF, ungated, imgsz=1024, single_cls) — DONE, full dev (28,360/28,361 frame). AUROC=0.8695 (winner=0.8317), Δ=+0.0378, CI event `[0.021,0.053]`/camera `[0.020,0.056]` — **cả 2 CI không chứa 0, candidate_better có ý nghĩa thống kê thật**. Vẫn leakage-caveat, KHÔNG thay winner. Artifact: `gate_g0_auroc_pyronear_yolo11s_rapid_raccoon.json`, `compare_winner_vs_pyronear_yolo11s_dev.json`.
- **Cả 2 số SOTA anchor trên đã ghi vào report tĩnh** `model_benchmark_2x2.md` mục 9.1.
- ⚠️ Câu hỏi treo cho user (chưa trả lời): (1) có mở `--split test` (final) cho 2 candidate reference mới (Pyronear yolo11s, SmokeyNet) không — cần sửa `figlib_dev_winner_lock.json` allowlist, đụng kỷ luật "final split one-shot", KHÔNG tự quyết; (2) mitigation catastrophic-forgetting cho Kaggle fine-tune notebook (giữ nguyên chấp nhận rủi ro / trộn D-Fire / giảm LR-epoch) — user chưa chọn; (3) user chưa báo đã chạy Kaggle fine-tune.
- (d) E1a qua đường tải thay thế — vẫn chưa làm, server `ai2` không kết nối HPWREN.
- NE4 (bbox area/short-side D-Fire, không phải L0 nhưng cùng đợt) — DONE: median short-side smoke ~29-31% width, fire ~5.5% — xa vendor near-field spec (1.1-1.6%) 20-27 lần → **D-Fire không phải tiny-object domain, không copy ưu tiên P2 sang near-field.** Artifact: `artifacts/smoke_fire_detection/ne4_dfire_scale_histogram.json`.
- FIRESENSE control eval — DONE 2026-07-22. Script mới `firesense_cache.py` (trích frame OpenCV + infer RF-DETR, tái dùng adapter `eval.py`) + `firesense_gate.py` (AUROC, tái dùng `temporal_eval.py`). Checkpoint `rfdetr_large_dfire` zero-shot 49/49 video: pooled AUROC 0.977, smoke 0.991, fire 0.983 (tất cả PASS >>0.80). NG2: smoke-neg vô hại, **fire-neg có false-alarm thật** (6/16 vượt conf 0.5, 3 video rank-invert với true positive thấp nhất) — số chốt đầy đủ ở `research_plan.md` mục 10.7. Quyết định mở bước 2 nhắm lớp fire — nhưng **CẢ 3 nguồn ưu tiên đã chốt đều bị chặn ngoài kiểm soát**: FireAndSmoke (Roboflow, Cloudflare JS-challenge thật), FASDD (không có Kaggle CLI/credential), MS-FSDB (link Google Drive 404 chết, tác giả đã xóa). Indoor Fire Smoke (Zenodo) tải được nhưng user cần tự quyết domain indoor có liên quan không. Đã dừng lại, chờ user — KHÔNG tự thử vượt Cloudflare/Kaggle-gate, KHÔNG tự chọn Indoor Fire Smoke thay user.
- Quy tắc dataset/checkpoint mới (chốt 2026-07-21): CHỈ lấy nguồn ungated/instant-grant (HF public, GitHub public, Zenodo open, Google Drive public-link). TUYỆT ĐỐI không lấy nguồn cần form/email duyệt thủ công (đã loại ONFIRE, LFDN vì lý do này).

## Task phụ — `tasks/results_dashboard/` — server GPU + video demo

- Scope: demo benchmark; chỉ đọc task/artifact smoke-fire.
- Platform: Linux server GPU; Windows/Modal backlog, không làm hiện tại.
- Git: user yêu cầu ignore toàn bộ `tasks/results_dashboard/`; `.gitignore` giữ đúng 1 dòng này; source/dashboard không hiện `git status`, không commit.
- Backend: FastAPI; demo registry chỉ RF-DETR winner + Pyronear YOLO11s reference; lazy-load, gọi lần đầu mới chiếm VRAM; queue 1 worker; FIgLib 6 mẫu mặc định; FIRESENSE 49 OOD; upload ≤250 MiB. YOLO26n chỉ còn benchmark tĩnh/control.
- Camera laptop: browser quay clip → upload → render; không live. Chỉ HTTPS/`localhost`; HTTP LAN bị browser chặn camera. Upload vẫn chạy qua LAN.
- Frontend: 4 tab; glossary collapsed global (CI/CI95/AUROC/FA/TTD/mAP/VRAM...); benchmark ảnh 6 model = 2 cột×3 hàng desktop, 1 cột mobile; `Kết quả` có research status collapsed; không clutter default.
- Exporter: parse `*_final` đúng split; `--skip-frames` giữ frame comparison cũ; metadata có Pyronear YOLO11s reference.
- Test 2026-07-21: 26 unit pass; Vite build pass; API sau restart đúng 2 model, cả 2 `resident=false`; dashboard PID không có GPU allocation; CSS production có grid 2 cột + glossary.
- Service sống: transient user unit `results-dashboard.service`; bind `0.0.0.0:8731`; LAN IP hiện tại `192.168.1.186`; loopback+LAN API curl 200.
- Truy cập LAN: `http://<LAN-IP>:8731/`; cần cùng subnet/firewall cho TCP 8731. SSH tunnel: `ssh -L 8731:127.0.0.1:8731 <server>`.
- Chưa verify: tương tác/layout bằng browser thật; camera permission thật trên laptop.
- Backlog: live stream; Windows/Modal; SmokeyNet GPU; serving TensorRT/FP16/INT8.
