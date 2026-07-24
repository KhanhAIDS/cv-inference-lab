- Vai trò: bộ nhớ ngắn hạn agent; dữ kiện sống; không command log.
- Repo: lab CV inference; core portable Windows/Linux; Modal wrapper tách core; không hard-code path/OS/GPU/user.
- Platform: Windows dùng `py`; Linux server dùng `.venv/bin/python`/pip.
- Tracking: append `CHANGELOG.md`; GMT+7; đủ command; file trực tiếp/gián tiếp.
- Không giữ hash/checksum/fingerprint như artifact/layer thường trực. Nếu user yêu cầu hoặc cho phép audit tạm: dùng được, phải xóa output tạm ngay sau kiểm tra và ghi nhận việc xóa.
- Root `docs/`: không đọc/sửa/xóa nếu user không yêu cầu rõ.
- Scope chính: `tasks/smoke_fire_detection/`; dataset `datasets/smoke_fire_detection/`; artifact `artifacts/smoke_fire_detection/`.
- Task phụ: `tasks/results_dashboard/` (xem mục riêng cuối file) — front-end demo, độc lập hoàn toàn, chỉ đọc artifact của task chính, không đụng ngược lại.

## Dataset/checkpoint

- D-Fire: train 15,500; valid 1,721; test 4,306; class smoke=0, fire=1. Label: `yolo26n_dfire_control` = gốc; `yolo26x_dfire`/`rfdetr_large_dfire` = đã relabel (khác control, không so trực tiếp).
- Pyro-SDIS YOLO: train 29,537; val 4,099; class smoke=0.
- FIgLib: 40,362 frame hợp lệ; 511 folder; 143 camera; split camera-disjoint seed 20260707; dev 359 seq/108 cam, final(test) 152 seq/35 cam, overlap 0.
- Near-field ZIP user tải trong `to_be_resolved/`, mới audit cấu trúc archive, chưa giải nén:
  - DFS-Fire v3: **8,735 ảnh thật trong ZIP**; train 6,117/valid 1,734/test 884; YOLO bbox; class `fire,smoke`; Public Domain; class order ngược D-Fire: DFS `fire=0,smoke=1` vs D-Fire `smoke=0,fire=1` → bắt buộc remap DFS trước merge. Số 8,939 trước đây là sai. Candidate train chính nếu audit bbox/ảnh âm PASS.
  - DeepQuestAI: **3,000 ảnh thật**; Train 2,700 = 900 Fire/900 Smoke/900 Neutral; Test 300 = 100 mỗi lớp. Classification folder-only, không bbox. Chỉ Neutral dùng trực tiếp làm detector empty-label negative; Fire/Smoke cần annotate/pseudo-label mới dùng làm positive.
  - FireAndSmoke v1: 100 ảnh; train 70/valid 20/test 10; YOLO 1 class `firerotation`; CC BY 4.0. Class semantic mơ hồ, không smoke, không hard-negative, gain nhỏ → loại khỏi lần train đầu; chỉ xem lại nếu audit ảnh chứng minh nhãn remap được.
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
- Snapshot backlog 2026-07-21 trong report mục 11 đã bị supersede một phần: L0, RQ5 control và SmokeyNet subsample nay đã làm; human-visibility stratification vẫn chưa làm.

## Quyết định user khác

- RQ5 near-field: đã mở sau khi RQ1 khóa winner; FIRESENSE control + NE4 đã xong; hiện ở NE1 audit dataset.
- SmokeyNet reference: đã chạy subsample; kết quả inconclusive do CPU-only/CI rộng; không chạy full.
- Kiến trúc triển khai đa phong bì (routing tĩnh theo camera + specialist + verifier chung) — chốt `research_plan.md` mục 2.3.
- Modal volume còn run-dir rác cũ (`yolo26x_pyro_sdis_budget9`, `_smoke_locked`, `_archive_benchmark`, `_warmstart`, `_resume_gate`, `resume_gate`) — ngoài scope, cần user xác nhận riêng nếu muốn dọn.

## P1 roadmap — TRẠNG THÁI 2026-07-24

- ✅ (a) L0 error-slice trên winner — DONE. Subcommand mới `temporal_eval.py error-slice` (giữ trong repo). Kết quả: bucket mean-confidence tăng đơn điệu theo bbox area (0.39→0.69 dev, 6 bucket) — xác nhận tiny-object là nguyên nhân miss chính cho winner cụ thể (không chỉ detector cũ). 26% positive frame dev không có detection nào. Per-camera AUROC median 0.876 (107/143 cam), ~10 camera chronic thấp (0.5-0.6) — nghi line-of-sight/vị trí, không phải nhiễu per-fire. Artifact: `l0_error_slice_rfdetr_large_dfire_{dev,final}.json`, `diagnose_rfdetr_large_dfire_dev.json` (chạy lần đầu cho winner RF-DETR, trước đây chỉ có cho detector cũ).
- ✅ (b) PYRONEAR-2025 data-lever — DONE, KẾT QUẢ ÂM 2026-07-24. Giải nén 145,657 file/9.2GB (`datasets/smoke_fire_detection/PYRONEAR-2025/`). Lọc leakage exact-match tên video vs `figlib_split_manifest.json` (511 sequence_id, cả dev+final): **315/592 video (53%) trùng thẳng FIgLib** — loại hết, còn 277 video sạch. Class_id (0/1) KHÔNG phải smoke/fire thật — verify bằng ảnh + thống kê nguồn (mỗi camera-source thuần 1 class_id) → artifact pipeline, remap cả 2 về `smoke`. Output: `datasets/smoke_fire_detection/PYRONEAR-2025-clean/{train,val,test}/{images,labels}` (train 32,784 ảnh/32,159 box, val 8,260/7,316, test 4,267/3,628) + `data.yaml` + `filter_report.json`.
  - Kaggle thủ công: continue-train `RFDETRLarge` 8 epoch, resolution train 800, PYRONEAR-clean-only, seed 20260707. User tải về 3 inference weight + metrics/config; chọn trước `checkpoint_best_total.pth` theo source validation thay vì target-guided sweep 8 checkpoint. Weight này cùng global_step 10245 với best EMA epoch 4; best EMA mAP50-95=0.45057.
  - FIgLib dev @1280, conf 0.05: candidate AUROC smoke **0.80938** vs winner gốc **0.83173**, Δ=-0.02235; paired CI95 event `[-0.03766,-0.00643]`, camera `[-0.03799,-0.00586]` — cả 2 âm hoàn toàn, fine-tune **tệ hơn có ý nghĩa thống kê**. G0 riêng vẫn PASS ≥0.80 nhưng data-only lever FAIL mục tiêu cải thiện; giữ winner gốc. Không mở final/test, không chạy checkpoint khác.
  - Full dev GB10 shared-load: 28,360 frame + 1 corrupt JPEG, batch 8, 1h25m10s, CPU avg 155%, peak RSS 2.65GiB, 0 swap; sampled process GPU 29-42% SM, total GPU 93-95%, 69-77°C. Cache + reports: `figlib_detector_cache_rfdetr_large_dfire_pyronear2025_ft_dev.jsonl`, `gate_g0_auroc_rfdetr_large_dfire_pyronear2025_ft_dev.json`, `compare_rfdetr_large_dfire_vs_pyronear2025_ft_dev.json`.
- ✅ (c) SmokeyNet reference — DONE (subsample). Checkpoint ONNX từ `sagecontinuum/sage-smoke-detection` (`s3-west.nrp-nautilus.io/smokeynet/model.onnx`). Adapter mới `tasks/smoke_fire_detection/smokeynet_cache.py` (onnxruntime, tách khỏi eval.py — input 2-frame tile-classifier khác box-detector). CPU-only trên GB10 (~22s/frame, không có onnxruntime-gpu phù hợp aarch64+CUDA13) → full dev bất khả (~7 ngày) → chạy subsample 30/359 sequence (seed 20260707). Kết quả: AUROC=0.8092 (CI event `[0.730,0.895]` rất rộng do n nhỏ). So paired đúng cùng subset với winner (winner tự đo lại trên subset này = 0.7798, khác hẳn AUROC toàn dev 0.8317 — biến thiên mẫu nhỏ thật): Δ=+0.0294, CI chứa 0 → **tie_or_not_proven, không kết luận được**. Giới hạn compute, không phải giới hạn phương pháp. Artifact: `gate_g0_auroc_smokeynet_reference_subsample30.json`, `compare_winner_vs_smokeynet_dev_subsample30.json`.
- ✅ SOTA anchor `pyronear/yolo11s_rapid-raccoon_v8.1.0` (HF, ungated, imgsz=1024, single_cls) — DONE, full dev (28,360/28,361 frame). AUROC=0.8695 (winner=0.8317), Δ=+0.0378, CI event `[0.021,0.053]`/camera `[0.020,0.056]` — **cả 2 CI không chứa 0, candidate_better có ý nghĩa thống kê thật**. Vẫn leakage-caveat, KHÔNG thay winner. Artifact: `gate_g0_auroc_pyronear_yolo11s_rapid_raccoon.json`, `compare_winner_vs_pyronear_yolo11s_dev.json`.
- **Cả 2 số SOTA anchor trên đã ghi vào report tĩnh** `model_benchmark_2x2.md` mục 9.1.
- ✅ Pooled dev+final AUROC cho winner — DONE 2026-07-23 (bổ sung theo yêu cầu user muốn giảm sampling noise): winner đã khóa xong trên dev trước khi final mở nên gộp dev+final không phát sinh selection-bias mới, chỉ tăng N. Kết quả AUROC(smoke)=0.8323, CI event [0.8156,0.8482], CI camera [0.8126,0.8519] — hẹp hơn rõ so với final-only. Ghi vào `model_benchmark_2x2.md` mục 12.1bis (bổ sung, không thay bảng one-shot gốc). Artifact `gate_g0_auroc_rfdetr_large_dfire_pooled_dev_final.json`.
- ✅ (2) mitigation catastrophic-forgetting Kaggle — CHỐT 2026-07-23: user quyết định GIỮ NGUYÊN plan gốc (PYRONEAR-2025-clean-only, KHÔNG gộp D-Fire/Pyro-SDIS) — giữ khả năng cô lập domain-match lever cho report, chấp nhận rủi ro forgetting làm caveat công khai (không mitigate). Không cần merge/re-zip/upload gì thêm — notebook hiện tại dùng được ngay.
- ⚠️ Câu hỏi treo cho user (chưa trả lời): có mở `--split test` (final) cho 2 candidate reference mới (Pyronear yolo11s, SmokeyNet) không — cần sửa `figlib_dev_winner_lock.json` allowlist, đụng kỷ luật "final split one-shot" — rủi ro thật là selection-bias (không phải leakage cổ điển), thấp hơn vì 2 candidate này không tranh winner (winner đã khóa trên dev trước khi đụng final) — xem `research_plan.md` mục 10.9-10.11.
- (d) E1a qua đường tải thay thế — vẫn chưa làm, server `ai2` không kết nối HPWREN.
- NE4 (bbox area/short-side D-Fire, không phải L0 nhưng cùng đợt) — DONE: median short-side smoke ~29-31% width, fire ~5.5% — xa vendor near-field spec (1.1-1.6%) 20-27 lần → **D-Fire không phải tiny-object domain, không copy ưu tiên P2 sang near-field.** Artifact: `artifacts/smoke_fire_detection/ne4_dfire_scale_histogram.json`.
- FIRESENSE control eval — DONE 2026-07-22. `rfdetr_large_dfire` zero-shot 49/49 video: pooled AUROC 0.977, smoke 0.991, fire 0.983. Fire-neg có false-alarm thật: 6/16 vượt conf 0.5, 3 video rank-invert với true positive thấp nhất. Threshold 0.70 trên FIRESENSE giữ recall 11/11, còn 3/16 false alarm; đây là development operating point, không phải final độc lập.
- Block dataset đã đổi: user tự tải 3 ZIP near-field; FASDD ScienceDB không cần login; Indoor Fire Smoke được xác nhận đúng domain nhưng chưa tải. FireAndSmoke CostiCatargiu/MS-FSDB vẫn blocked, không còn là blocker cho NE1.
- Quy tắc dataset/checkpoint mới (chốt 2026-07-21): CHỈ lấy nguồn ungated/instant-grant (HF public, GitHub public, Zenodo open, Google Drive public-link). TUYỆT ĐỐI không lấy nguồn cần form/email duyệt thủ công (đã loại ONFIRE, LFDN vì lý do này).

## Near-field — next

- Trước train: audit ZIP trên CPU, không GPU; kiểm split/class, bbox malformed/out-of-range, empty-label/ảnh âm, sample ảnh Neutral và `firerotation`. Không cần train để biết DeepQuest Neutral có đúng nuisance mục tiêu hay không.
- Ưu tiên thời gian user: nếu audit PASS, làm **1 candidate gộp** `D-Fire + DFS-Fire + DeepQuest Train/Neutral`; D-Fire = rehearsal giữ năng lực cũ, DFS-Fire = supervised positive mới, DeepQuest Neutral = hard/background negative. Chấp nhận không tách được causal contribution của từng nguồn.
- Không dùng DeepQuest Fire/Smoke trong lần đầu vì thiếu bbox. Không dùng FireAndSmoke v1 trong lần đầu vì semantic class mơ hồ + chỉ 100 ảnh.
- FIRESENSE đã dùng để phát hiện failure, chọn hướng data và tham khảo threshold → chỉ còn vai trò development/regression paired set. Không được gọi điểm cải thiện trên FIRESENSE là final độc lập.
- Held-out = dữ liệu không dùng cho train/tune/selection; không đồng nghĩa cả dataset Indoor phải bị cấm train. Sau audit: nếu Indoor có split video/scene-disjoint, dùng train split để train và giữ test split held-out; nếu split không sạch, dùng Indoor cho train và dành FASDD/nguồn độc lập khác làm held-out.

## Task phụ — `tasks/results_dashboard/` — server GPU + video demo

- Scope: demo benchmark; chỉ đọc task/artifact smoke-fire.
- Platform: Linux server GPU; Windows/Modal backlog, không làm hiện tại.
- Git: user yêu cầu ignore toàn bộ `tasks/results_dashboard/`; `.gitignore` giữ đúng 1 dòng này; source/dashboard không hiện `git status`, không commit.
- Backend: FastAPI; demo registry chỉ RF-DETR winner + Pyronear YOLO11s reference; lazy-load, gọi lần đầu mới chiếm VRAM; queue 1 worker; FIgLib 6 mẫu mặc định; FIRESENSE 49 OOD; upload ≤250 MiB. YOLO26n chỉ còn benchmark tĩnh/control.
- Camera laptop: browser quay clip → upload → render; không live. Chỉ HTTPS/`localhost`; HTTP LAN bị browser chặn camera. Upload vẫn chạy qua LAN.
- Frontend: 4 tab; glossary collapsed global (CI/CI95/AUROC/FA/TTD/mAP/VRAM...); benchmark ảnh 6 model = 2 cột×3 hàng desktop, 1 cột mobile; `Kết quả` có research status collapsed; không clutter default.
- Exporter: parse `*_final` đúng split; `--skip-frames` giữ frame comparison cũ; metadata có Pyronear YOLO11s reference.
- Dashboard refresh 2026-07-24: `ResearchStatus` up-to-date + panel `FollowUpResults`; pooled/Pyronear YOLO11s/PYR fine-tune/SmokeyNet/FIRESENSE/NE4/NE1. 26 unit pass; Vite build pass.
- Service `results-dashboard.service`: từng OOM-kill 2026-07-23 do áp lực toàn hệ thống (process peak 49.4MiB); restart 2026-07-24 PASS; bind `0.0.0.0:8731`; API đúng 2 model, cả 2 `resident=false`; RSS ~76MiB; dashboard PID không có GPU allocation.
- Truy cập LAN: `http://<LAN-IP>:8731/`; cần cùng subnet/firewall cho TCP 8731. SSH tunnel: `ssh -L 8731:127.0.0.1:8731 <server>`.
- Chưa verify: tương tác/layout bằng browser thật; camera permission thật trên laptop.
- Backlog: live stream; Windows/Modal; SmokeyNet GPU; serving TensorRT/FP16/INT8.
