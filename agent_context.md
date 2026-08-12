- Vai trò: bộ nhớ ngắn hạn agent; dữ kiện sống; không command log.
- Repo: lab CV inference; core portable Windows/Linux; Modal wrapper tách core; không hard-code path/OS/GPU/user.
- Platform: Windows dùng `py`; Linux server dùng `.venv/bin/python`/pip.
- Tracking: append `CHANGELOG.md`; GMT+7; đủ command; file trực tiếp/gián tiếp.
- Không giữ hash/checksum/fingerprint như artifact/layer thường trực. Nếu user yêu cầu hoặc cho phép audit tạm: dùng được, phải xóa output tạm ngay sau kiểm tra và ghi nhận việc xóa.
- Root `docs/`: không đọc/sửa/xóa nếu user không yêu cầu rõ.
- Scope chính: `tasks/smoke_fire_detection/`; dataset `datasets/smoke_fire_detection/`; artifact `artifacts/smoke_fire_detection/`.
- Task phụ: `tasks/results_dashboard/` (xem mục riêng cuối file) — front-end demo, độc lập hoàn toàn, chỉ đọc artifact của task chính, không đụng ngược lại.

## Product reset 2026-07-31

- Primary: fixed-camera near-field; detect cả `fire` + `smoke`; quality max.
- Cost: FN > FP; selection = event recall từng lớp → worst-class recall → giảm FA.
- Bbox geometry phụ; class/missing-label/ignore đúng mới là bắt buộc.
- Canonical detection taxonomy: `smoke=0`, `fire=1`; overlap được; nuisance = negative/metadata; ambiguous = ignore.
- Far-field wildfire: closed benchmark/reference/regression; không còn roadmap chính.
- `rfdetr_large_dfire`: near-field baseline, không gọi product winner trước near-field selection.
- Split: source/video/scene/camera-disjoint; cấm random frame split; chống duplicate/near-duplicate qua split.
- Image-level positive: classifier branch hoặc annotate/pseudo-label+review; image-level negative audit PASS → detector empty-label.
- Metrics chính: per-class event recall/miss, worst-class recall, TTD, FA/camera-hour, FA/alarm episode. mAP/AUROC = diagnostic.
- Threshold fire/smoke riêng; alarm OR; temporal không được chặn đường báo nhanh high-confidence.
- Current phase: complete near-field source inventory → audit/license/lineage → canonical export → train/eval near-field.

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
- Near-field inventory local: D-Fire; DFS v3; DeepQuestAI; FireAndSmoke base 10k; Home Fire 6.5k; Indoor 5k; Annotated 11,021; FASDD-CV archive. User chốt MS-FSDB hoàn tất bằng các nguồn thành phần local; FireAndSmoke 2024 chỉ dùng base 10k, không lấy bản 22k augment.
- Audit 2026-07-31→08-04: 7 nguồn local clean pairing và 0 malformed bbox; Indoor có 87/6,935 bbox vượt biên; DFS `fire=0,smoke=1`; 5 exact duplicate cross-source giữa base 10k và FireAndSmoke v1, loại v1. Annotated: user cho phép dùng theo CC BY trên trang phát hành, bất kể metadata archive `Private`. FASDD-CV V9 raw: 95,314 ảnh/label, 0 corrupt/missing/malformed, 39,199 empty label, class raw `fire=0` 73,297 box/`smoke=1` 53,080; 3 nonpositive, 161 outside, 170 exact-duplicate group nội bộ. Mirror 3,104 ảnh xác nhận byte-identical subset V9, ĐÃ XÓA archive+raw mirror. Chưa canonicalize; cần lineage/split/near-field role trước train.
- Quy tắc dataset/checkpoint mới (chốt 2026-07-21): CHỈ lấy nguồn ungated/instant-grant (HF public, GitHub public, Zenodo open, Google Drive public-link). TUYỆT ĐỐI không lấy nguồn cần form/email duyệt thủ công (đã loại ONFIRE, LFDN vì lý do này).

## Near-field — next

- Trước train: hoàn tất inventory mọi nguồn near-field khả dụng; audit license, lineage/video, split, class, missing-label, bbox malformed/out-of-range, empty-label/ảnh âm và nuisance trên CPU.
- Candidate gộp cũ `D-Fire + DFS-Fire + DeepQuest Train/Neutral` chỉ còn là baseline nhanh, không phải kế hoạch data cuối nếu nguồn near-field tốt hơn audit PASS.
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
- Presentation độc lập: `tasks/results_dashboard/project_history_presentation.html`; 21 slide; tập trung kết quả phát hiện khói/lửa; 3 JPEG bbox nhúng data URI; tải 1 HTML vẫn đủ ảnh; offline.
- Exporter: parse `*_final` đúng split; `--skip-frames` giữ frame comparison cũ; metadata có Pyronear YOLO11s reference.
- Dashboard refresh 2026-07-24: `ResearchStatus` up-to-date + panel `FollowUpResults`; pooled/Pyronear YOLO11s/PYR fine-tune/SmokeyNet/FIRESENSE/NE4/NE1. 26 unit pass; Vite build pass.
- Service `results-dashboard.service`: từng OOM-kill 2026-07-23 do áp lực toàn hệ thống (process peak 49.4MiB); restart 2026-07-24 PASS; bind `0.0.0.0:8731`; API đúng 2 model, cả 2 `resident=false`; RSS ~76MiB; dashboard PID không có GPU allocation.
- Truy cập LAN: `http://<LAN-IP>:8731/`; cần cùng subnet/firewall cho TCP 8731. SSH tunnel: `ssh -L 8731:127.0.0.1:8731 <server>`.
- Chưa verify: tương tác/layout bằng browser thật; camera permission thật trên laptop.
- Backlog: live stream; Windows/Modal; SmokeyNet GPU; serving TensorRT/FP16/INT8.
- Visual triage 2026-08-04: DFS, Home, Indoor, FASDD V9 xem 170 ảnh mẫu; raw mapping đều `0=fire`, `1=smoke`; không thấy lỗi nhãn hệ thống. Indoor 86, FASDD 153 ảnh bbox vượt biên trong traversal; chỉ clip bbox đã xác minh ở canonical export.
- Cleanup: codebase mặc định agent-owned; dọn script one-shot/artifact trung gian sau tác vụ, chỉ giữ output quyết định.
- GPU thuê ngoài: tối đa utilization an toàn để giảm chi phí thuê. GPU server công ty: trước job báo estimate thời gian/chi phí GPU nội bộ, kiểm shared load/VRAM, không ảnh hưởng job khác.
- Lineage triage 2026-08-04: Indoor có 375 nhóm filename-gốc bắc train/valid/test → vendor split leak. Manifest `indoor_fire_smoke_augmentation_group_split.json`: group-disjoint 3,500/750/750, raw không đổi; chỉ diagnostic, chưa chứng minh scene/video-disjoint. Home không có lineage signal → train phụ, không dùng vendor valid/test làm benchmark.
- Round-1 label audit: `near_field_review.py` tạo queue cân bằng 182 ảnh cho DFS/Home/Indoor/Annotated/FASDD; queue `near_field_round1_review_queue.json`, trạng thái đều pending, không phải benchmark/split. Exact duplicate audit: 1 ảnh trùng DFS-valid ↔ FASDD; canonical export phải loại một bản khỏi eval.
- Lineage update 2026-08-04: DFS 0 nhóm tên lặp/cross-split; Annotated 2,214 nhóm lặp, 879 cross-split → vendor split leak. Manifest `annotated_fire_smoke_augmentation_group_split.json`: group-disjoint 7,715/1,653/1,653; diagnostic, không chứng minh scene/video-disjoint.
- Review UX: `near_field_round1_review.html` = offline page ảnh+bbox, phím 1 đúng / 2 thiếu / 3 sai-thừa / 4 không chắc, tải JSON decision. Queue JSON chỉ machine list, không yêu cầu user sửa tay.
- Review HTML update: đỏ=fire, xanh=smoke; trạng thái auto-save browser localStorage, nhưng cần tải JSON kết quả để đưa lại repo/agent xử lý.
- Round-1 human review imported/validated: 182/182 valid; pass 155; missing 17; bbox wrong/extra 9; uncertain 1. Pattern missing co-label: Annotated 9 missing/32 (6 fire-only, 2 smoke-only, 1 empty) + 1 uncertain; Indoor 5 fire-only/30; DFS 2 fire-only/40; FASDD 1 fire-only/40; Home 0/40. Không auto-relabel/raw edit; bbox rộng/hẹp chỉ ghi nhận, chưa sửa.
- Artifact outcome: `artifacts/smoke_fire_detection/near_field_round1_review_outcome.json`; tool review có mode `--queue ... --decisions ... --output ...` để validate 1:1 và tổng hợp.
- Train blocker: chưa có đơn giá GPU nội bộ; trước bất kỳ job GPU phải báo cost estimate, không tự chạy. Triage hiện tại: Annotated + Indoor không được đưa nguyên trạng vào baseline nghiêm túc vì missing co-label sample cao.
- GPU plan user 2026-08-04: Modal chỉ dùng profile ngắn với 1 hoặc 2 T4 để đo compute/VRAM và chọn cấu hình training Kaggle 2×T4 / Colab 1×T4; không train thực sự trên Modal hiện tại. Chờ user login Modal rồi mới inspect shared/load và báo estimate profile; không cần đơn giá GPU nội bộ cho job Modal.
- Correction triage: lỗi missing xuất hiện trong strata `fire_only`/`smoke_only`/`empty`; kết luận đúng là không train nguyên trạng toàn bộ Annotated/Indoor. Chưa loại toàn bộ source: ảnh `both`/đã pass có thể vào curated seed sau export; source-wide inclusion chỉ sau model-assisted audit.
- Modal profile 2026-08-05: YOLO26x @800 trên 256 ảnh D-Fire. 1×T4 batch 8: 68.21s, log GPU ~13.0/14.9GiB. 2×T4 DDP batch tổng 16 (8/GPU): 80.64s, log ~13.1/14.9GiB mỗi GPU; job ngắn bị overhead DDP nên không nhanh hơn end-to-end. Chốt safe config: Colab 1×T4 batch 8; Kaggle 2×T4 batch tổng 16. Chỉ hardware profile, không phải train/eval chất lượng.
- Model-assisted audit: `rfdetr_large_dfire` weight local 130MiB, 2 class, chọn làm teacher. `near_field_model_audit.py`: inference → queue mâu thuẫn strong (missing / unsupported), không auto-relabel; `--html` tạo offline reviewer. Chấm bbox dataset: đỏ/xanh nét liền; teacher nét đứt+confidence. Review multi-issue: missing_or_extra, wrong_class, geometry; `--queue --decisions` validate. Checkpoint ngoài: Pyronear/SmokeyNet far-field 1-class hoặc leakage-caveat; không dùng teacher near-field 2-class.
- Kaggle audit: notebook tự chứa code `near_field_model_audit_kaggle.ipynb`; Input 7 source expanded (5 bbox + DeepQuest + FireAndSmoke v1) + D-Fire RF-DETR checkpoint; 2 shard/GPU, output bundle ZIP review offline.
- DeepQuest không bị bỏ audit: nhãn cấp ảnh, teacher đề xuất bbox; chỉ `adopt_teacher` do user bấm duyệt mới thành nhãn dataset dẫn xuất. FireAndSmoke v1: audit đủ 100 ảnh, chỉ giữ/remap nếu user xác nhận semantic. Raw không bao giờ bị sửa.
- Cleanup strict: `to_be_resolved/` temp-only, phải empty/delete sau import; xóa ZIP/copy/checkpoint/cache/notebook/script one-shot khi source đã giải nén hoặc result quyết định giữ ở nơi khác.
- Cleanup 2026-08-05: `to_be_resolved/` removed (~16.5GB); removed task `__pycache__`, completed Pyro fine-tune notebook, FireSense/SmokeyNet one-shot scripts. Candidate delete cần user confirm riêng: Pyro-SDIS/PYRONEAR data 13.8GB + historical checkpoint/cache.

## Label audit tool — chốt 2026-08-12 (viết lại theo phản biện user, 2 lượt)

- Nguồn: Kaggle audit v2 sinh 2 bundle root `review_error_0000.zip` (29,378 error) + `review_pseudo_0000.zip` (1,047 pseudo, DeepQuest). Cả 2 zip ĐÃ XÓA sau merge (queue nằm trong workspace, ảnh đọc trực tiếp `datasets/`). run_signature `fa788fc586e18ac0d553df7bbeb9a6db8ce9f94d`.
- Tool: `tasks/smoke_fire_detection/label_audit_tool.py` (stdlib only) + `label_audit_tool.html` (canvas editor). Subcommand: `import` / `serve` / `status` / `export` / `pack` / `merge`.
- Workspace: `artifacts/smoke_fire_detection/label_audit/tasks.jsonl.gz` (30,425 task) + `workspace.json` (đường dẫn workspace đổi từ `near_field_label_audit` → `label_audit` do tổ chức lại artifacts). `decisions.jsonl` = ground truth, append-only, key = task_id (đường dẫn ảnh tương đối sau `datasets/smoke_fire_detection/`, bền qua mọi model run).
- **Class chỉ 2: `smoke=0`, `fire=1`. KHÔNG có `ignore`.** User đã relabel D-Fire thật, xác nhận ambiguous case hiếm và lượng data đủ lớn để không cần layer này — bỏ hẳn khỏi tool + doc. Bản đầu từng có `ignore=2`/`eval_safe`/`.audited.ignore.txt`/nuisance-flag, đã revert toàn bộ theo phản biện.
- `label_source` mỗi decision: `dataset`/`teacher`/`human`. Rule: eval split chỉ dùng `dataset`+`human`; `teacher` (phím `W` = lấy bbox model + lưu 1 phím) train-only vì teacher = `rfdetr_large_dfire` = winner hiện tại → circular evaluation nếu dùng để eval.
- Status chỉ 3: `confirmed`(=GT)/`excluded`/`draft`. Bỏ `uncertain` (dead status).
- UI tối giản theo yêu cầu user — bỏ mọi màu mè (marching-ants/blink/spotlight/grayscale/palette): chỉ smoke/fire màu cố định (`#C6FF00`/`#FF2BD6`) + casing tương phản, ẩn/hiện bbox model (`T`), ngưỡng conf (`[`/`]`). Rules gộp vào 2 `<details>` dropdown ở panel phải, không phải sidebar cố định.
- **Hit-test bbox lồng nhau đúng yêu cầu**: click chọn bbox NHỎ NHẤT chứa điểm; kéo bbox đang chọn chỉ move đúng nó, bbox to trùm lên không đổi; `Shift`+kéo ép vẽ mới ngay trong vùng có bbox chọn sẵn. Verified bằng chromium headless dispatch pointer event thật, không chỉ đọc code.
- **Resume**: `localStorage` lưu task_id cuối + filter đã chọn; mở lại app tự nhảy đúng ảnh, không cần server-side session.
- **Label offline**: `pack --output <dir> --zip [--issue ...] [--source ...] [--limit N]` copy ảnh + tool + launcher (`START_WINDOWS.bat`/`START_LINUX.sh`) thành 1 thư mục tự chứa, chạy độc lập không cần server/mạng/`datasets/`. `merge --decisions <path>` gộp `decisions.jsonl` từ pack về workspace chính, chặn task_id lạ, bỏ qua bản trùng y hệt. Pack full 30,425 ảnh không filter đã build: `/home/tts01/label_audit_offline.zip` (2.7GB) — user nên pack có filter `--issue` cho lần dùng thật để nhỏ hơn.
- Verified: py_compile; curl POST decision (status/label_source hợp lệ + không hợp lệ); export end-to-end (file nhãn chỉ chứa 0/1); chromium headless dispatch pointerdown/move/up thật cho nested-bbox (box to không đổi khi kéo box nhỏ và ngược lại — PASS); pack standalone serve từ thư mục pack không cần repo, POST decision, merge về, export ra đúng nhãn.
- Quy ước nhãn: `tasks/smoke_fire_detection/docs/label_convention_smoke_fire.md`. R1-R13 + 20 edge case (E1-E20), đã bỏ mọi nhắc `ignore`/nuisance-flag để khớp tool; bảng case→quyết định (mục 4bis) cho case lặp lại (khói mờ, khí thải, phản chiếu, TV, nến...).
- Rule quan trọng khác FASDD gốc: FASDD nói "khác màu rõ → khác object" sẽ tách sai 1 cột khói đổi màu theo độ cao. R3 sửa: tách CHỈ khi khác **gốc phát**. Thêm R3b trần ~6 bbox/ảnh, R8 fill ratio ≥0.35, R9 cấm bbox ≥90% ảnh khi khói thật <60% (lỗi thật trong `annotated_fire_smoke_2025`), R12 cấm bbox lồng cùng class.
- Ưu tiên audit: `possible_geometry_mismatch` (9,504 = 31% queue) giá trị THẤP nhất vì metric lab là event recall. User chốt audit HẾT 30,425 ảnh → chiến lược hạ giá/ảnh (Enter/W 1 phím ≈ 4s/ảnh ≈ 34h), không bỏ ảnh.
- Cleanup 2026-08-12 (2 lượt): xóa `near_field_audit_sources.zip` (15GB dup `datasets/`), 2 review zip root (2.9GB), 3 detector cache experiment đã đóng, `near_field_round1_review.html/.json`, `near_field_visual_audit_sample.json`, `near_field_model_audit.py`, `near_field_review.py`, `near_field_model_audit_kaggle.ipynb` (v1, user cho xóa — "mất thì tạo lại"), `datasets/smoke_fire_detection/PYRONEAR-2025` raw 9.2GB (đã có bản clean dẫn xuất). `artifacts/smoke_fire_detection` 16G→953M rồi tổ chức lại thư mục con (xem mục dưới). `datasets/near_field_fire_detection/FIRESENSE` gộp vào `datasets/smoke_fire_detection/FIRESENSE` — trước đó có 2 root `datasets/` con song song không có lý do, giờ chỉ còn 1.
- Giữ nguyên: mọi checkpoint/config/metrics training, cache winner+firesense, figlib index/split/lock, mọi audit/lineage JSON, `human_review_labels.json`, toàn bộ dataset (kể cả PYRONEAR-2025-clean vì user muốn giữ dataset cháy rừng cho sau).
- CẢNH BÁO reproducibility đã hết do xóa notebook: nếu cần audit lại từ đầu (model khác, threshold khác), phải viết lại `near_field_model_audit_kaggle.ipynb`. Registry 7 source (root/splits/labels/class_map) không còn lưu ở đâu trong repo ngoài suy ra từ `tasks.jsonl.gz` hiện có — nếu cần regenerate, đọc lại `SOURCES` cũ từ git history (`git log -p -- tasks/smoke_fire_detection/near_field_model_audit_kaggle.ipynb`) trước khi viết mới.

## Artifacts reorganize 2026-08-12

- `artifacts/smoke_fire_detection/` giờ có cấu trúc con, không còn ~90 file JSON/checkpoint nằm chung 1 tầng:
  - `checkpoints/<candidate>/` (checkpoint .pt/.pth + config/metrics đi kèm), `checkpoints/external/` (Pyronear, SmokeyNet onnx).
  - `benchmark/{gate,compare,temporal,diagnostic,profile}/` — theo loại kết quả benchmark.
  - `dataset_audit/` — mọi JSON audit/lineage/group-split nguồn near-field.
  - `figlib/` — index, split manifest, winner lock, human_review_labels.
  - `detector_cache/` — jsonl cache + sidecar meta (đã xóa `.errors.json` không dùng).
  - `label_audit/` — workspace tool label (đổi tên từ `near_field_label_audit`).
- Path cứng đã sửa theo cấu trúc mới: `tasks/results_dashboard/backend/models.py` (2 checkpoint path); `tasks/smoke_fire_detection/modal_app.py` (mọi default arg path); `tasks/smoke_fire_detection/label_audit_tool.py` (default workspace). `tasks/results_dashboard/exporter/export.py` đổi `glob`→`rglob` + helper `find_artifact()` cho 2 chỗ dùng path cố định (`figlib_{split}_temporal_*`, `figlib_detector_cache_{cid}_dev.jsonl`) — giờ tìm theo tên bất kể nằm thư mục con nào, không cần sửa lại nếu cấu trúc đổi tiếp.
- CHƯA verify: chạy thật `modal_app.py`/dashboard sau khi sửa path (cần Modal/GPU, ngoài khả năng phiên này). Chỉ verify bằng `grep` xác nhận không còn path cũ nào sót.
