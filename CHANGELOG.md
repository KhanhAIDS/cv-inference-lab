- **2026-07-14 07:57:04 +0700**
  - **Task:** poll Modal training `yolo26x_pyro_sdis_budget9`; xác minh resume/checkpoint.
  - **Result:** container `ta-01KXERQ6TWZND3DTZS7A4W5N8R`; epoch `4/20`, batch `4095/7385`; L4; batch `4`; imgsz `1280`.
  - **Checkpoint:** `last_resume.pt` epoch 3; optimizer/EMA/scaler/scheduler/train args/updates đủ; ngắt giữa epoch 4 sẽ lặp epoch 4.
  - **Files trực tiếp:** `agent_context.md`; `CHANGELOG.md`.
  - **Files gián tiếp:** Modal log/training tiếp tục trên Volume; không stop/redeploy/sửa Volume.

- **2026-07-14 09:40:19 +0700**
  - **Task:** trim code theo user duyệt; giữ training/resume/inference weight.
  - **Code:** bỏ D-Fire, hard-negative, tiling/motion probes, smoke/resume-gate, CPU preflight, staging cache/direct-copy, checkpoint manifest/hash/snapshot; giữ `train_segment`, full-state `last_resume.pt`, `best.pt`/`last.pt`, Pyro/FIgLib eval.
  - **Sửa:** Modal evaluate mặc định budget9/best.pt + Pyro val + imgsz 1280; evaluator hỗ trợ YAML trỏ thư mục ảnh.
  - **Test:** syntax 6/6; converter 3/3; parser; Modal import; segment args; directory split; unused import; diff check PASS.
  - **Files trực tiếp:** `tasks/smoke_fire_detection/dataset.py`; `eval.py`; `temporal_eval.py`; `train.py`; `modal_app.py`; `agent_context.md`; `CHANGELOG.md`.
  - **Files gián tiếp:** xóa Python bytecode cache `tasks/smoke_fire_detection/__pycache__`; temp test dirs tự xóa.
  - **Modal:** không deploy; không stop/poll container; không sửa Volume/checkpoint/weight.

- **2026-07-14 09:45:00 +0700**
  - **Task:** rút gọn tài liệu; xóa timeline và trạng thái outdated; giữ snapshot run cuối, checkpoint contract, code trim, next step.
  - **Commands:** `Get-Content agent_context.md -Encoding utf8`; `Select-String CHANGELOG.md`; `Get-Date`; `apply_patch`.
  - **Files trực tiếp:** `agent_context.md`; `CHANGELOG.md`.
  - **Files gián tiếp:** không có.

- **2026-07-14 09:59 +0700**
  - **Task:** trả lời câu hỏi kế hoạch (đọc research_plan/research_toolbox/agent_context, không đổi plan); dọn cache tự sinh theo quy tắc tối giản.
  - **Commands:** `nvidia-smi -L`; `ls tasks/smoke_fire_detection/`; `rm -rf tasks/smoke_fire_detection/__pycache__`.
  - **Files trực tiếp:** xóa `tasks/smoke_fire_detection/__pycache__/`; append `CHANGELOG.md`.
  - **Files gián tiếp:** không có.

- **2026-07-14 10:34 +0700**
  - **Task:** đồng nhất số liệu FIgLib giữa các tài liệu; tạo split dev/final; bỏ hash lock + cost ledger theo quyết định user; thêm rule chống complexity layer + rule static/volatile doc; sửa dẫn chiếu hỏng trong research_plan.
  - **Commands:** `.venv/bin/python tasks/smoke_fire_detection/dataset.py figlib --data-root datasets/smoke_fire_detection/FIgLib --index-out artifacts/smoke_fire_detection/figlib_index.jsonl --audit-out artifacts/smoke_fire_detection/figlib_audit.json --manifest-out artifacts/smoke_fire_detection/figlib_split_manifest.json`; grep/đọc `dataset.py`, `pyro_sdis_audit.json`.
  - **Kết quả verify:** 511 folder, 510 sequence có frame hợp lệ, 143 camera-string, 40362 frame; camera 136→143 do parser fallback brand-token cho 71 folder dạng dash (parser change, không phải data change); split camera-disjoint seed 20260707: dev 359 seq/108 cam, final 152 seq/35 cam, overlap 0; 13a gate PASS khớp audit (`invariant_differences={}`).
  - **Files trực tiếp:** `CLAUDE.md` (thêm rule mục 5 + mục 10); `tasks/smoke_fire_detection/docs/research_plan.md` (cập nhật số liệu, bỏ hash lock/SHA shard/cost ledger, sửa toàn bộ dẫn chiếu agent_context/artifact chết, đánh dấu 13a done, human review done); `agent_context.md`; `CHANGELOG.md`.
  - **Files gián tiếp:** `artifacts/smoke_fire_detection/figlib_index.jsonl`, `figlib_audit.json`, `figlib_split_manifest.json` (script tự sinh).
  - **Modal:** không deploy/stop/poll; không đụng Volume/checkpoint.

- **2026-07-14 11:12 +0700**
  - **Task:** bỏ hoàn toàn budget cap $20/free-quota-stop-rule theo yêu cầu user (không giữ constraint chi phí train nào); giải thích lại cascade/multi-head/Frigate bằng ngôn ngữ dễ hiểu (chỉ trả lời chat, không đổi code/plan cho phần này).
  - **Files trực tiếp:** `tasks/smoke_fire_detection/docs/research_plan.md` (xóa dòng "Chi phí" 13, sửa 13c/13e/kiểm-thử-bắt-buộc bỏ chữ "budget"/"cap"); `agent_context.md` (xóa dòng Budget); `CHANGELOG.md`.
  - **Files gián tiếp:** không có.

- **2026-07-14 13:16:08 +0700**
  - **Task:** phản biện hệ thống nhận diện gần–xa; sửa hiểu lầm FIgLib `60s`/Modal/GPU/RQ1; research distillation, small-object architecture, dynamic/coarse-to-fine/temporal inference; bổ sung architecture ablation plan.
  - **Kết quả:** A/B/C đổi thành taxonomy A–I; gần–xa đổi thành scale/visibility continuum; `13d` xác nhận chưa có winner; distillation phân loại theo inference graph; thêm S0–S3 deployment envelope, G4 architecture, E8a detector KD.
  - **Architecture:** inventory A0–A13; ưu tiên data-scale control → P2/stride-4 → P2↔P5 compute-match → adaptive ROI; kernel/neck/head/loss/formulation/temporal/routing/KD; ladder L0–L4; iso-config + iso-cost; scale/event/cost metric; pass/stop rule.
  - **Research web:** paper gốc FPN, PANet, EfficientDet/BiFPN, TridentNet, SAHI, AutoFocus, QueryDet, DynamicDet, ScaleKD, Localization Distillation, Focal Loss, ATSS, Dynamic Head, FGFA, MEGA, SmokeyNet, PYRONEAR.
  - **Commands:** `sed -n '1,240p' agent_context.md`; `uname -a`; `git status --short`; `sed` research plan/toolbox + `tail CHANGELOG.md`; `wc -l`; `git diff`; `nl -ba`; `rg`; `apply_patch`; `git diff --check`; `TZ=Asia/Ho_Chi_Minh date`.
  - **Verify:** Linux ARM64 detected; `git diff --check` PASS; doc-only, không chạy Python/model/test; không poll/stop/deploy Modal.
  - **Files trực tiếp:** `tasks/smoke_fire_detection/docs/research_plan.md`; `agent_context.md`; `CHANGELOG.md`.
  - **Files gián tiếp:** không có.

- **2026-07-14 15:05:55 +0700**
  - **Task:** audit lineage 3 model (`pyronear/yolov8s`, `dfire_yolo26n_baseline_full_vram`, `yolo26x.pt`/Pyro-SDIS fine-tune). Chỉ research + đọc git history, không Modal/GPU/benchmark.
  - **Kết quả chính:** claim cũ trong research_plan.md ("Pyronear train trên pyro-sdis, không dính HPWREN/FIgLib") SAI/không có nguồn — model card HF + repo `pyro-vision` + paper PYRONEAR-2025 đều không công bố dataset train của checkpoint `pyronear/yolov8s`. Theo decision rule: lineage không sạch → loại Pyronear khỏi toàn bộ luồng so sánh (kể cả zero-shot control).
  - **D-Fire (`dfire_yolo26n_baseline_full_vram`):** khôi phục từ git history (`args.yaml` tại commit `ceee019`, script gốc tại `02c7a59`/`faeff13^`) — model `yolo26n.pt` (base COCO), fine-tune D-Fire (`gaiasd/DFireDataset`, nguồn Belo Horizonte fire simulation + UFMG/Serra Verde surveillance + Internet, không liên quan HPWREN/FIgLib/Pyro-SDIS), imgsz `640`, seed `20260707`, val carved 10% từ train gốc (stratified theo label type), test = D-Fire test gốc giữ nguyên, augmentation Ultralytics default (mosaic 1.0, hsv/translate/scale/fliplr default, không override).
  - **YOLO26x:** base `yolo26x.pt` pretrain COCO 640×640 (Ultralytics official, MuSGD, batch 128) — xác nhận qua `docs.ultralytics.com/models/yolo26` + arXiv 2606.03748; fine-tune Pyro-SDIS đúng theo research_plan hiện có (imgsz 1280/epochs 20/patience 5/seed 20260707/batch 4), augmentation cũng Ultralytics default (train.py không override) — cùng augmentation policy với D-Fire, khác imgsz/epoch/patience theo domain.
  - **Fairness gap phát hiện thêm:** Pyro-SDIS chỉ có train/val (không test riêng) trong khi D-Fire có train/val/test — YOLO26x chọn epoch VÀ report Pyro val cùng một split; D-Fire baseline train imgsz 640 vs YOLO26x imgsz 1280 — so sánh tuyệt đối giữa 2 model nội bộ bị confound bởi resolution nếu không tách riêng theo domain.
  - **Web research:** `huggingface.co/pyronear/yolov8s` (model card + raw README + file list), `huggingface.co/datasets/pyronear/pyro-sdis`, `github.com/pyronear/pyro-vision`, arXiv 2402.05349 (PYRONEAR-2025), `github.com/gaiasd/DFireDataset`, `docs.ultralytics.com/models/yolo26`, arXiv 2606.03748.
  - **Commands:** `find`/`grep` D-Fire/yolo26x refs; `git log --all --oneline`; `git log --all --diff-filter=A --name-only`; `git show <hash>:<path>` (args.yaml, train_dfire_yolo.py, dfire_baseline_report.md); `git log -p -- dataset.py` cho split D-Fire; `TZ=Asia/Ho_Chi_Minh date`.
  - **Files trực tiếp:** `tasks/smoke_fire_detection/docs/research_plan.md` (sửa 3 chỗ: lineage check mục 12, baseline 13b, dev candidate 13d — bỏ Pyronear); `CHANGELOG.md`.
  - **Files gián tiếp:** không có. Không tạo file mới/artifact/hash/manifest; không chạy Modal/GPU/benchmark.

- **2026-07-14 15:35:47 +0700**
  - **Task:** user override quyết định lineage: giữ Pyronear trong so sánh (leakage-caveat thay vì loại); ghi nhận kế hoạch relabel D-Fire; verify lại seed 2 model theo chất vấn user.
  - **Verify seed:** D-Fire — `args.yaml` của run thật (git `ceee019`) ghi `seed: 20260707`, `deterministic: true`; YOLO26x — `train.py` dòng 491 hard-gate `ValueError` nếu `seed != 20260707`, khớp poll log 07:57. Chưa đọc `train_args` trong checkpoint budget9 trên Modal Volume (ngoài scope, không đụng Modal) — bằng chứng là code gate + poll log, không phải checkpoint trực tiếp.
  - **Sửa plan:** mục 12 lineage check — đổi decision từ "loại Pyronear" sang "giữ, mọi số FIgLib/Pyro-SDIS mang leakage-caveat, trục so ít rủi ro nhất là D-Fire test"; 13b — thêm lại Pyronear vào baseline cache; 13d — Pyronear vào dev nhưng không được khóa winner từ số FIgLib khi lineage chưa xác minh, final chạy với vai trò reference; mục 8.1 — thêm ghi nhận kế hoạch relabel D-Fire + ràng buộc so sánh (FIgLib axis so được, D-Fire mAP axis vỡ nếu đổi test label, closed results giữ làm control lịch sử, kỳ vọng gain không nằm ở scale-gap).
  - **Commands:** `git show ceee019:...args.yaml | grep seed`; `sed -n '488,497p' train.py`; `TZ=Asia/Ho_Chi_Minh date`; `git diff --check` (PASS).
  - **Files trực tiếp:** `tasks/smoke_fire_detection/docs/research_plan.md` (4 chỗ: mục 12, 13b, 13d, 8.1); `CHANGELOG.md`.
  - **Files gián tiếp:** không có. Không tạo file mới; không chạy Modal/GPU/benchmark.

- **2026-07-14 16:00:22 +0700**
  - **Task:** trả lời phản biện tiếp theo của user — (1) số mAP D-Fire 0.684 đang so với cái gì, (2) relabel thực ra chỉ <1% ảnh, (3) plan đã có candidate model family ngoài YOLO (RF-DETR...) chưa, (4) giải thích rõ rủi ro imgsz zero-shot của Pyronear.
  - **Kết quả:** xác nhận 0.684 KHÔNG bị khóa vào gate/thống kê nào khác trong plan (G0/G1 dùng AUROC từ detector cache, độc lập label D-Fire) — sửa lại framing "mất tính so sánh" trước đó, quá thận trọng cho quy mô <1%; relabel giờ ghi nhận đơn giản hơn (report song song 2 mAP, không cần đóng băng test); xác nhận **plan hiện KHÔNG có candidate model family ngoài YOLO** ở track chọn winner (13a-13e) — transformer chỉ xuất hiện như module thay thế nội bộ (A3/L4), sau G4; thêm gap note.
  - **Không sửa thêm gì về Pyronear imgsz** (câu hỏi 4 chỉ yêu cầu giải thích thêm trong chat, không đổi plan).
  - **Commands:** `git diff --check` (PASS); `TZ=Asia/Ho_Chi_Minh date`.
  - **Files trực tiếp:** `tasks/smoke_fire_detection/docs/research_plan.md` (mục 8.1 sửa lại relabel framing; mục 13 thêm gap note model family); `CHANGELOG.md`.
  - **Files gián tiếp:** không có. Không tạo file mới; không chạy Modal/GPU/benchmark.

- **2026-07-14 16:41:12 +0700**
  - **Task:** verify thật imgsz train của `pyronear/yolov8s` (câu hỏi user: "có cách nào kiểm tra rồi ghi vào không") — không cần Modal/GPU, chỉ đọc metadata file model đã export.
  - **Commands:** tải `yolov8s.onnx` (44.5MB) từ `huggingface.co/pyronear/yolov8s` về scratchpad session (ngoài repo); `.venv/bin/pip install onnx` (tạm thời, không ghi vào `requirements.lock`); `onnx.load(...).metadata_props`; `.venv/bin/pip uninstall -y onnx` + xóa file scratchpad ngay sau khi đọc xong (dọn sạch, không để lại dependency/artifact thừa).
  - **Kết quả:** metadata Ultralytics export xác nhận `imgsz=[1024,1024]`, `names={0:'smoke'}`, `stride=32`, `date=2024-05-30`, `description` trỏ path nội bộ `pyronear-mlops/data/03_model_input/yolov8/full/datasets/data.yaml`. Phát hiện thêm: ngày export (2024-05-30) **sớm hơn** ngày `pyro-sdis` tự nhận "full release January 2025" trên dataset card → không thể đồng nhất tuyệt đối checkpoint này với bản `pyro-sdis` HF hiện tại (nhiều khả năng cùng nguồn camera nhưng khác snapshot). Không đổi kết luận leakage risk (vẫn chưa loại trừ HPWREN/ALERTWildfire), chỉ thay giả thuyết imgsz từ "không rõ" thành "1024 xác nhận".
  - **Sửa plan:** mục 12 (thêm bullet verify), 13b (Pyronear cache 2 lượt `imgsz=1024` số chính + `1280` số phụ, bỏ giả thuyết 640).
  - **Files trực tiếp:** `tasks/smoke_fire_detection/docs/research_plan.md`; `CHANGELOG.md`.
  - **Files gián tiếp:** không có trong repo (file `.onnx` tải về nằm ở scratchpad ngoài project, đã xóa sau khi đọc; `onnx` package cài tạm vào `.venv` rồi gỡ ngay, không đổi `requirements.lock`).
