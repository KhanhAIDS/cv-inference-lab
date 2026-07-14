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
