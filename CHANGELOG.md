- 2026-07-16 19:04 +07:00
  - Mục tiêu: xử lý 4 quyết định user duyệt sau khi chất vấn đợt trim trước — xóa `requirements*.lock` chết, untrack `best.pt` ngoại lệ cho nhất quán, xóa notebook/script D-Fire Kaggle/Colab (user xác nhận training đã xong), viết doc external assets.
  - Xóa `tasks/smoke_fire_detection/requirements.lock` + `requirements-cu130.lock` (đã gỡ chỗ dùng `add_lock()` lượt trước).
  - `git rm --cached artifacts/smoke_fire_detection/runs/dfire_yolo26n_baseline_full_vram/weights/best.pt` — untrack, file vẫn còn trên đĩa (5.2M), demo vẫn chạy được local. Không còn ngoại lệ track/untrack giữa các weight.
  - Xóa `colab_train_dfire_yolo26x.ipynb`, `kaggle_train_dfire_rfdetr.ipynb` (từ `tasks/smoke_fire_detection/`), `train_dfire_yolo26x.py`, `train_dfire_rfdetr.py` — user xác nhận D-Fire YOLO26x/RF-DETR-L trên Kaggle/Colab đã train xong, không cần script nữa. **Không đụng** `train_rfdetr.py` (Pyro-SDIS, đang cần resume từ `checkpoint_7.ckpt`, khác tên dễ nhầm với `train_dfire_rfdetr.py`).
  - Thêm `tasks/smoke_fire_detection/docs/external_assets.md` — liệt kê dataset (`D-Fire/`, `pyro-sdis/`, `pyro-sdis-yolo/`, `FIgLib/`) + checkpoint (`dfire_yolo26n_baseline_full_vram`, `yolo26x_pyro_sdis_budget9`, `rfdetr_large_pyro_sdis_gb10`) không track git, kèm nguồn/cách lấy lại. Lưu ý: `D-Fire-train-ready.zip` đã xóa nên `datasets/smoke_fire_detection/D-Fire/` giờ là bản sao DUY NHẤT — không còn nơi tải lại, phải tự backup thủ công.
  - Verify: `py_compile` PASS toàn bộ `.py` còn lại trong `tasks/smoke_fire_detection/`.
  - Thay đổi trực tiếp: xóa `requirements.lock`, `requirements-cu130.lock`, `colab_train_dfire_yolo26x.ipynb`, `kaggle_train_dfire_rfdetr.ipynb`, `train_dfire_yolo26x.py`, `train_dfire_rfdetr.py`; untrack (không xóa file) `runs/dfire_yolo26n_baseline_full_vram/weights/best.pt`; thêm `tasks/smoke_fire_detection/docs/external_assets.md`; sửa `agent_context.md`, `CHANGELOG.md`.
  - Thay đổi gián tiếp: không có.
  - Command: `git rm --cached ...`; `git rm -f ...` (2 notebook, do vướng staged rename từ lượt trước); `git rm ...` (2 script); `du -sh datasets/smoke_fire_detection/*/`; `du -sh artifacts/.../weights/* .../*.ckpt .../*.pth`; `.venv/bin/python -m py_compile` toàn bộ `.py` còn lại.

- 2026-07-16 18:56 +07:00
  - Mục tiêu: user chất vấn lại đợt trim trước — phát hiện `pyro_sdis_snapshot.json` vẫn là cơ chế hash-lock (vi phạm rule "không hash" đã có từ trước) và yêu cầu gỡ hẳn khỏi code thay vì chỉ giữ file. Trả lời thêm về 2 best checkpoint RF-DETR, ý nghĩa "add metadata 3 run dir", tính nhất quán track/untrack `.pt`, và rà soát thêm `requirements*.lock` + notebook/script D-Fire.
  - **Gỡ hash-lock Pyro-SDIS (đã làm xong):**
    - `dataset.py`: xóa `import hashlib`, xóa `--expected-shards` CLI arg, xóa hàm `sha256_file`, xóa toàn bộ block validate SHA-256 shard + block "invariant_differences" trong `cmd_pyro_sdis` (cả hai chỉ được nuôi bởi `--expected-shards`, cùng gỡ). `cmd_pyro_sdis` giờ chỉ làm đúng 1 việc: convert parquet → YOLO layout + audit thống kê (không hash, không so khớp expected).
    - `modal_app.py`: xóa hẳn 2 Modal function `convert_pyro_sdis`/`package_pyro_sdis` (one-time dataset packaging đã xong, archive đã có sẵn trên Modal volume, không còn lý do rerun); xóa nhánh `action == "convert"|"package"` + 4 param liên quan (`data_root`,`out`,`audit_out`,`expected_shards`) khỏi `pyro_sdis_cli`; xóa import `SimpleNamespace` (hết chỗ dùng). Đã kiểm tra `sha256_path`/`PYRONEAR_SHA256` KHÔNG đụng — dùng cho checkpoint kill-resume audit + verify checksum third-party model, phạm vi khác, ngoài scope câu hỏi này.
    - Xóa `artifacts/smoke_fire_detection/pyro_sdis_snapshot.json` (orphan hoàn toàn sau khi gỡ code trên).
    - Sửa `test_pyro_sdis_converter.py` cho khớp signature mới (bỏ `expected_shards=None, workers=None` khỏi lệnh gọi test).
    - Verify: `py_compile` PASS cả 3 file; `python -m unittest tasks.smoke_fire_detection.test_pyro_sdis_converter` chạy thật PASS 3/3; smoke-import `modal_app.py` PASS; `dataset.py pyro-sdis --help` xác nhận hết `--expected-shards`.
  - **Điều tra `requirements.lock`/`requirements-cu130.lock` (đã gỡ chỗ dùng, CHƯA xóa file — chờ user xác nhận):** xác nhận 2 file này bị `add_lock()` copy vào Modal image nhưng KHÔNG có chỗ nào chạy `pip install -r` từ chúng — version cài thật trong `yolo_image` là hardcode trực tiếp trong lời gọi `pip_install(...)`, và version đó cũng lệch với `requirements.lock` (vd `numpy` lock ghi `2.5.1`, `.venv` cài `2.4.6`; `pyarrow` lock `21.0.0` vs `.venv` `24.0.0`). Đã xóa hàm `add_lock` + 2 lời gọi trong `modal_app.py` (dead code thật, đã verify compile/import PASS). CHƯA xóa 2 file `.lock` — hỏi lại user trước.
  - Thay đổi trực tiếp: sửa `tasks/smoke_fire_detection/dataset.py`, `tasks/smoke_fire_detection/modal_app.py`, `tasks/smoke_fire_detection/test_pyro_sdis_converter.py`, `agent_context.md`, `CHANGELOG.md`; xóa `artifacts/smoke_fire_detection/pyro_sdis_snapshot.json`.
  - Thay đổi gián tiếp: `py_compile`/`unittest` sinh `tasks/smoke_fire_detection/__pycache__/` (đã xóa lại sau mỗi lần verify).
  - Command: `.venv/bin/python -m py_compile ...`; `.venv/bin/python -m unittest tasks.smoke_fire_detection.test_pyro_sdis_converter -v`; `python -c "import ..."`; `python -m tasks.smoke_fire_detection.dataset pyro-sdis --help`; `.venv/bin/pip freeze | grep -iE ...` (so version lock vs venv thật).

- 2026-07-16 18:20 +07:00
  - Mục tiêu: trim/refactor lần 1 (server ai2, Linux) — thay dataset D-Fire, dọn tooling/artifact đã xong việc, dọn checkpoint dở dang, gitignore weight, trim dead code, purge `.git` rác, merge kết quả train YOLO26x/Pyro-SDIS, dọn report folder, viết lại `agent_context.md`.
  - Kết quả tổng: dung lượng repo `64G → 51G` (`du -sh .`); `.git` `2.7G → 132M`.

  - **1. Thay dataset D-Fire:**
    - `unzip -t D-Fire-train-ready.zip` PASS (12.6s) → giải nén tạm `datasets/smoke_fire_detection/_dfire_extract_tmp/` → audit: `43,055` file; train `15,500`/valid `1,721`/test `4,306` — khớp kỳ vọng.
    - Rename `D-Fire/` → `D-Fire_old/`; move bản mới vào `D-Fire/`; audit lại PASS (giống hệt số trên).
    - Xóa `D-Fire_old/` (3.0G), `datasets/smoke_fire_detection/D-Fire.zip` (3.0G, xác nhận cũ hơn qua mtime 2026-07-15 07:08 so với zip mới 2026-07-16 10:03, cùng layout `D-Fire/{train,valid,test}`), `D-Fire-train-ready.zip` ở root (3.0G, sau khi audit PASS).

  - **2. Dọn tooling + artifact D-Fire đã xong việc:**
    - Xóa `tasks/smoke_fire_detection/review_dfire_label_fixes.py`, `review_dfire_valid_labels.py` (chỉ còn tham chiếu lịch sử trong `CHANGELOG.md` cũ — chấp nhận được).
    - Move `colab_train_dfire_yolo26x.ipynb`, `kaggle_train_dfire_rfdetr.ipynb` từ root vào `tasks/smoke_fire_detection/` (`git mv`).
    - Grep toàn repo từng file trong `artifacts/smoke_fire_detection/*.json|md` trước khi xóa. Xóa: `dfire_label_fixes.json` (chỉ ref lịch sử), `human_review_pack_key.json` (0 tham chiếu), `human_review_labels.json`/`human_review_pack.md` (chỉ ref lịch sử trong `research_plan.md`, số liệu đã inline sẵn), `pyro_sdis_audit.json` (chỉ được ghi bởi `dataset.py`, không nơi nào đọc lại — `package_pyro_sdis` trong `modal_app.py` đọc một file audit ephemeral khác, không phải file này), `pyro_sdis_yolo_archive.json` (chỉ dùng làm idempotency short-circuit, tar archive tương ứng hiện không tồn tại local nên nhánh đó không kích hoạt). **Giữ** `pyro_sdis_snapshot.json` — vẫn được `dataset.py:cmd_pyro_sdis` và `modal_app.py:package_pyro_sdis` đọc thật để validate SHA-256 shard trước convert (live read-path, không phải log lịch sử). Giữ nguyên `figlib_index.jsonl`, `figlib_split_manifest.json`, `figlib_audit.json` theo yêu cầu.

  - **3. Checkpoint RF-DETR/Pyro-SDIS:** xóa `checkpoint_0.ckpt` → `checkpoint_6.ckpt` (mỗi file 566,181,923 byte, tổng ~3.7G) trong `artifacts/smoke_fire_detection/runs/rfdetr_large_pyro_sdis_gb10/`. Giữ `checkpoint_7.ckpt`, `checkpoint_best_ema.pth`, `checkpoint_best_regular.pth`, `metrics.csv`.

  - **4. `.gitignore` + git add metadata:**
    - Thêm `*.pt`, `*.ckpt`, `*.pth`, `*.onnx`, `*.engine` vào `.gitignore` (giữ nguyên rule cũ).
    - `git add` metadata: `runs/rfdetr_large_pyro_sdis_gb10/metrics.csv`; `runs/yolo26x_pyro_sdis_budget9/{args.yaml,results.csv,run_metadata.json,train_segment_state.json}`.
    - Verify: `runs/dfire_yolo26n_baseline_full_vram/weights/best.pt` (đã track từ trước) không bị ảnh hưởng bởi gitignore mới; weight 3 run dir đều bị ignore đúng (`git check-ignore -v`).

  - **5. Trim dead code:** dùng 1 agent research (đọc `dataset.py`, `train.py`, `train_rfdetr.py`, `eval.py`, `temporal_eval.py`, `modal_app.py`, `report/demo_inference.py`, `test_pyro_sdis_converter.py`) tìm code chết. Kết quả: hầu hết vẫn sống (Modal entrypoint, CLI subcommand, hoặc entrypoint chạy tay hợp lệ như `train.py:main()`). Chỉ xóa `--workers` CLI arg trong `dataset.py` (declare nhưng không đâu đọc `args.workers`) + dòng `workers=None` tương ứng truyền vào `cmd_pyro_sdis(...)` trong `modal_app.py`. Xóa `tasks/smoke_fire_detection/__pycache__/` (sinh ra từ bước verify, không phải pre-existing). Verify: `py_compile` PASS cả 10 file; smoke-import `dataset.py` + `modal_app.py` PASS; `dataset.py pyro-sdis --help` xác nhận hết `--workers`.

  - **6. Merge kết quả train YOLO26x/Pyro-SDIS (user duyệt):**
    - Thư mục root `yolo26x pyro sdis/` (tải từ Kaggle) là 2 epoch cuối (19-20) resume từ checkpoint budget9, hoàn tất **20/20 epoch**. Xác nhận qua `args.yaml` (`resume: .../last_resume.pt`, `epochs: 20`) + `results.csv` (epoch 19 mAP50-95=`0.4949`, epoch 20=`0.49303`).
    - Merge vào `artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis_budget9/`: copy `best.pt` + `args.yaml` đè bản cũ (epoch 17); nối `results.csv` (epoch 3-18 cũ + 19-20 mới = full history epoch 3-20, epoch 1-2 thiếu từ trước không phục hồi được); xóa `weights/last.pt` (113M) + `weights/last_resume.pt` (450M) — lỗi thời, training đã xong, không còn resume.
    - Xóa thư mục root `yolo26x pyro sdis/` sau merge.
    - **Epoch improvement summary (mAP50-95, val Pyro-SDIS):** tăng đơn điệu từ `0.352` (epoch 3) lên đỉnh `0.4949` (epoch 19), dao động nhẹ quanh epoch 10-18 (`0.462-0.492`) trước khi đạt đỉnh cuối; epoch 20 giảm nhẹ về `0.493`. Best = epoch 19.

  - **7. Report folder** (`tasks/smoke_fire_detection/report/`, chuẩn bị làm lại toàn bộ theo yêu cầu user): xóa `README.md`, `slide_script.md`, `demo_results/` (13M, ảnh + `summary.json` — gắn với checkpoint dở dang lúc viết báo cáo #1, nay đã lỗi thời vì YOLO26x đã xong 20/20). Giữ `demo_inference.py` (script tái dùng được, path checkpoint 3 model vẫn khớp hiện trạng). Xóa `report/__pycache__/`.

  - **8. `.git` gc/prune (user duyệt):** phát hiện `.git` pack chứa 5 blob checkpoint RF-DETR cũ (mỗi blob 566,181,923 byte, tổng `~2.64GB`) **unreachable** — không thuộc lịch sử branch hiện tại (`git fsck --unreachable --no-reflogs`), khả năng từ 1 commit từng bị amend/reset trước khi push. Chạy `git reflog expire --expire=now --expire-unreachable=now --all` + `git gc --prune=now`. Kết quả: `.git` `2.7G → 132M`; `git fsck --full` sạch; `git log`/`git status` vẫn nguyên vẹn. Lưu ý: đây là repo clone riêng trên server ai2, độc lập với `.git` máy Windows local (máy đó đã prune riêng lúc `2026-07-16 16:25 +07:00`, `2.9GiB → 53.7MiB`).

  - **9. `agent_context.md`:** viết lại — bỏ toàn bộ log Kaggle debug/GUI CRUD/notebook fix đã lỗi thời (đã xong việc, không còn actionable); cập nhật trạng thái mới: D-Fire final, YOLO26x DONE 20/20, RF-DETR còn dừng ở checkpoint_7, report folder đã dọn, trim 2026-07-16.

  - Thay đổi trực tiếp:
    - Sửa: `.gitignore`, `tasks/smoke_fire_detection/dataset.py`, `tasks/smoke_fire_detection/modal_app.py`, `agent_context.md`, `CHANGELOG.md`.
    - Move (`git mv`): `colab_train_dfire_yolo26x.ipynb`, `kaggle_train_dfire_rfdetr.ipynb` (root → `tasks/smoke_fire_detection/`).
    - Xóa: `tasks/smoke_fire_detection/review_dfire_label_fixes.py`, `review_dfire_valid_labels.py`; `artifacts/smoke_fire_detection/{dfire_label_fixes.json,human_review_labels.json,human_review_pack.md,human_review_pack_key.json,pyro_sdis_audit.json,pyro_sdis_yolo_archive.json}`; `artifacts/smoke_fire_detection/runs/rfdetr_large_pyro_sdis_gb10/checkpoint_{0,1,2,3,4,5,6}.ckpt`; `tasks/smoke_fire_detection/report/{README.md,slide_script.md,demo_results/}`; `datasets/smoke_fire_detection/{D-Fire_old/,D-Fire.zip}`; `D-Fire-train-ready.zip` (root); `yolo26x pyro sdis/` (root); `artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis_budget9/weights/{last.pt,last_resume.pt}`.
    - Ghi đè: `artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis_budget9/{weights/best.pt,args.yaml,results.csv}` (merge từ thư mục root).
    - `git add`: `artifacts/smoke_fire_detection/runs/rfdetr_large_pyro_sdis_gb10/metrics.csv`; `artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis_budget9/{args.yaml,results.csv,run_metadata.json,train_segment_state.json}`.
    - `git gc --prune=now` sau `git reflog expire --expire=now --expire-unreachable=now --all`.

  - Thay đổi gián tiếp:
    - `unzip` sinh thư mục tạm `datasets/smoke_fire_detection/_dfire_extract_tmp/` (tự xóa sau khi move xong).
    - `py_compile` sinh `tasks/smoke_fire_detection/__pycache__/` + `report/__pycache__/` (đã xóa lại).
    - `git gc` viết lại toàn bộ `.git/objects/pack/*` (pack mới, pack cũ 2.6G bị loại bỏ).

  - Command chính đã chạy:
    - `unzip -t D-Fire-train-ready.zip`; `unzip -q ... -d _dfire_extract_tmp`; `mv`/`rm -rf` swap D-Fire.
    - `git mv` 2 notebook; `git rm --cached` (không tác dụng, file chưa track) + `rm -f` cho review script/artifact json.
    - `rm -fv` checkpoint_0-6; `git add` metadata 3 run dir; `git check-ignore -v`.
    - `.venv/bin/python -m py_compile <file>` × 10; `.venv/bin/python -c "import ..."`; `python -m tasks.smoke_fire_detection.dataset pyro-sdis --help`.
    - `git rev-list --objects --all | git cat-file --batch-check`; `git verify-pack -v`; `git fsck --unreachable --no-reflogs`; `git reflog expire --expire=now --expire-unreachable=now --all`; `git gc --prune=now`; `git fsck --full`.
    - `cp`/`rm` merge `yolo26x pyro sdis/` vào `runs/yolo26x_pyro_sdis_budget9/`; `tail -n +2 results.csv >> results.csv`.
    - `git rm -r` report README/slide_script/demo_results.
    - `du -sh`/`git status --short`/`git log --oneline` để verify trước/sau ở từng bước.
