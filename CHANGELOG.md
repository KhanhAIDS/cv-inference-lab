# CHANGELOG

## 2026-07-20 18:26 GMT+7 — Giai đoạn 3: migrate job sang systemd persistent scope

- Bối cảnh: job cache FIgLib dev (8 lượt, `/tmp/run_giai_doan3_cache.sh`) đang chạy trong `session-14.scope` — không sống nổi nếu SSH/laptop đóng. User tự chạy `loginctl enable-linger tts01` (agent bị permission classifier chặn, không tự chạy được).
- Verify trước khi động vào: `loginctl show-user tts01 | grep -i linger` → `Linger=yes`.
- Dừng an toàn chain cũ: `kill -TERM` lần lượt PID python đang chạy (`rc=143` ghi log) rồi PID bash cha (`228791`/`228811`); phát hiện 1 process con bị mồ côi (step yolo26n_dfire_control, PPID chuyển về `1`) do dừng không đúng thứ tự — kill nốt.
- Verify không hỏng dữ liệu trước khi relaunch: `wc -l` + check dòng cuối JSON hợp lệ cho 3 file cache dở dang (`yolo26x_dfire` 24,015 dòng, `rfdetr_large_dfire` 32 dòng, `yolo26n_dfire_control` 208 dòng) — cả 3 `ends-ok`.
- Relaunch: `systemd-run --user --scope --unit=giai-doan3-cache --working-directory=<repo> bash /tmp/run_giai_doan3_cache.sh >> /tmp/giai_doan3_cache.log 2>&1`. Script tự resume từ đầu (bước 1-2 resume tức thì vì đã xong, bước 3 tiếp tục từ dòng cũ) — không mất frame.
- Verify cgroup process mới: `cat /proc/<pid>/cgroup` → `user@1003.service/app.slice/giai-doan3-cache.scope` (không còn `session-*.scope`).
- File gián tiếp (job tự ghi tiếp, không phải AI trực tiếp sửa): `artifacts/smoke_fire_detection/figlib_detector_cache_{yolo26x_dfire,rfdetr_large_dfire,yolo26n_dfire_control}_dev.jsonl`, `/tmp/giai_doan3_cache.log`.

## 2026-07-20 18:26 GMT+7 — Quyết định resolution winner + đính chính label D-Fire

- User xác nhận: `yolo26x_dfire` và `rfdetr_large_dfire` train trên D-Fire label ĐÃ RELABEL (khác `yolo26n_dfire_control` dùng label gốc). Không relabel thêm.
- User chốt qua `AskUserQuestion`: winner dùng resolution `1280` cho cả 4 candidate sạch (kể cả 2 model D-Fire train@800) — theo đúng plan gốc, không đổi sang native-resolution-per-arm. Cache `@800` D-Fire vẫn chỉ diagnostic.
- Việc mở giao lại cho Giai đoạn 4: so AUROC@1280 vs @800 của 2 model D-Fire, báo rõ nếu @800 vượt trội — không giấu, không tự đổi rule.
- File trực tiếp sửa:
  - `tasks/smoke_fire_detection/docs/research_plan.md` — sửa dòng mô tả kế hoạch relabel D-Fire (2026-07-14) từ "ghi nhận, chưa thực thi" → "đã thực thi", ghi rõ chỉ áp dụng cho 2 run fresh, không áp dụng cho `yolo26n` control.
  - `tasks/smoke_fire_detection/docs/post_train_benchmark_plan.md` — thêm dòng D-Fire label variant ở Giai đoạn 0; thêm xác nhận resolution winner + câu hỏi mở @1280 vs @800 ở Giai đoạn 3.
  - `agent_context.md` — viết lại toàn bộ, gọn hơn (nén lịch sử Giai đoạn 0-2 đã đóng), thêm fact label D-Fire, xác nhận resolution winner, cập nhật trạng thái Giai đoạn 3 thật, thêm mục bàn giao Giai đoạn 4+ sang agent/session khác.
  - `CHANGELOG.md` — user đã tự xóa trắng trước đó (hành vi bình thường theo quy ước); ghi lại từ đây, không phục hồi nội dung cũ từ git.

## 2026-07-20 18:26 GMT+7 — Bàn giao Giai đoạn 4+ sang agent/session mới

- Lý do: hội thoại hiện tại đã dài; user muốn agent khác/session khác tiếp quản Giai đoạn 4 (compare-matrix + winner-lock) trở đi, kèm ràng buộc thời gian hạn chế + mục tiêu so sánh SOTA cho report.
- Agent hiện tại KHÔNG tự chạy Giai đoạn 4 trong session này — chỉ chuẩn bị context (`agent_context.md`, plan docs) + soạn prompt bàn giao gửi trực tiếp cho user (không lưu file riêng, xem prompt trong hội thoại).

## 2026-07-20 19:11 GMT+7 — Audit dọn codebase `tasks/smoke_fire_detection` (session khác, tiếp quản)

- Yêu cầu user: rà file/code/function thừa trong `tasks/smoke_fire_detection`, không đụng job cache Giai đoạn 3 đang chạy nền (`giai-doan3-cache` scope, `rfdetr_large_dfire` ~74%).
- Phạm vi audit: 7 file Python (`dataset.py`, `eval.py`, `modal_app.py`, `temporal_eval.py`, `test_eval.py`, `test_temporal_eval.py`, `report/demo_inference.py`, tổng 2874 dòng), 4 file `docs/*.md`, 3 notebook, `report/`.
- Phương pháp: đọc toàn bộ code; AST-check import không dùng (0 kết quả); đối chiếu tên hàm với toàn bộ codebase để tìm hàm chết (0 kết quả thật — mọi hit "count=1" là test method/`setUp`/`__init__`/`@app.local_entrypoint` gọi qua reflection/framework, không phải code chết); chạy lại test suite xác nhận không hỏng gì.
- Kết quả: codebase đã qua nhiều vòng dọn trước đó (ghi nhận trong `research_plan.md`: "đã trim", "đã gộp", "file cũ đã xóa") — không còn hàm/file thừa đáng kể. Duy nhất tìm thấy `__pycache__/` (cache biên dịch, đã gitignore, không track git) → xóa.
- Việc CHỦ ĐỘNG KHÔNG làm (đã có quyết định treo/ghi rõ trong `post_train_benchmark_plan.md`, không phải thẩm quyền của audit này): không restore/xóa subcommand `spatial-persistence overlay` đã bị trim khỏi `temporal_eval.py` (mục Giai đoạn 4 của plan, chờ quyết định a/b); không đụng `artifacts/smoke_fire_detection/` (dọn cache/pilot/profiling thừa đã lên kế hoạch ở Giai đoạn 5, chưa tới lượt, đang có job ghi nền).
- Quan sát phụ, không sửa (ngoài scope "redundant", cần user quyết riêng): `report/demo_inference.py` mới demo 3/6 candidate (thiếu 2 model D-Fire fresh + Pyronear reference); `modal_app.py:evaluate` không có action tương ứng trong `pyro_sdis_cli` local_entrypoint (chỉ gọi được qua `modal run ...::evaluate` trực tiếp).
- File trực tiếp xóa: `tasks/smoke_fire_detection/__pycache__/` (5 file `.pyc`, tự sinh lại khi chạy test — đã xóa lại lần 2 sau khi verify).
- Command đã chạy: `.venv/bin/python -m unittest tasks.smoke_fire_detection.test_eval tasks.smoke_fire_detection.test_temporal_eval -v` (31 test, pass).

## 2026-07-20 19:25 GMT+7 — Giai đoạn 3: phát hiện crash cache, relaunch + đổi thứ tự script

- Bối cảnh: agent mới tiếp quản Giai đoạn 4 (theo bàn giao trong `agent_context.md`). Kiểm tra job Giai đoạn 3 đang chạy nền — user yêu cầu "kiểm tra lại" sau báo cáo ETA lần đầu.
- Phát hiện: unit `giai-doan3-cache.scope` chết âm thầm giữa chừng candidate `rfdetr_large_dfire` (dừng ở 24,383/28,360 dòng lúc ~18:55 GMT+7). `systemctl status` báo nhầm `active (running)` dù `Tasks: 0` — do lỗi phụ lúc start unit (`Failed to add ... inotify watch ... No space left on device`, cạn inotify instance chứ không phải disk, disk còn 2.9T) nghi làm systemd mất theo dõi cgroup exit. Không xác định được nguyên nhân gốc (không có quyền đọc `dmesg`/kernel log, không có NOPASSWD sudo — thử `sudo -n dmesg` bị từ chối). Dòng cache cuối cùng vẫn là JSON hợp lệ, không hỏng, an toàn resume.
- Command chẩn đoán đã chạy: `wc -l`, `pgrep -af`, `ps -p <pid>`, `systemctl --user status/is-active/is-failed`, `journalctl --user --since ...`, `journalctl -k` (rỗng do không có quyền, xác nhận bằng `journalctl -k -n 5` trả "No entries" + hint thiếu group `adm`/`systemd-journal`), `nvidia-smi -q` (ECC/Xid đều N/A), `free -h`, `df -h`, `cat /proc/sys/fs/inotify/max_user_watches max_user_instances`.
- User duyệt trực tiếp: dừng unit cũ + relaunch + đổi thứ tự script ưu tiên 2 cache D-Fire `@800` (cần cho câu hỏi bắt buộc so AUROC@1280 vs @800) lên trước `yolo26n_dfire_control`/`pyronear_yolov8s_reference` (không cần cho winner) — vì ràng buộc thời gian, user sẽ tắt máy sớm và giao agent/session khác chọn winner.
- Command thực thi:
  - `systemctl --user stop giai-doan3-cache.scope` (dừng sạch unit kẹt).
  - Sửa `/tmp/run_giai_doan3_cache.sh` (scratch, không thuộc repo) — dời 2 dòng `run_cache ... 800 ...` (yolo26x_dfire, rfdetr_large_dfire) lên trước dòng `yolo26n_dfire_control`/`pyronear_yolov8s_reference`.
  - `systemd-run --user --scope --unit=giai-doan3-cache-r2 --working-directory="$(pwd)" bash /tmp/run_giai_doan3_cache.sh >> /tmp/giai_doan3_cache.log 2>&1` — unit mới thay `giai-doan3-cache` (unit cũ đã stop, tránh trùng tên). Verify `loginctl show-user tts01 | grep -i linger` vẫn `Linger=yes`.
- Verify sau relaunch: `pgrep -af run_giai_doan3_cache` thấy PID bash+python mới; `systemctl --user status giai-doan3-cache-r2.scope` — `Tasks: 21`, đang chạy `rfdetr_large_pyro_sdis` (resume, sẽ ghi 0 frame mới vì đã xong).
- Quyết định user thêm (19:20 GMT+7, ghi vào `agent_context.md`): nếu AUROC@800 vượt rõ AUROC@1280 cho 2 model D-Fire → viết thẳng là **oversight quy trình** trong report (winner đã khóa bằng @1280 trước khi có số @800), không giấu, không tự đổi ngược winner.
- File trực tiếp sửa: `agent_context.md` (thêm mục crash/relaunch/thứ tự mới, mục SOTA facts, viết lại mục "Next" thành checklist lệnh sẵn sàng chạy); `/tmp/run_giai_doan3_cache.sh` (scratch, không thuộc repo — liệt kê vì ảnh hưởng trực tiếp tiến trình nền).
- File gián tiếp (job tự ghi tiếp): `artifacts/smoke_fire_detection/figlib_detector_cache_rfdetr_large_dfire_dev.jsonl` (tiếp tục tăng dòng), `/tmp/giai_doan3_cache.log` (log unit mới append vào cùng file).
- Việc khác đã làm song song (không đổi file): research SOTA qua agent con (FIgLib/SmokeyNet/Pyronear/D-Fire baseline mAP/latency GPU) — kết quả đã lưu vào `agent_context.md` mục "SOTA facts", không lưu file riêng, tránh trùng lặp với static doc trước khi report chính thức viết.

## 2026-07-20 20:33 GMT+7 — Cache 4 candidate @1280 DONE, compare-matrix chạy nền persistent, bàn giao lần 2 (user rời server)

- Cache `rfdetr_large_dfire`@1280 đạt 28,360/28,360 dòng lúc ~20:13 GMT+7 (unit `giai-doan3-cache-r2.scope`), verify dòng cuối JSON hợp lệ. Cả 4 candidate sạch @1280 giờ DONE.
- Command đã chạy: `.venv/bin/python -m tasks.smoke_fire_detection.temporal_eval compare-matrix ...` (đầy đủ tham số `--candidate`/`--architecture`/`--dataset`/`--dataset-order pyro,dfire`/`--latency-ms` cho 4 candidate sạch, xem nguyên văn trong `agent_context.md` mục "Next"). Lần chạy đầu (foreground, tự động chuyển background do >120s) bị tôi chủ động `kill` sau ~8 phút để chuyển sang chạy bền vững qua systemd (bootstrap 1000 mẫu không có checkpoint giữa chừng — chạy foreground/background thường sẽ mất nếu session ngắt kết nối giữa chừng).
- Command relaunch bền vững: viết `/tmp/run_giai_doan4_compare_matrix.sh` (scratch, không thuộc repo) chứa đúng lệnh compare-matrix; `systemd-run --user --scope --unit=giai-doan4-compare-matrix --working-directory="$(pwd)" bash /tmp/run_giai_doan4_compare_matrix.sh >> /tmp/giai_doan4_compare_matrix.log 2>&1`. Verify: `systemctl --user status giai-doan4-compare-matrix.scope` active, `pgrep -af compare-matrix` thấy PID python. Ước ~20-40 phút (thuần CPU, không progress log giữa chừng).
- Bàn giao lần 2: user phải rời server (mang máy về) trước khi compare-matrix + phần còn lại của Giai đoạn 4 xong. Đã rà soát và bổ sung `agent_context.md` cho đủ chi tiết thực thi không cần hỏi lại: field name thật của `temporal_eval.py temporal` output (`false_alarm_rate_per_hour`, `event_recall`, `ttd_median_seconds`), công thức quy đổi FA/ngày và FA/tuần sang FA/giờ, template JSON đầy đủ cho `figlib_dev_winner_lock.json`, nhắc kiểm tra file `figlib_dev_compare_matrix.json` đã tồn tại chưa trước khi chạy lại (tránh tốn công trùng).
- File trực tiếp sửa: `agent_context.md` (viết lại toàn bộ mục "Next" thành checklist thực thi chi tiết + field name + công thức quy đổi + template JSON); `CHANGELOG.md` (mục này).
- File trực tiếp tạo (scratch, không thuộc repo): `/tmp/run_giai_doan4_compare_matrix.sh`.
- File gián tiếp (job nền tự ghi, chưa xong lúc viết mục này): `artifacts/smoke_fire_detection/figlib_dev_compare_matrix.json` (chưa xuất hiện), `/tmp/giai_doan4_compare_matrix.log`.

## 2026-07-20 20:51 GMT+7 — Commit + push lên GitHub để user làm tiếp từ local clone (không VPN, mất kết nối server nội bộ)

- Bối cảnh: user phải rời mạng công ty (mang laptop về nhà), không có VPN nên không SSH lại được server `ai2` (GB10) tới khi quay lại công ty. Kiểm tra phát hiện repo có remote GitHub (`https://github.com/KhanhAIDS/cv-inference-lab.git`) nhưng local đang ahead origin 6 commit chưa push — nếu không push, clone từ nhà sẽ thiếu toàn bộ tiến độ Giai đoạn 0-3.
- Phát hiện thêm: `tasks/smoke_fire_detection/temporal_eval.py` (compare-matrix/temporal/g0/diagnose) không phụ thuộc torch/ultralytics/rfdetr, chỉ cần `numpy`+`rich` — verify bằng cách đọc import ở đầu file. Cache detection JSONL (`figlib_detector_cache_*.jsonl`) đã track trong git, KHÔNG bị `.gitignore` (chỉ `datasets/`, `*.pt/*.pth/*.onnx/*.ckpt/*.engine`, `.venv/` bị ignore). => toàn bộ phần phân tích còn lại của Giai đoạn 4 (compare-matrix, operating point, winner-lock, report) chạy được ở local Windows không cần GPU/server, miễn đã `git pull` cache JSONL.
- User xác nhận: commit + push ngay.
- Command đã chạy: `git add CHANGELOG.md agent_context.md artifacts/smoke_fire_detection/figlib_detector_cache_yolo26x_dfire_dev_res800.jsonl`; `git commit -m "..."`; `git push origin master`.
- File trực tiếp sửa (commit): `CHANGELOG.md`, `agent_context.md` (thêm mục "Làm việc từ local clone" — phân định rõ cái gì có/không có sau `git pull`, việc nào chạy được/không chạy được ở local, setup `py -m pip install numpy rich`).
- File trực tiếp thêm vào commit (tiến độ Giai đoạn 3, đang dở, không phải final): `artifacts/smoke_fire_detection/figlib_detector_cache_yolo26x_dfire_dev_res800.jsonl` (partial, job vẫn chạy tiếp trên server sau khi push — sẽ cần pull lại bản đầy đủ sau).
- Đã push kèm 6 commit trước đó của phiên trước (Giai đoạn 0-2, RF-DETR backend, profiling, v.v.) — lần đầu các commit này lên GitHub.
