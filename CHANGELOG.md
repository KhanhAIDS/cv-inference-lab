## 2026-07-22 08:35 GMT+7 — SmokeyNet subsample xong; chốt số SOTA anchor vào report tĩnh

- Job nền `smokeynet-subsample30.scope` hoàn tất qua đêm: 2366/2366 frame, 0 lỗi, 30 sequence.
- `temporal_eval.py g0`: AUROC(smoke) = **0.8092** trên subsample (CI event `[0.730,0.895]`, CI camera `[0.728,0.893]` — rất rộng do n=30 sequence).
- So paired đúng cách với winner: lọc cache winner xuống ĐÚNG cùng 2366 frame-key của subsample (winner đo lại trên riêng subset này = 0.7798, khác biệt rõ so với AUROC toàn dev 0.8317 — minh chứng biến thiên mẫu nhỏ, không phải winner đổi). `compare-candidates` Δ=+0.0294, CI event `[-0.065,0.133]`, CI camera `[-0.069,0.126]` — **CI chứa 0 → "tie_or_not_proven"**, không kết luận được SmokeyNet hơn/kém winner. Giới hạn do compute (CPU-only, không đủ chạy full dev), không phải giới hạn phương pháp.
- **Đã ghi toàn bộ 2 số SOTA anchor (Pyronear yolo11s + SmokeyNet) vào report tĩnh chính thức** `tasks/smoke_fire_detection/docs/model_benchmark_2x2.md` mục 9.1 (mới) — đúng quy tắc "số chốt ghi inline vào file static, không dẫn chiếu file volatile".
- File trực tiếp tạo: `gate_g0_auroc_smokeynet_reference_subsample30.json`, `compare_winner_vs_smokeynet_dev_subsample30.json`.
- File trực tiếp sửa: `tasks/smoke_fire_detection/docs/model_benchmark_2x2.md` (mục 9.1 mới), `CHANGELOG.md` (mục này).
- File tạm (scratch, không thuộc repo): `winner_dev_subsample30_matched.jsonl` (lọc winner cache xuống subset để so paired).
- Còn treo (chưa tự quyết, chờ user): (1) mở `--split test` final cho 2 candidate reference mới hay không; (2) Kaggle fine-tune PYRONEAR-2025 — user chưa báo đã chạy; (3) mitigation catastrophic-forgetting cho notebook fine-tune (giữ nguyên/trộn D-Fire/giảm LR) — user chưa chọn.
- Cập nhật `agent_context.md` cùng lượt.

## 2026-07-22 11:45 GMT+7 — Khởi động lại results dashboard

- User yêu cầu kích hoạt frontend. `results-dashboard.service` không còn tồn tại trong user systemd (`Unit ... could not be found`).
- Khởi động trực tiếp nền: `nohup .venv/bin/python -m uvicorn tasks.results_dashboard.backend.app:app --host 0.0.0.0 --port 8731 > /tmp/results-dashboard-8731.log 2>&1 &`.
- Verify: `curl http://127.0.0.1:8731/api/models` HTTP 200; trả đúng 2 model, cả hai `resident=false`; PID `65395`; listener `0.0.0.0:8731`.
- LAN IP ưu tiên: `192.168.1.186`; truy cập `http://192.168.1.186:8731/` từ cùng mạng. IP `100.70.229.110`/Docker bridge không dùng cho LAN thường.
- Command đọc/diagnostic: `uname -a`; `sed -n ... agent_context.md`; `rg --files tasks/results_dashboard`; `sed ... README.md package.json app.py`; `systemctl --user status`; `hostname -I`; `tail`; `curl`.
- File trực tiếp sửa: `CHANGELOG.md` (mục này).
- File gián tiếp tạo ngoài repo: `/tmp/results-dashboard-8731.log`, `/tmp/results-dashboard-8731-health.json`.

## 2026-07-22 11:47 GMT+7 — Correction process dashboard

- `nohup` PID `65395` bị harness dừng sau khi command kết thúc; không còn dùng.
- Tạo transient user service bền: `systemd-run --user --unit=results-dashboard --working-directory=/home/tts01/Luyen_Minh_Khanh/cv-inference-lab .venv/bin/python -m uvicorn tasks.results_dashboard.backend.app:app --host 0.0.0.0 --port 8731`.
- Verify cuối: `systemctl --user is-active results-dashboard.service` = `active`; `curl --retry ... http://127.0.0.1:8731/api/models` HTTP 200; PID `66582`; 2 model đều `resident=false`.
- File trực tiếp sửa: `CHANGELOG.md` (mục này).
- File gián tiếp: transient unit `/run/user/1003/systemd/transient/results-dashboard.service`; systemd journal.
- File scratch `/tmp/results-dashboard-8731.log` và `/tmp/results-dashboard-8731-health.json` từ thử `nohup` còn tồn tại, ngoài repo.

## 2026-07-22 15:02 GMT+7 — Ghi gap "lửa xa trong D-Fire" vào research_plan.md; research kiến trúc production + dataset (không đụng code)

- Bối cảnh: user phản biện loạt giải thích trước (kiến trúc pipeline/hardware/dataset/label strategy/distillation) — không chạy command GPU/training, chỉ research web (2 agent: kiến trúc smart-city production, dataset lửa-xa) + 1 lần đọc code `rfdetr/detr.py` verify claim reinit head + 1 sửa doc.
- Research xác nhận: không có dataset public bbox cho lửa-xa thật (multi-km, fixed-tower) — mọi dataset khói-xa đã biết (FIgLib/PYRONEAR-2025/NEMO/Fuego) verify trực tiếp README/paper là smoke-only, không có lớp fire. Không tải gì thêm (không có gì đạt tiêu chuẩn).
- User phản biện: D-Fire CÓ ảnh lửa xa (quan sát trực tiếp dataset, không phải suy từ median NE4) — median không loại trừ tail. Đã ghi gap này vào `research_plan.md` mục 10.4 (dưới NE4): chưa audit riêng đuôi phân phối lớp fire, chưa phân biệt được giả thuyết "domain gap" vs "thiếu mẫu lửa xa trong train" cho kết quả AUROC(fire)=0.509. Chưa audit — chỉ ghi gap, chưa chạy.
- File trực tiếp sửa: `tasks/smoke_fire_detection/docs/research_plan.md` (mục 10.4, thêm bullet gap), `CHANGELOG.md` (mục này).
- Không file gián tiếp nào (không chạy script/training lượt này).

## 2026-07-22 17:35 GMT+7 — Threshold sweep miễn phí trên cache FIRESENSE có sẵn (lớp fire); trả lời loạt câu hỏi treo

- Bối cảnh: user hỏi loạt quyết định treo (final split 2 candidate reference, mitigation forgetting Kaggle, domain PYRONEAR-2025, dataset near-field có gồm D-Fire không) + đưa kết quả FIRESENSE control eval từ 1 agent/session khác (đã tự ghi vào `research_plan.md` mục 10.7, `agent_context.md`, `CHANGELOG.md` trước đó — không phải lượt này).
- Đọc lại `agent_context.md` + `research_plan.md` mục 10.6-10.7 để lấy đúng state mới nhất trước khi trả lời (không đoán).
- Chạy 1 lệnh Python inline (không file mới) đọc trực tiếp `artifacts/smoke_fire_detection/firesense_detector_cache_rfdetr_large_dfire.jsonl` đã có sẵn — quét ngưỡng conf lớp fire, không cần GPU/data mới. Kết quả: `thr=0.65-0.70` giữ nguyên recall 11/11, giảm false-alarm 6/16→3/16; `thr=0.80` giảm còn 1/16 nhưng mất 1 true positive yếu nhất; xác nhận rank-inversion (`negsVideo9.866`=0.810 > `posVideo1.868`=0.706) khiến không threshold nào đạt đồng thời 100% recall + 0% FA.
- Ghi kết quả sweep vào `research_plan.md` mục 10.7 (bullet mới, đánh dấu rõ "chưa áp dụng, chờ user xác nhận" — không tự chốt threshold thay user).
- File trực tiếp sửa: `tasks/smoke_fire_detection/docs/research_plan.md` (mục 10.7, thêm bullet threshold sweep), `CHANGELOG.md` (mục này).
- Không file gián tiếp nào (chỉ đọc artifact JSONL có sẵn, không chạy script/training/tải dataset lượt này).

## 2026-07-22 16:48 GMT+7 — Near-field bước 1: FIRESENSE control eval (checkpoint D-Fire zero-shot) — lộ lỗ hổng fire-class, mở bước 2

- Bối cảnh: RQ5 near-field đã duyệt mở (research_plan.md mục 10.6 point 1), NE1-NE5 chưa chạy. Nhiệm vụ: chạy control eval checkpoint `rfdetr_large_dfire` (đã có sẵn, train thuần D-Fire, mAP50-95 0.48) zero-shot trên FIRESENSE (49 video, CC BY 4.0, đã tải sẵn) — tương đương C0 của far-field, KHÔNG train gì.
- `eval.py detector-cache` không dùng được (cứng schema FIgLib, cần `ignition_offset_seconds`; FIRESENSE chỉ có nhãn pos/neg cấp video) → viết script mới tách riêng, đúng tiền lệ `smokeynet_cache.py`.
- Viết `tasks/smoke_fire_detection/firesense_cache.py`: trích 1 frame/giây/video (OpenCV, stride=round(native_fps)), infer RF-DETR qua `RFDETRLarge.from_checkpoint` (tái dùng `class_map_from_names`/`rfdetr_box_records` từ `eval.py`, không viết lại parse detection), gộp max-confidence/lớp mỗi video, batch=8. Smoke-test 2 video trước khi chạy full.
- Viết `tasks/smoke_fire_detection/firesense_gate.py`: tính AUROC pooled + per-category (smoke/fire) + bootstrap CI (video-level, 1000 resample, seed 20260707) — tái dùng nguyên `auroc`/`auroc_bootstrap_ci` từ `temporal_eval.py`, không viết lại toán bootstrap. Thêm báo cáo riêng NG2 (trick-negative: mean/median/n-above-threshold/danh sách video vượt 0.5 mỗi category).
- Command chạy full: `.venv/bin/python tasks/smoke_fire_detection/firesense_cache.py --weights artifacts/smoke_fire_detection/runs/rfdetr_large_dfire/checkpoint_best_total.pth --out artifacts/smoke_fire_detection/firesense_detector_cache_rfdetr_large_dfire.jsonl --target-fps 1.0 --imgsz 1280 --conf 0.05 --batch 8`. Chạy nền ~439s (GPU GB10 dùng chung với ~10 process khác trên server, không phải lỗi script), 49/49 video, 0 lỗi. Sau đó `.venv/bin/python tasks/smoke_fire_detection/firesense_gate.py --cache artifacts/smoke_fire_detection/firesense_detector_cache_rfdetr_large_dfire.jsonl --out artifacts/smoke_fire_detection/gate_firesense_auroc_rfdetr_large_dfire.json`.
- **Kết quả:** pooled AUROC = 0.977 (CI `[0.933,1.0]`), smoke AUROC = 0.991 (CI `[0.95,1.0]`), fire AUROC = 0.983 (CI `[0.932,1.0]`) — cả 3 PASS gate 0.80 rất rộng, cao hơn cả AUROC winner trên FIgLib (0.8317). Không miss trên positive (video pos thấp nhất vẫn đạt 0.604 smoke/0.706 fire).
- **NG2 trick-negative (toàn bộ 25/49 video neg đều là trick-negative theo thiết kế gốc dataset, không có tập negative "thường" riêng):** smoke neg vô hại (mean 0.161, 1/9 vượt 0.5). **Fire neg có vấn đề thật: mean 0.352, 6/16 (37.5%) vượt 0.5, đỉnh 0.81 — 3 video (`negsVideo9/5/4`) confidence CAO HƠN true positive thấp nhất (0.706, `posVideo1`), tức rank-inversion thật, không phải suy diễn.**
- **Quyết định: MỞ bước 2, nhắm riêng lớp fire** — bằng chứng false-alarm cụ thể (rank-inversion), không phải AUROC thấp chung chung. Ưu tiên **FireAndSmoke** (github.com/CostiCatargiu/NEWFireSmokeDataset_YoloModels) trước FASDD vì có sẵn lớp "other" hard-negative (nắng/đèn/ánh sáng xe) khớp đúng vấn đề vừa phát hiện. FASDD (Kaggle mirror) hoãn — server không có `kaggle` CLI/credential (`~/.kaggle/` không tồn tại), giống tình trạng PYRONEAR-2025 trước đây; cần user tự làm qua Kaggle UI nếu muốn nguồn này.
- File trực tiếp tạo: `tasks/smoke_fire_detection/firesense_cache.py`, `tasks/smoke_fire_detection/firesense_gate.py`, `artifacts/smoke_fire_detection/firesense_detector_cache_rfdetr_large_dfire.jsonl` (+ `.meta.json`), `artifacts/smoke_fire_detection/gate_firesense_auroc_rfdetr_large_dfire.json`.
- File trực tiếp sửa: `tasks/smoke_fire_detection/docs/research_plan.md` (mục 10.7 mới), `CHANGELOG.md` (mục này), `agent_context.md`.
- File tạm (scratch, ngoài repo): `firesense_smoke_test.jsonl`, `firesense_gate_smoke_test.json`, `firesense_full_run.log` tại scratchpad — không thuộc repo.
- Tiếp theo (cùng lượt, đang làm): tải/convert/dedup FireAndSmoke cho bước 2, chưa động vào train (bước 3 chờ user).

## 2026-07-22 16:54 GMT+7 — Bước 2: cả 3 nguồn dataset ưu tiên đều bị chặn ngoài tầm kiểm soát — dừng lại, chờ user

- Thử tải theo đúng thứ tự ưu tiên đã chốt (mục 10.5 research_plan.md), chỉ đọc/probe, không cài đặt gì mới ngoài `gdown` (đã có sẵn trong venv).
- **FireAndSmoke** (git clone `github.com/CostiCatargiu/NEWFireSmokeDataset_YoloModels`, repo chỉ 236KB): xác nhận repo chỉ chứa script `DownloadFireDataset.py`, dữ liệu thật host trên Roboflow Universe (2 project). Link export cứng trong script (vd `https://universe.roboflow.com/ds/7dbfrMSkjo?key=2yEW8dC3nG`) trả `HTTP 403 cf-mitigated: challenge` — Cloudflare JS-challenge thật, test cả plain curl lẫn browser User-Agent đều bị chặn. Không cố vượt challenge (đúng nguyên tắc không evasion).
- **FASDD**: không thử tải — server không có `kaggle` CLI, `~/.kaggle/` không tồn tại (đã biết từ trước, PYRONEAR-2025).
- **MS-FSDB** (git clone `github.com/XiaoyiHan6/MS-FSDB`, license MIT rõ nhất): dữ liệu host Google Drive, link trong README trỏ file ID `14ylxaNBVmXjAFXt2h4lnyBe7xELhOVHc`. Thử `gdown` (lỗi "cannot retrieve public link") rồi verify tay bằng `curl` theo redirect chain (`drive.google.com/uc?export=download` → `drive.usercontent.google.com/download`) → **HTTP 200 nhưng nội dung là trang lỗi Google 404 thật** (không phải quota/permission) — file đã bị tác giả xóa/di chuyển.
- **Indoor Fire Smoke** (Zenodo record 15826133): verify record còn sống, `curl -A "Mozilla/5.0..." ` trả HTTP 200 bình thường (HEAD trần bị 403 nhưng GET với User-Agent chuẩn thì được — không phải bot-block thật). Nguồn này CÓ THỂ tải được, nhưng theo đúng điều kiện đã chốt trước ("chỉ nếu domain indoor thật sự liên quan tới mục tiêu") — không tự quyết thay user.
- **Quyết định: DỪNG bước 2 tại đây, không tự đổi sang nguồn khác ngoài danh sách đã chốt, không cố vượt Cloudflare/Kaggle-gate, không tự phán đoán domain indoor thay user.** Đã ghi đầy đủ vào `research_plan.md` mục 10.7 kèm 3 lựa chọn cho user (tự xác thực Roboflow/Kaggle thủ công; duyệt Indoor Fire Smoke; hoặc không mở dataset mới, đổi hướng khác cho lớp fire near-field).
- Không tải/convert/dedup dataset nào thành công — không có file gián tiếp nào tạo trong `datasets/`. Không đụng vào train (bước 3 vẫn chờ user, đúng scope giao ban đầu).
- File trực tiếp sửa: `tasks/smoke_fire_detection/docs/research_plan.md` (mục 10.7, cập nhật phần quyết định bước 2), `CHANGELOG.md` (mục này).
- File tạm ngoài repo (scratchpad, không thuộc repo): `firesmoke_probe/` (git clone FireAndSmoke), `msfsdb_probe/` (git clone MS-FSDB), `msfsdb_test.zip` (rỗng/lỗi, không phải dataset thật), `gdrive_page.html`, `zenodo_test.html` — toàn bộ probe artifact, không cần dọn (session-scoped).

## 2026-07-22 19:26 GMT+7 — Khảo sát 5 link dataset user đề xuất (FASDD/furg/aiformankind/Boreal/zenodo) + câu hỏi UAV — chỉ đọc, không tải

- Command: `WebFetch`/`WebSearch` trên 5 URL user đưa (không tải file, chỉ đọc trang/metadata), `curl` probe `scidb.cn` (API trả 404, page JS-render nên không lấy được nội dung đầy đủ).
- **FASDD** (github.com/openrsgis/FASDD): xác nhận host thật là ScienceDB (`scidb.cn`, DOI `10.57760/sciencedb.j00104.00103`), KHÔNG PHẢI Kaggle — nhưng trang JS-render, chưa xác nhận được có cần tài khoản không.
- **furg-fire-dataset** (github.com/steffensbola/furg-fire-dataset): license CC0, nhưng domain lẫn indoor/outdoor/robot-di-động (không phải fixed camera), format XML cũ (OpenCV 2.4.9, không phải YOLO/COCO), ~30 video, repo ngừng cập nhật — đánh giá không phù hợp near-field track hiện tại.
- **aiformankind/wildfire-smoke-dataset**: xác nhận nguồn ảnh là camera cố định HPWREN (KHÔNG PHẢI UAV — đính chính nếu có nhận định trước đó nói ngược lại), license CC BY-NC-SA, cùng họ camera network với FIgLib/PYRONEAR nên rủi ro chính là leakage far-field, không phải domain.
- **Boreal Forest Fire** (etsin.fairdata.fi, paper Nature Sci Data 2025): xác nhận UAV thật 100% (DJI Phantom 4, 4K, prescribed burn Phần Lan), Subset A (4954 ảnh + bbox txt) nhẹ hơn Subset B (292 video 4K — có lẽ là phần user thấy "nặng").
- **zenodo.org/records/15826133**: xác nhận đây chính là "Indoor Fire Smoke" đã biết từ trước (research_plan.md mục 10.7), không phải nguồn mới.
- Trả lời câu hỏi UAV của user: dự án hiện không có track UAV (kiến trúc mục 2.3 theo camera cố định far-field/near-field). Ghi nhận UAV là track thứ 3 tiềm năng (giống cách `FLAME 3` đã parked), chưa mở, không tự quyết thay user.
- File trực tiếp sửa: `tasks/smoke_fire_detection/docs/research_plan.md` (mục 10.5/10.7 — thêm bullet khảo sát 5 nguồn + ghi chú UAV, thêm lựa chọn (d) ScienceDB), `CHANGELOG.md` (mục này).
- Không tải/cài gì mới, không đụng dataset/train. File gián tiếp: không có.

## 2026-07-22 19:51 GMT+7 — Soát `to_be_resolved/` (user tự tải) + user phản biện trục cố định/di động + cảnh báo mismatch scenario

- User phản biện: trục "chỉ dùng camera cố định" có thể quá khắt khe (deployment thật chưa chốt, có thể là robot/UAV tuần tra); và cho rằng near-field hiện tại (mục 10.2, dựa FM 3232/ISO 7240-29/UL 268B/BS 5839-1 — toàn chuẩn hệ báo cháy LẮP CỐ ĐỊNH) không khớp scenario thật nào user hình dung — nêu nghi vấn hiểu lầm nghiêm trọng trong plan.
- Command: `unzip -l`/`unzip -p` (chỉ đọc metadata/data.yaml/README, KHÔNG giải nén ảnh) trên 3 file zip có sẵn trong `to_be_resolved/` (user tự tải, không rõ nguồn gốc từ agent, đã luận ra qua metadata Roboflow nhúng trong data.yaml).
  - `DFS-Fire.v3i.yolov11.zip`: Roboflow `cv-atyqk/dfs-fire` v3, 8939 ảnh, YOLO 2 lớp `fire,smoke`, License Public Domain.
  - `FireAndSmoke.v1i.yolov11.zip`: Roboflow `undying/fireandsmoke-gr81i` v1, 100 ảnh, 1 lớp `firerotation`, CC BY 4.0 — xác nhận KHÁC dataset CostiCatargiu đã nhắm trước (không có lớp "other", không giải quyết vấn đề NG2).
  - `FIRE-SMOKE-DATASET_from_DeepQuestAI.zip`: classification-only (`Fire/Smoke/Neutral` folder, không bbox), không dùng trực tiếp cho RF-DETR.
- User xác nhận: FASDD tải được qua `scidb.cn` không cần login (gỡ block Kaggle-credential); Indoor Fire Smoke (zenodo 15826133) domain liên quan (duyệt).
- Trả lời: leakage-check có cách làm (kỹ thuật dedup giống PYRONEAR-2025-clean), chưa chạy. Reassess trục cố định/di động: đồng ý quá khắt khe, trục đúng là khoảng cách/góc nhìn/blur chứ không phải "camera di chuyển hay không".
- Hỏi lại user qua AskUserQuestion: kịch bản deployment thật là gì (robot mặt đất / UAV / camera cố định / chưa biết-muốn tổng quát). **User trả lời: "Vẫn giả định camera cố định, các cái khác thì thôi để sau."**
- **Quyết định (2026-07-22 19:54 GMT+7): giữ nguyên mục 10.1-10.3 (envelope S2/gate NG dựa FM3232/ISO7240-29/UL268B/BS5839-1), không đổi. Robot mặt đất/UAV tuần tra: hoãn, không mở track song song.** furg-fire-dataset (phần robot) và Boreal Forest Fire (UAV) vẫn ở trạng thái "biết nguồn, chưa dùng" — không cần hành động thêm.
- File trực tiếp sửa: `tasks/smoke_fire_detection/docs/research_plan.md` (mục 10.8 mới + cập nhật quyết định), `CHANGELOG.md` (mục này).
- Không giải nén/copy/di chuyển file trong `to_be_resolved/` — chỉ đọc metadata. File gián tiếp: không có.
