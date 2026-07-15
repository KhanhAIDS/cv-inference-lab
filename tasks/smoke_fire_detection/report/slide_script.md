# Slide Script — Early Smoke/Fire Detection (báo cáo lần 1)

- Đưa file này cho agent/người dựng slide. Không cần đọc lại toàn bộ repo — mọi số liệu cần thiết đã trích sẵn dưới đây, nguồn gốc: `README.md` cùng thư mục (bằng chứng đầy đủ hơn nếu cần).
- Ảnh dùng cho slide: lấy trong `demo_results/` (đường dẫn cụ thể ghi ở từng slide).
- Tông giọng: thẳng, có số liệu, không tô hồng. Đây là lab nghiên cứu — thất bại (tiling) và việc "chưa train xong" phải xuất hiện rõ ràng, không giấu.
- Ước tính: 15 slide, trình bày ~10-12 phút + demo trực tiếp ~3-5 phút.

---

## Slide 1 — Trang bìa

- Tiêu đề: **Early Smoke/Fire Detection từ camera cố định — báo cáo tiến độ**
- Sub: lab nghiên cứu CV inference — báo cáo lần đầu
- Không cần visual đặc biệt.

## Slide 2 — Bài toán & mục tiêu

- Nội dung:
  - Phát hiện khói/lửa sớm từ camera tháp canh cố định (early wildfire detection).
  - Metric mục tiêu thật: **time-to-detect (TTD)** và **false-alarm/giờ** — KHÔNG phải chỉ mAP.
  - Vì sao khó: khói ở target camera rất nhỏ/xa, khác hẳn ảnh cận cảnh thường dùng để train detector.
- Visual: không cần ảnh, dùng text lớn.
- Speaker note: nhấn mạnh ngay từ đầu đây là bài toán research thật, có nhiều ẩn số, không phải demo model có sẵn.

## Slide 3 — Phương pháp luận (rigor)

- Nội dung (bullet ngắn):
  - AMOC protocol: quét threshold, đo TTD vs FA/giờ theo curve — không chốt 1 điểm ngẫu nhiên.
  - Bootstrap CI theo event (≥1,000 resample) — 2 phương pháp chỉ coi là khác nhau khi CI không giao nhau.
  - Split camera-disjoint (không rò rỉ camera giữa train/eval).
  - Gate quyết định rõ ràng (G0→G1→G2→...) — mỗi bước có ngưỡng pass/fail định trước, không hậu kiểm.
- Visual: sơ đồ gate đơn giản G0→G1→G2→G3→G4 (có thể vẽ mũi tên ngang).
- Speaker note: đây là điểm khác biệt chính so với "chạy thử rồi báo cáo số đẹp nhất" — mọi quyết định đều có ngưỡng định trước.

## Slide 4 — Dữ liệu & hạ tầng

- Nội dung:
  - 3 dataset: D-Fire (21,527 ảnh, cận cảnh, có bbox), Pyro-SDIS (33,636 ảnh, camera tháp, có bbox), FIgLib (40,362 frame, camera tháp thật, KHÔNG có bbox — chỉ nhãn yếu theo mốc thời gian cháy).
  - Hạ tầng train đa nền tảng: Modal (cloud GPU on-demand), server GPU cục bộ (DGX GB10), Kaggle (free tier T4).
  - Checkpoint/resume đã kiểm chứng bằng thực nghiệm kill-resume (tắt đột ngột giữa lúc train, resume đúng optimizer/scheduler/EMA) — không phải giả định.
- Visual: 3 logo/tên dataset + số ảnh; hoặc bảng nhỏ.

## Slide 5 — Baseline đầu tiên (E0)

- Nội dung:
  - Model: YOLO26n, train trên D-Fire (domain cận cảnh).
  - Kết quả test set: **P=0.764, R=0.714, mAP50=0.684, mAP50-95=0.404**.
  - Riêng smoke: mAP50-95=0.482. Riêng fire: mAP50-95=0.326.
  - Latency: 21.6ms mean, 46.4 fps (GPU GB10, ảnh 640px).
- Visual: ảnh demo D-Fire nếu có box detect rõ (kiểm tra `demo_results/dfire_test/dfire_yolo26n_baseline/`; nếu ảnh mẫu hiện tại không có detection, dùng số liệu bảng thay ảnh — xem ghi chú Slide 13).
- Speaker note: baseline này CHỈ để học pipeline + làm điểm đối chứng, không phải model target.

## Slide 6 — Câu hỏi cốt lõi: model cận cảnh có transfer sang camera xa không?

- Nội dung:
  - Gate G0: chạy detector D-Fire (chưa từng thấy ảnh tháp canh) trên toàn bộ FIgLib, đo AUROC phân biệt frame trước/sau khi cháy xảy ra.
  - Kết quả: **AUROC = 0.70** (imgsz 640) → **0.7425** (imgsz 1280).
  - Ngưỡng quyết định: ≥0.80 pass thẳng, <0.60 fail hẳn, ở giữa là "marginal" — kết quả rơi đúng vùng marginal.
- Visual: thanh số trục AUROC 0.5→1.0, đánh dấu vùng fail/marginal/pass và điểm 0.70/0.7425.
- Speaker note: đây là phát hiện quan trọng — model tốt trên domain gần không tự động tốt ở domain xa.

## Slide 7 — Vì sao marginal: domain gap đo được

- Nội dung:
  - Per-sequence AUROC lưỡng cực: 26% sequence AUROC≥0.9 (dễ), 41% AUROC<0.6 (khó) — không phải detector yếu đều, mà "ăn trọn nhiều, mù nhiều".
  - Đo trực tiếp: bbox khói mà model phát hiện được ở FIgLib nhỏ hơn **~20 lần** so với bbox trung vị lúc train.
  - → Nguyên nhân định lượng, không phải suy đoán.
- Visual: 2 ảnh so sánh kích thước bbox (to ở data train vs nhỏ ở FIgLib) nếu có sẵn; nếu không, dùng con số "20x" làm big number.

## Slide 8 — THẤT BẠI: thử tiling, không hiệu quả (bắt buộc có trong báo cáo)

- Nội dung:
  - Ý tưởng: chia ảnh 2×2, detect từng ô độ phân giải gốc → hy vọng bắt được khói nhỏ hơn.
  - Kết quả: 39/131 sequence khó phục hồi được tín hiệu — **NHƯNG** tỷ lệ false-trigger tăng mạnh (zero-detection-rate ở frame negative giảm từ 0.956 xuống 0.787), 18/131 sequence chuyển từ "im lặng" sang "báo động giả".
  - Thử vá (chỉ giữ detection được ≥2 tile xác nhận): giảm được false alarm nhưng giết luôn phần lớn true-positive (số sequence phục hồi giảm còn 16/131).
  - Quyết định: **bỏ hướng tiling**, dùng thẳng kết quả imgsz=1280.
- Visual: 2 con số lớn đối lập — "39/131 phục hồi tín hiệu" vs "18/131 báo động giả mới" — hoặc mũi tên chỉ tăng/giảm.
- Speaker note (quan trọng): đây là minh chứng quy trình nghiên cứu nghiêm túc — thử, đo, thất bại có kiểm soát, ghi nhận, đóng lại đúng lúc thay vì cố đấm ăn xôi. Code hướng này đã dọn khỏi repo (giữ codebase gọn) nhưng số liệu vẫn giữ nguyên trong tài liệu.

## Slide 9 — Nhãn tạm dùng temporal có giúp gì không?

- Nội dung:
  - Gate G1: so single-frame vs N-of-M/EMA (gộp nhiều frame liên tiếp).
  - Ở ngân sách lỏng (FA≤1/ngày/camera): **không có lợi rõ ràng** so với single-frame.
  - Ở ngân sách chặt (FA≤1/tuần/camera): **có lợi thật** — TTD≈19 phút @ recall 21%, single-frame không đạt được ngân sách này ở bất kỳ ngưỡng nào.
  - Thêm: hướng "motion/frame-differencing" đã thử và đóng vĩnh viễn (AUROC=0.53, ~random).
- Visual: 2 cột so sánh FA≤1/ngày vs FA≤1/tuần.

## Slide 10 — Hạ tầng đa nền tảng (điểm mạnh kỹ thuật)

- Nội dung:
  - Train song song trên Modal (cloud), server GPU cục bộ, Kaggle — cùng một checkpoint di chuyển được giữa 3 nơi.
  - Xử lý thực tế: Modal ephemeral run tự dừng ở epoch 18/20 (không phải lỗi) → resume tiếp trên Kaggle; GPU cục bộ bị chia sẻ với job khác → đo bằng `nvidia-smi pmon` (không đoán) để quyết định ưu tiên job nào.
  - Checkpoint/resume kiểm chứng bằng kill-resume test thật trước khi chạy train dài không giám sát.
- Visual: sơ đồ 3 node (Modal / Server cục bộ / Kaggle) nối vào 1 checkpoint chung.

## Slide 11 — Model hiện có (bảng trạng thái)

- Bảng:

| Model | Vai trò | Trạng thái | mAP50-95 |
|---|---|---|---|
| YOLO26n / D-Fire | Baseline | Xong | 0.404 (test) |
| YOLO26x / Pyro-SDIS | Candidate chính | Đang train (18/20 epoch) | 0.492 (val, chưa final) |
| RF-DETR-L / Pyro-SDIS | Candidate DETR-family | Đang train (7/20 epoch) | 0.445–0.463 (val, chưa final) |

- Speaker note: nói rõ 2 model dưới CHƯA xong, số sẽ đổi.

## Slide 12 — Demo trực tiếp (hướng dẫn cho người trình bày)

- **Khuyến nghị: KHÔNG chạy inference trực tiếp lúc demo trên server dùng chung** — GPU server đang bị nhiều job khác chiếm (đã gặp OOM một lần khi chuẩn bị báo cáo này). Dùng ảnh đã sinh sẵn ở Slide 13 thay vì chạy live.
- Nếu bắt buộc chạy live (có kết nối server, môi trường đã sẵn `.venv`):
  1. `cd` vào repo, kích hoạt venv: `source .venv/bin/activate`.
  2. Chạy: `python tasks/smoke_fire_detection/report/demo_inference.py`.
  3. Script tự động thử GPU, tự fallback CPU nếu GPU đang bị chiếm hết (đã test).
  4. Kết quả mới ghi đè `demo_results/` — ảnh annotate + `summary.json`.
- Nếu offline/không server: chỉ trình chiếu ảnh có sẵn trong `demo_results/`, không cần chạy gì.

## Slide 13 — Kết quả demo (ảnh thật, dùng weight hiện tại)

- Ảnh 1: `demo_results/pyro_sdis_val/yolo26x_pyro_sdis/force-06_cabanelle-125_2024-02-24T09-07-27.jpg` — YOLO26x bắt đúng cột khói rất nhỏ ở xa, confidence 0.70.
- Ảnh 2: `demo_results/figlib_sample/rfdetr_large_pyro_sdis/1466360053_+00600.jpg` — RF-DETR-L (mới train 7/20 epoch) vẫn bắt được vệt khói mờ trên ảnh FIgLib thật (domain đích cuối cùng).
- Ảnh 3 (tùy chọn, để trung thực): bất kỳ ảnh nào trong `demo_results/dfire_test/` — cả 3 model đều 0 detection trên mẫu nhỏ 3 ảnh D-Fire test → dùng làm ví dụ "mẫu demo nhỏ không đại diện toàn bộ, số chính thức xem Slide 5/11".
- Speaker note: nhắc lại đây là 3 ảnh/dataset để minh họa, không phải eval chính thức toàn tập.

## Slide 14 — Giới hạn & rủi ro cần nói thẳng

- Nội dung:
  - 2/3 model demo dùng checkpoint giữa chừng — số liệu sẽ đổi khi train xong.
  - Chưa chạy gate chọn candidate chính thức (13d) trên FIgLib đầy đủ cho 2 model mới.
  - Chưa mở final camera-held-out (chỉ mở 1 lần sau khi khóa winner).
  - GPU cục bộ dùng chung với người khác → tốc độ train không ổn định, đã phải điều phối ưu tiên.
- Visual: không cần, dùng bullet.

## Slide 15 — Bước tiếp theo

- Nội dung:
  - Train xong YOLO26x (Kaggle, 2 epoch cuối) + RF-DETR-L (server cục bộ, 13 epoch còn lại).
  - Chạy detector cache full FIgLib cho cả 2 candidate → gate 13d chọn model chính thức bằng bootstrap CI (event + camera).
  - Mở final camera-held-out đúng 1 lần sau khi khóa winner.
- Kết thúc bằng câu hỏi/thảo luận.

---

## Ghi chú cho người dựng slide

- Không cần thêm hiệu ứng/animation phức tạp — nội dung là số liệu nghiên cứu, ưu tiên đọc được, không ưu tiên hoa mỹ.
- Nếu cắt bớt vì thời gian: giữ bắt buộc Slide 2, 5, 6, 8 (thất bại tiling), 11, 13, 14 — đây là khung sườn tối thiểu để không đánh mất tính trung thực của báo cáo.
- Nguồn số liệu gốc nếu cần đối chiếu thêm: `../docs/research_plan.md` (mục 6-8), `README.md` cùng thư mục.
