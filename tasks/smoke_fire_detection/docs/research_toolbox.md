# Research Toolbox — Smoke/Fire (v2)

- Updated: `2026-07-08`
- Vai trò: menu kỹ thuật + **điều kiện rút ra dùng** (trigger). Quyết định và thứ tự nằm ở `research_plan.md`.
- Nguyên tắc chọn: một kỹ thuật chỉ được dùng khi (a) một gate/failure mode trong plan gọi tên nó, và (b) cost implement + cost inference của nó được khai báo trước. Toolbox không phải wish list.

## Insight khung (sửa so với v1)

- Không tồn tại "best fixed architecture" cho constraint mơ hồ — vẫn đúng.
- Default philosophy giữ nguyên: **modular + temporal + hard-negative driven + calibrated + cascade**.
- **Bổ sung quan trọng:** cadence tower camera là 1 frame/phút → latency budget mỗi frame là hàng chục giây. Prior "phải dùng model nano" là dogma nhập từ video 30fps; kích thước model là **biến của cost frontier (E4)**, không phải ràng buộc cứng.
- Cascade "light always-on + heavy verifier" vẫn hợp lý, nhưng vì lý do `$` (GPU share nhiều camera), không phải vì ms.

## Menu theo failure mode

### Detector không thấy khói nhỏ/xa (G0 fail hoặc AUROC thấp)
- imgsz lớn hơn — cost tăng dự đoán được, thử đầu tiên.
- tiling/SAHI, candidate crop → reprocess high-res — hợp cadence 1/min.
- tile-classifier (SmokeyNet-style) thay bbox — đổi formulation, thuộc E5.
- multi-scale FPN, super-resolution, smoke-specific low-level features — chỉ khi các mục trên fail.

### False alarm cao từ cloud/fog/glare (đo được sau E1a)
- hard-negative mining event-level (FP persistence cao) — E3, ưu tiên nhất.
- explicit negative categories: cloud / fog / steam / dust / glare / sunset / lamp / welding / candle / stove / campfire.
- temporal persistence rule — đã nằm trong E2.
- context/scene classifier, thermal verify, VLM verify — plugin sau, xếp theo cost tăng dần.

### Muốn giảm TTD thêm khi FA đã đạt budget (G1/G2)
- input giàu hơn cho verifier: chuỗi box+conf, ROI embedding (E2 factorized).
- learned aggregator theo đúng thứ tự: logistic-window → LSTM/GRU/TCN → tiny Transformer.
- optical flow / background model: rẻ, thử làm feature phụ trước khi lên 3D CNN / video foundation model.
- bbox tracking / temporal tube linking: khi cần gắn alarm với một plume cụ thể.

### Label thiếu hoặc yếu (FIgLib không có bbox)
- weak label từ dấu offset (đang dùng) + ignore band `0..+180s`.
- pseudo-label bằng D-Fire teacher + human verify subset nhỏ (~200 frame) → gold set cho eval.
- VLM làm annotation assistant / triage (offline — nơi latency không quan trọng).
- synthetic smoke composite (E6) khi cần ground-truth onset chính xác tuyệt đối.
- semi-supervised / active learning / cross-dataset — mở khi có PYRONEAR-2025.

### Không chắc kết luận có thật (áp cho mọi experiment)
- bootstrap CI theo event; split kép event/camera; deduplication audit; cross-dataset test (FIgLib ↔ PYRONEAR-2025).
- calibration: temperature scaling trước; evidential / ensemble / MC-dropout sau (P2).
- abstention / reject zone: chỉ sau khi calibration được đo.
- conformal prediction, OOD detection — parked đến khi có use case cụ thể.

### Tối ưu chi phí inference (E4)
- frame skipping / keyframe theo cadence thật của camera.
- batch nhiều camera trên 1 GPU; asynchronous pipeline; candidate clip buffering.
- cascade on-demand (heavy verifier chỉ chạy trên candidate).
- đo `$/camera-tháng` thực trên Modal thay vì suy diễn từ FLOPs.

## Unverified literature leads (giữ lại từ v1 — chưa tự verify, đọc trước khi cam kết dùng)

v1 có trích một loạt paper theo dạng citation gãy (`[1]`, `[6]`...) không có bibliography — bản v2 đã bỏ các trích dẫn đó vì chưa kiểm chứng, nhưng bản thân ý tưởng vẫn đáng đọc khi tới đúng gate. Liệt kê lại ở đây làm reading list, KHÔNG phải fact đã verify:

- **Hard-negative flywheel (E3):** một hướng tên "SKLFS/SNSM" được cho là thiết kế quanh smoke-specific feature + negative sampling, hard negative lấy từ false detection của detector đang chạy, đánh giá trên ~1.200 camera thực tế — nếu đúng, là precedent trực tiếp cho architecture data-backflow ở E3. Cần tự tìm và đọc paper gốc để verify trước khi trích dẫn tiếp.
- **Segmentation formulation (E5):** một hướng dùng foundation-model/larger-model supervision từ bbox label để train lightweight smoke segmentation student, số liệu tự nhận đạt khoảng `63% mIoU` ở `~25 FPS` trên Jetson Orin NX qua real-world forest burn — nếu verify được, là lý do cụ thể để thử segmentation thay vì giả định suông.
- **RGB-T fusion (G3/E7):** naive early-fusion (4-channel concat) được cho là có information-interference/domain-gap issue trong detection RGB-T nói chung; dual-branch fusion mạnh hơn nhưng tốn inference hơn — khớp lý do v2 giữ 4-channel concat chỉ làm "dumb baseline" chứ không phải main method.
- **RGB-T teacher → RGB student (RQ3/E8):** một hướng tên gợi ý "SAM-TIFF"-style cho biết student thường khó nhất trên ảnh chỉ có smoke/cây, không có flame visible — nếu đúng, đây chính là hard-slice quan trọng nhất cần đo ở E8 (`no visible flame`, `heavy smoke`).
- **VLM/MLLM verifier (E9/P3):** vài nguồn cho rằng current MLLM vẫn fail đáng kể ở presence-detection dưới lớp khói dày, ủng hộ nghi ngờ trong v2 rằng specialist classifier thắng VLM về cost/reliability cho binary verification; VLM có thể có giá trị ở semantic reasoning (loại lửa, mức độ nguy hiểm) hơn là raw detection.
- **Uncertainty (P2):** một số work 2026 về wildfire smoke thử evidential uncertainty/selective prediction, cho rằng ảnh khói mật độ mơ hồ có epistemic uncertainty cao hơn và tăng khi chất lượng ảnh giảm — nếu verify được, ủng hộ việc mở uncertainty head sau khi có baseline calibration.

**Cách dùng đúng:** trước khi bắt đầu E3/E5/E7/E8/E9 hoặc P2, dành 30 phút tìm và đọc paper thật đứng sau các claim trên (search theo tên hướng/kỹ thuật + "smoke fire detection"), verify số liệu, rồi mới quyết định áp dụng — đừng copy số liệu ở trên vào báo cáo vì chưa có nguồn xác nhận.

## Menu near-field (PROPOSAL 2026-07-17, chờ duyệt track RQ5 — xem `research_plan.md` mục 10)

- Trigger: chỉ dùng khi RQ5/NG0-NG4 được user duyệt mở. Chưa áp dụng cho RQ1/FIgLib.

### Detector không thấy khói/lửa near-field (NG0/NG3)
- **Không mặc định copy P2/stride-4/tiny-object của RQ1** — near-field object size do kỹ sư thiết kế theo khoảng cách lắp (vendor: flame/smoke min ~1.1-1.6% bề rộng ảnh), khác cơ chế tiny-xa tự nhiên của tower camera. Error-slice D-Fire (bbox area/short-side) trước (NE4).
- Nếu error-slice xác nhận đúng là tiny-object trên D-Fire (ví dụ smoke sớm/mỏng trong khung hình rộng) → mới mở lại toolkit A1 hiện có.
- Motion/turbulence-based classical feature (background subtraction, optical flow, turbulence energy/wavelet) — precedent near-field cổ điển, đáng thử vì cadence liên tục (15-25fps) làm motion rẻ; **khác kết luận đã đóng của RQ1** (frame-differencing AUROC=0.5276 chỉ đúng ở cadence 60s FIgLib, không suy rộng sang đây).

### False alarm cao từ nuisance near-field (NG2, khác hẳn category cloud/fog/glare của RQ1)
- Category test theo chuẩn FM 3232/ISO 7240-29: hàn hồ quang, sunlight trực tiếp/phản xạ, vật nóng, đèn incandescent/fluorescent/halogen/LED beacon/sodium.
- Category field thực tế (vendor doc): steam/hơi nước, bụi, khí xả forklift, conveyor belt chuyển động ngang, quạt quay, phản xạ kim loại, backlight/cửa sổ.
- Đo FP rate riêng từng category — không gộp chung một số "FA/hour" như RQ1.
- Vendor thực tế xử lý bằng verification window (delay cấu hình 4-30s) trước khi báo alarm — khác N-of-M/EMA phút của RQ1, cùng ý tưởng temporal persistence nhưng thang giây.

### Response-time budget khác biệt cấu trúc (NG1)
- Mục tiêu ≤30s (FM 3232) thay vì phút — không nội suy trực tiếp từ curve TTD-phút hiện có của RQ1; cần AMOC riêng trục giây.
- FA budget ~1/50-100 detector/năm (BS 5839-1) thay vì 1/camera-ngày — chênh ~3-4 bậc độ lớn; không dùng chung operating point với RQ1.

### Deployment/privacy khác biệt cấu trúc (NG4)
- Edge/on-device là ràng buộc CỨNG (privacy GDPR + bandwidth), không phải trade-off cost mở như E4 của RQ1.
- Không thiết kế lưu trữ footage mặc định; ưu tiên kiến trúc chạy được ngay trên camera/edge-box.

## Kỹ thuật bị hạ cấp so với v1 (kèm lý do)

- **Thermal / RGB-T fusion / RGB-T→RGB distillation:** parked theo G3 — FLAME 3 là UAV cận cảnh, khác domain tower, kết luận không transfer trực tiếp sang RQ1.
- **VLM làm runtime verifier:** P3 — giữ nguyên nghi ngờ của v1 rằng specialist classifier thắng về cost/reliability; VLM chuyển sang data loop (annotation, triage).
- **Giant multi-head model:** bỏ khỏi roadmap; nếu cần chỉ tồn tại như một điểm tham chiếu đắt trên cost frontier.
- **Weather/sensor fusion:** chưa có nguồn data — không giữ trong menu active.
