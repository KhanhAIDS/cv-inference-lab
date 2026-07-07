* **P0 — Đổi research question**

  * Không còn:

    * `detect smoke + fire bbox`
  * Đổi thành:

    * **“Early fire detection from video với false alarm thấp: temporal và thermal cải thiện RGB bao nhiêu; thermal knowledge có distill sang RGB-only được bao nhiêu?”**
  * Đây là câu hỏi:

    * có baseline
    * có ablation
    * có optimization
    * có research novelty
    * có production relevance

* **Experiment 0 — giữ D-Fire YOLO baseline**

  * `RGB single-frame`
  * `YOLO26n`
  * Không bỏ.
  * Vai trò:

    * control baseline
    * học pipeline
    * profiling target
  * **Không coi nó là final model.**

* **Experiment 1 — ưu tiên số 1: temporal confirmation**

  * Dataset:

    * PyroNear-2025 video
    * FIgLib
  * Pipeline:

    * `frame detector`
    * `↓ low threshold`
    * `candidate ROI`
    * `↓`
    * `temporal verifier`
    * `↓`
    * `event confidence`
  * Test 4 verifier:

    * `N-of-M rule`
    * `EMA confidence`
    * `LSTM`
    * `small temporal Transformer`
  * **N-of-M và EMA bắt buộc.**
  * Lý do:

    * LSTM thắng simple rule hay không?
    * Transformer đáng latency hay không?
  * PyroNear baseline hiện dùng detector bbox → ResNet representations → LSTM binary classification và đã cho recall/TTD gain rõ ràng. ([arXiv][6])

* **Metric Experiment 1:**

  * primary:

    * `false alarm events / hour`
    * `event precision`
    * `time-to-detection`
  * secondary:

    * event recall
    * frame precision
    * frame recall
    * bbox metric
  * **Đây là thay đổi metric quan trọng nhất.**

* **Experiment 2 — hard-negative flywheel**

  * Chạy Experiment 1 trên negative videos.
  * Thu:

    * cloud
    * fog
    * steam
    * dust
    * glare
    * sunset
    * lamp
    * welding
    * candle
    * stove
    * campfire
  * Lấy:

    * high-confidence FP
    * long-persistence FP
  * Human review.
  * Add vào hard-negative set.
  * Retrain.
  * Repeat.
  * Đây gần với data-backflow pattern trong SKLFS-WildFire: difficult negatives được lấy từ false detections của existing detectors; system/paper đã đánh giá trên application data và 1,200 cameras. ([arXiv][2])
  * **Tôi xếp cái này ngang temporal về importance.**

* **Experiment 3 — task formulation cho smoke**

  * Compare:

    * `A: bbox detector`
    * `B: image/ROI smoke presence classifier`
    * `C: smoke segmentation`
  * Same data split.
  * Same event metrics.
  * Không assume segmentation thắng.
  * Một precedent đáng thử: Pesonen et al. dùng larger/foundation-model supervision từ bbox labels để train lightweight smoke segmentation student; đạt `63.3% mIoU` và khoảng `25 FPS` trên Jetson Orin NX tại real-world forest burns. ([arXiv][9])
  * **Research question:**

    * segmentation mask có giúp temporal verifier distinguish smoke motion tốt hơn bbox không?

* **Experiment 4 — thermal ablation. Chưa distill.**

  * Dataset:

    * FLAME 3 paired RGB + radiometric thermal
  * Train cùng task/split:

    * `RGB-only`
    * `thermal-only`
    * `RGB-T late fusion`
    * `RGB-T dual encoder + feature fusion`
  * Không `RGBT = 4-channel concat` làm main method.
  * Có thể giữ 4-channel concat làm **dumb baseline**.
  * General RGB-T detection research đã chỉ ra naive early fusion có information interference/domain-gap problem; dual-branch thường mạnh nhưng tốn inference hơn. ([arXiv][10])
  * FlameFinder cũng xuất phát từ đúng complementary failure:

    * RGB miss smoke-obscured flame
    * thermal-only confuse non-flame hotspots
    * paired modalities giúp học discriminative flame representation. ([arXiv][11])

* **Experiment 4 phải slice metric:**

  * `visible flame`
  * `smoke-obscured flame`
  * `no visible flame`
  * `night`
  * `smoke-only`
  * `hot non-fire`
  * `small fire`
  * **Không chỉ aggregate mAP.**
  * Gain của thermal có thể bị aggregate metric che mất.

* **Experiment 5 — đây mới là RGB-T teacher → RGB student**

  * Chỉ làm sau Experiment 4.
  * Điều kiện:

    * `RGB-T teacher thực sự thắng RGB-only`
  * Teacher:

    * best RGB-T fusion model
  * Student:

    * best RGB temporal model
  * Distill:

    * event logits
    * intermediate features
    * smoke/fire masks nếu Experiment 3 chứng minh mask useful
  * Optional:

    * thermal/temperature auxiliary regression
  * **Không bắt temperature head chỉ vì SAM-TIFF có.**
  * Temperature chỉ đáng giữ nếu:

    * giúp fire discrimination
    * hoặc project thật sự cần temperature estimate

* **Experiment 5 metric quan trọng nhất:**

  * `Teacher`
  * `RGB baseline`
  * `Distilled RGB student`
  * So:

    * aggregate
    * hard slices
    * cross-dataset
  * Đặc biệt:

    * `no visible flame`
    * `heavy smoke`
  * Tôi dự đoán:

    * student lấy lại **một phần** teacher gain
    * không lấy lại hết
    * gap lớn nhất ở information truly invisible in RGB
  * Đây là inference trực tiếp từ SAM-TIFF: student gặp khó trên ảnh chỉ smoke/trees, không visible fire. ([arXiv][1])

* **Experiment 6 — uncertainty + abstention**

  * Không cần tạo `uncertainty head` ngay.
  * Bắt đầu:

    * confidence calibration
    * event score
    * disagreement:

      * RGB vs thermal
      * detector vs temporal verifier
    * reject zone
  * Ví dụ:

    * `<0.2` → negative
    * `>0.8` → alarm candidate
    * `0.2–0.8` → uncertain
  * Threshold chỉ ví dụ; tune trên validation.
  * Sau đó mới test:

    * evidential head
    * ensemble
    * MC dropout
  * 2026 wildfire smoke work đã thử evidential uncertainty và selective prediction; kết quả cho thấy ambiguous smoke density có epistemic uncertainty cao hơn và uncertainty tăng khi image quality giảm. ([arXiv][12])
  * **Nhưng tôi xếp sau temporal + hard negatives.**

* **Experiment 7 — VLM verifier**

  * Chỉ input:

    * uncertain candidate clip
    * crop + context frames
  * Prompt/task:

    * `smoke/cloud/fog/steam/dust?`
    * `controlled/uncontrolled fire?`
    * `what is burning?`
    * `risk level?`
  * Compare:

    * no verifier
    * specialist classifier
    * VLM
  * Metric:

    * FP removed
    * true events incorrectly suppressed
    * latency
    * GPU cost
  * **Tôi nghi specialist classifier sẽ thắng VLM về cost/reliability cho binary verification.**
  * VLM có thể thắng ở:

    * semantic context
    * stove vs house fire
    * candle vs curtain ignition
  * DetectiumFire được thiết kế chính để hỗ trợ burning-object/environment/severity reasoning. ([arXiv][7])

* **Architecture production-like tôi đánh giá hợp lý nhất để Bro hướng tới:**

  * `RGB stream`
  * `↓`
  * `lightweight per-frame candidate detector`
  * `↓`
  * `temporal event verifier`
  * `↓`
  * `hard-negative-trained confidence`
  * `↓`
  * `optional thermal verifier/fusion`
  * `↓`
  * `calibrated decision + abstain`
  * `↓`
  * `optional VLM/human verification`
  * **Không phải:**

    * `giant RGB-T temporal 7-head VLM monster chạy mọi frame`

* **Priority tôi chốt:**

  * `P0 MUST` → event-level benchmark + TTD + false alarms/hour
  * `P0 MUST` → RGB temporal
  * `P0 MUST` → hard-negative mining
  * `P0 MUST` → event/camera/location split + cross-dataset
  * `P1 HIGH` → bbox vs presence classifier vs segmentation
  * `P1 HIGH` → RGB vs thermal vs RGB-T fusion
  * `P1 RESEARCH HIGH` → RGB-T teacher → RGB student
  * `P2` → calibration/uncertainty/abstention
  * `P3` → VLM verifier
  * `P4` → giant multi-head model

* **Plan ngắn nhất tôi khuyên Bro làm ngay:**

  * `1.` Giữ D-Fire YOLO baseline.
  * `2.` Chuyển sang PyroNear-2025 video.
  * `3.` Implement `N-of-M + EMA + LSTM temporal verifier`.
  * `4.` Benchmark `false alarms/hour + TTD`.
  * `5.` Hard-negative mining loop.
  * `6.` Sau đó mới mở FLAME 3 và làm `RGB / thermal / RGB-T` ablation.
  * `7.` **Chỉ khi RGB-T thắng rõ → distill RGB-T teacher sang RGB temporal student.**
  * **Đây là hướng tôi đánh giá mạnh nhất cho project của Bro: `temporal-first → multimodal ablation → privileged-modality distillation`, không phải nhảy thẳng vào teacher-student.**