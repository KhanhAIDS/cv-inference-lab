* **Không có 1 pipeline/model “tốt nhất” chung.**

  * Bài toán hiện tại **under-specified**: không domain, không camera cố định, không khoảng cách, không night/day, không sensor phụ.
  * Research hiện cũng phân nhánh mạnh:

    * FIgLib/SmokeyNet → RGB + thời gian cho early smoke. ([arXiv][1])
    * PYRONEAR-2025 → detector + sequential model; video giúp phát hiện sớm hơn/tăng recall. ([arXiv][2])
    * SKLFS → smoke-specific features + negative sampling vì boundary/false positive khó. ([arXiv][3])
    * Multimodal SmokeyNet → RGB + weather sensors. ([arXiv][4])
    * SAM-TIFF → RGB-T teacher → RGB student; **research mới**, không phải consensus production. ([arXiv][5])
    * MLLM/VLM hiện vẫn fail đáng kể ở presence detection dưới heavy smoke. ([arXiv][6])

* **Có 1 “meta-pipeline” tôi coi là default tốt nhất khi constraint mơ hồ:**

  * `RGB video`
  * `→ candidate detection/localization`
  * `→ temporal evidence`
  * `→ calibrated event decision`
  * `→ uncertain case: verifier khác / VLM / human`
  * **Modular cascade. Không phải 1 giant model.**

* **Toàn bộ toolbox có thể áp dụng cho bài này:**

  * **Spatial task**

    * image classification
    * ROI/patch classification
    * object detection
    * smoke segmentation
    * fire segmentation
    * multi-task / multi-head

  * **Small/faint smoke**

    * high-resolution input
    * multi-scale features
    * FPN/PAFPN
    * tiling / SAHI
    * candidate crop → reprocess high-res
    * super-resolution
    * smoke-specific low-level features

  * **Temporal**

    * N-of-M frames
    * EMA confidence
    * persistence rule
    * bbox tracking
    * temporal tube linking
    * background change
    * optical flow / motion features
    * LSTM / GRU
    * TCN
    * 3D CNN
    * temporal Transformer
    * video foundation model

  * **Data**

    * hard-negative mining
    * selective OHEM
    * active learning
    * model-error mining
    * label cleanup
    * pseudo-label
    * semi-supervised learning
    * synthetic fire/smoke
    * diffusion-generated data
    * domain adaptation
    * cross-dataset training
    * split theo event/video/camera/location
    * deduplication

  * **Model strategy**

    * specialist smoke model
    * specialist flame model
    * shared backbone + multiple heads
    * ensemble
    * mixture-of-experts
    * light model → heavy verifier
    * teacher/student distillation
    * large teacher → edge student
    * RGB-T teacher → RGB student
    * foundation model teacher → specialist student

  * **Thermal / multimodal**

    * thermal-only
    * RGB-only
    * early fusion
    * dual encoder
    * feature fusion
    * late fusion
    * RGB + thermal + temporal
    * RGB-T teacher → RGB student
    * radiometric temperature regression
    * CO / heat / smoke / weather sensor fusion

  * **Reliability**

    * threshold tuning
    * confidence calibration
    * temperature scaling
    * uncertainty estimation
    * ensemble disagreement
    * evidential learning
    * OOD detection
    * abstention / reject option
    * conformal prediction
    * risk-based decision

  * **False-positive suppression**

    * explicit cloud/fog/steam/dust negatives
    * lamp/glare/sunset negatives
    * welding/candle/stove/campfire negatives
    * context classifier
    * temporal persistence
    * specialist verifier
    * scene-aware model
    * thermal verification
    * VLM verification
    * human-in-the-loop

  * Hard-negative handling đặc biệt relevant: SKLFS/SNSM được thiết kế trực tiếp quanh confusing negative supervision của smoke detection. ([arXiv][3])

  * **VLM/MLLM**

    * fire/no-fire verifier
    * smoke vs cloud/fog/steam reasoning
    * controlled vs dangerous fire
    * identify burning object
    * severity/risk assessment
    * scene context
    * alarm explanation
    * **không nên làm primary pixel detector hiện tại**

  * DetectiumFire đã mở hướng vision-language fire understanding/risk reasoning; FlameVQA cho thấy current MLLMs vẫn có failure ở heavy-smoke presence detection. ([arXiv][7])

  * **System**

    * frame skipping
    * keyframe inference
    * ROI inference
    * detector + tracker
    * candidate clip buffering
    * lightweight always-on model
    * expensive on-demand verifier
    * asynchronous pipeline
    * multi-camera aggregation
    * human verification

  * **Evaluation**

    * frame precision/recall
    * bbox AP
    * segmentation IoU
    * event precision/recall
    * false alarms/hour
    * time-to-detection
    * alarm duration
    * calibration error
    * hard-slice metrics
    * cross-dataset metrics

* **Câu trả lời lạnh lùng nhất:**

  * `Best fixed architecture` → **không tồn tại cho constraint hiện tại**
  * `Best default philosophy` → **có**
  * **`modular + temporal + hard-negative driven + calibrated + cascade`**
  * Thermal, sensor, VLM, teacher-student, segmentation, giant Transformer → **plugin/experiment**, thêm khi ablation chứng minh có gain.
  * **Đừng chốt `RGB-T Temporal Multi-head Transformer` trước khi biết failure mode. Đó là thiết kế architecture bằng tưởng tượng, không phải optimization.**