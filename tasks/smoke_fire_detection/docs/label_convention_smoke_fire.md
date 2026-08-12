# Quy ước đánh nhãn khói/lửa + chiến lược audit

## 1. Nguyên tắc gốc

- Mục tiêu quy ước: giảm **bất định nhãn**, không phải vẽ bbox đẹp.
- Metric chốt của lab: per-class event recall, worst-class recall, FA/camera-hour. Hình học bbox chỉ là phụ.
- Suy ra thứ tự lỗi đắt: **sai class > thiếu object > ảnh âm dán nhãn dương > bbox lệch hình học**.
- Mỗi object phải quyết được trong ~2 giây. Quá 2 giây → xem mục 4 (mơ hồ), không ngồi cân nhắc.
- Chi phí bất đối xứng đã chốt: **FN > FP**. Khi lưỡng lự giữa "vẽ" và "không vẽ" một vùng có khả năng là khói/lửa → **vẽ**.

## 2. Bằng chứng từ nguồn công khai

- **FASDD** (nguồn lớn nhất trong pool: 95,314 ảnh, 82,666 flame + 57,742 smoke instance) là dataset duy nhất công bố rule cụ thể:
  - "A flame or a smoke object that is partially occluded but obviously connected is regarded as a separate object"
  - "Multiple tiny objects clustered together are considered to be a particular object"
  - "Flame or smoke with significantly different colors are not considered to be the same object"
  - "Objects smaller than 10×10 pixels and without apparent flame or smoke characteristics are ignored"
  - "Reflections of flame and smoke on smooth surfaces such as water shall be ignored if they do not match the shape and texture features of the corresponding original objects"
  - "Objects smaller than 10×10 pixels with prominent shape and texture features shall be not omitted"
  - "Images smaller than 48×48 and difficult to be interpreted shall be deleted"
  - QC 3 vòng: cross-check theo cặp → panel inspector → domain expert xử lý conflict → script tự sửa box rỗng/trùng/vượt biên.
- **PYRONEAR-2025**: chủ động **không dùng mAP**, lý do công bố nguyên văn: contour khói *"can be subjective"*. Dùng precision/recall/F1. Mỗi ảnh 5 annotator, đo Krippendorff α trên **có/không có khói** (không phải trên bbox), batch 150 ảnh/lượt.
- **FIgLib relabel gần đây**: SAM2 + YOLO11x sinh mask → convert sang tight bbox → human validate để loại mây. Máy vẽ, người duyệt.
- **COCO / Open Images**: COCO `iscrowd=1` (vùng crowd, **bỏ khỏi tính metric**), Open Images `IsGroupOf` (>5 instance chồng lấn + chạm nhau → 1 box group). Tiền lệ cho "n đám khói nhỏ".
- **Literature khói**: khói phân tán từ đặc → loãng nên không tồn tại bbox duy nhất đúng; cả cục bộ và toàn thể đều "là khói". Hệ quả: detector sinh nhiều box cho 1 đám, NMS không dập được, metric bbox-level tụt giả tạo.

Chỗ FASDD SAI và tôi đề xuất sửa: rule *"significantly different colors are not the same object"* nếu áp máy móc sẽ **tách 1 cột khói duy nhất** thành nhiều box (khói đen ở gốc → xám ở giữa → trắng ở đỉnh là chuyện bình thường của một đám cháy). Xem R3 đã sửa.

## 3. Cây quyết định — dùng cái này khi audit

```
Vùng đang xét có phải khói/lửa thật của một sự kiện cháy?
├─ KHÔNG (đèn, mặt trời, TV, hơi nước, mây, phản chiếu, nến/bếp có kiểm soát)
│     → không vẽ.
├─ CÓ → nó là lửa (phát sáng) hay khói (hạt che)?
│        vùng phát sáng cam/vàng/trắng = fire | vùng che mờ = smoke | chồng nhau thì vẽ CẢ HAI
│     → Có bao nhiêu object? Áp R1-R3.
│     → Vẽ bbox theo R4-R10.
└─ KHÔNG QUYẾT ĐƯỢC trong 2 giây → nghiêng về "là khói/lửa" → vẽ (FN > FP)
      Chỉ khi cả ảnh không quyết được gì (không chỉ 1 object) → phím E, loại ảnh.
```

Không có nhánh "mơ hồ, để riêng" cho từng object — xem mục 4bis vì sao.

## 4. Rule cơ bản

Canonical class file nhãn train: `smoke=0`, `fire=1`. Chỉ 2 class.

### Đếm object

- **R1 — object = 1 vùng liền mạch.** Một cột khói / một ngọn lửa liên tục = **1 bbox**. Bị vật thể che ngang mà mắt vẫn thấy rõ nó là một khối liền → vẫn 1 bbox. Không tách 1 đám thành n box.
- **R2 — cụm nhỏ = 1 bbox chung.** Nhiều đốm lửa / vệt khói nhỏ tụ thành cụm → 1 bbox bao cụm. Ngưỡng: khoảng cách giữa 2 đốm ≤ 1× chiều dài đốm lớn hơn → cùng cụm.
- **R3 — tách chỉ khi khác NGUỒN PHÁT, không phải khác màu.** 2 điểm cháy tách biệt về không gian, 2 cột khói có 2 gốc riêng → 2 bbox. Cùng một cột đổi màu theo độ cao (đen → xám → trắng) → **1 bbox**. Test dứt điểm: che một vùng đi, vùng còn lại có **gốc/base riêng** không? Có → tách. Không → 1 box.
- **R3b — trần số box.** Nếu phải vẽ >6 bbox khói/lửa cho 1 ảnh, gần như chắc chắn đang tách quá nhỏ → gộp lại. Ngoại lệ thật: nhiều điểm cháy rời rạc thực sự (đốt đồng, nhiều bếp).

### Vẽ bbox

- **R4 — ngưỡng kích thước.** <10×10 px và không rõ đặc trưng → bỏ. <10×10 px nhưng đặc trưng rõ (đốm lửa sáng rõ) → **phải giữ**.
- **R5 — phản chiếu bỏ.** Lửa/khói phản chiếu trên nước, kính, sàn bóng, kim loại → không vẽ, trừ khi vùng phản chiếu tự nó có shape/texture đúng như object gốc.
- **R6 — biên khói bám phần nhìn thấy được.** Vẽ tới nơi khói còn phân biệt được với nền. Phần đã tan hòa vào trời/nền → không bao. Không cố bao "vùng khói đúng về vật lý".
- **R7 — fire và smoke độc lập, ĐƯỢC overlap.** Ảnh có cả hai phải có cả hai nhãn. bbox fire nằm trong bbox smoke là bình thường, không phải lỗi. Không có class `fire_and_smoke`.
- **R8 — fill ratio ≥ 0.35.** `fill = diện tích object thật / diện tích bbox`. Nếu fill < 0.35 (bbox toàn khoảng trống), xét tách theo R3 — nhưng chỉ tách nếu mỗi phần sau khi tách có fill ≥ 0.5 và tổng số box không vượt R3b. Nếu không thỏa → giữ 1 bbox rộng, chấp nhận.
- **R9 — bbox không được ≥90% diện tích ảnh** trừ khi khói/lửa thật phủ ≥60% ảnh. Đây là lỗi thật đã thấy trong `annotated_fire_smoke_2025` (2 bbox trùng nhau 0.989×0.991 = gần cả ảnh). Full-image box biến ảnh thành "cả ảnh là khói" → model học báo động trên mọi ảnh mù mờ → FA tăng.
- **R10 — modal, không amodal.** Chỉ bao phần trong khung ảnh. Không suy diễn phần bị crop. Clip về biên ảnh.
- **R11 — nhãn rỗng là quyết định thật.** 0 bbox = đã xác nhận không có khói/lửa (hard negative). Không dùng "rỗng" thay cho "chưa xem".
- **R12 — không bbox lồng nhau cùng class.** 1 bbox smoke nằm hoàn toàn trong 1 bbox smoke khác → gộp thành 1. Khác class thì được (R7).
- **R13 — nhãn cấp ảnh không tự thành bbox.** DeepQuest chỉ có nhãn cấp ảnh; bbox teacher chỉ thành nhãn khi bạn bấm duyệt.

Khi R1–R3 xung đột: ưu tiên **ít box hơn**. Gộp không mất event recall; tách sai tạo FP nhãn và dạy model chia nhỏ object.

## 4bis. Đã bỏ class `ignore` — chốt theo dữ liệu thực tế của bạn

Bản đầu của doc này đề xuất thêm class `ignore` cho vùng mơ hồ. Bạn đã relabel D-Fire thật và cho biết: gần như không gặp trường hợp không biết là khói/lửa hay không, và với lượng data hiện có, vài ảnh mơ hồ không ảnh hưởng gì. Chốt theo đó — bỏ `ignore` hoàn toàn.

- Tool giờ chỉ có 2 class: `smoke=0`, `fire=1`. Không còn nhãn thứ 3, không còn file `.audited.ignore.txt`.
- Cây quyết định ở mục 3 không còn nhánh "vẽ ignore cho 1 vùng". Mọi object phải chốt vẽ hoặc không vẽ.
- `E` (excluded) vẫn còn nhưng chỉ dùng khi **cả ảnh** không dùng được (hỏng, không phải cảnh cháy, không quyết được gì) — không dùng cho từng object riêng lẻ trong một ảnh mà phần còn lại vẫn ổn.
- Với 20 edge case ở mục 5, hầu hết đã có câu trả lời dứt điểm là "vẽ" hoặc "không vẽ" — bảng dưới tổng hợp lại để không ai phải nhớ ngoại lệ nào cần "để riêng":

| Case | Quyết định |
|---|---|
| Khói mờ có gốc + hướng | Vẽ |
| Khói mờ không gốc, phủ đều (haze/sương khí quyển) | Không vẽ |
| Khí thải ống khói công nghiệp | Không vẽ (test: miệng ống cố định + đều + màu đồng nhất) |
| Phản chiếu nước/kính | Không vẽ |
| Nhìn qua kính trong | Vẽ |
| Nến/bếp/đèn/TV/mặt trời | Không vẽ |
| Lửa/khói ban đêm quá mờ | Không vẽ, không dùng ảnh làm positive cho lớp đó |
| Còn lưỡng lự sau tất cả câu hỏi trên | Vẽ (FN > FP), không loại ảnh |

Lý do bỏ `ignore` không chỉ vì bạn ít gặp mơ hồ — còn vì lý do kỹ thuật: Ultralytics YOLO và RF-DETR không hỗ trợ ignore-region/loss-masking sẵn. Ghi `class 2` vào file nhãn YOLO sẽ khiến model học nó thành một loại object thứ ba, hỏng cả class fire/smoke. Nên dù muốn dùng `ignore`, nó cũng không thể nằm cùng file nhãn train — chỉ có thể tách riêng. Với data lớn và mơ hồ hiếm như bạn nói, tách riêng là công đoạn thừa. Bỏ hẳn là đúng.

Ghi chú: canonical taxonomy chốt 2026-07-31 (`agent_context.md`) có dòng "ambiguous = ignore" ở tầng product/taxonomy (khác tool này) — dòng đó nói về cách phân loại ảnh input, chưa nói tool label phải có class `ignore`. Không mâu thuẫn với quyết định ở đây; tool label vẫn chỉ 2 class.

## 5. Edge case — trả lời cụ thể

### E1 — Lửa lẫn khói trong cùng ảnh
- 2 bbox độc lập, chồng nhau tùy ý. bbox smoke thường bao cả bbox fire. Đúng, không phải lỗi.
- Không có class gộp. Ảnh `bothFireAndSmoke` của FASDD vẫn phải có đủ 2 nhãn.

### E2 — Vùng chuyển tiếp nửa lửa nửa khói (giữa ngọn lửa và cột khói)
- Câu hỏi này chỉ khó nếu tưởng phải phân vùng độc quyền. **Bbox được overlap** → không cần chọn.
- Quy tắc: **bbox fire** bao tới nơi còn thấy phát sáng cam/vàng/trắng. **bbox smoke** bao từ đâu bắt đầu thấy hạt che, kể cả phần chồng lên lửa.
- Vùng chuyển tiếp thuộc **cả hai** bbox. Không cắt đôi, không vẽ box thứ ba.

### E3 — Khói/lửa sau hàng cây, song sắt, lưới, hàng rào
- Occluder mảnh và lặp **không chia object**. R1 áp dụng: **1 bbox bao toàn bộ**, bao luôn cả song sắt/cành cây ở giữa.
- Ngưỡng: occluder che ≤50% diện tích object → 1 bbox, không do dự.
- Che >50% và các mảnh còn lại rời rạc: vẫn ưu tiên 1 bbox (R3 chỉ tách khi có gốc riêng — song sắt không tạo gốc riêng).

### E4 — Khói mờ như sương vì ảnh phân giải kém
- Dùng **test bối cảnh**, không phải test pixel:
  - Có **gốc (base)** không? Khói thật xuất phát từ một điểm/đường trên mặt đất.
  - Có **hướng** không? Khói thật có hướng phun/tỏa.
  - Có nguồn cháy, vệt sẫm, điểm nóng ở gốc không?
- Có gốc + hướng → **smoke, vẽ**.
- Lớp mờ phủ đều toàn ảnh, không gốc, không hướng → **haze/sương/khói bụi khí quyển, không vẽ**.
- Phải zoom >200% mới thấy → không phải target near-field → không vẽ, và ảnh này không được dùng làm smoke-positive.
- Còn lưỡng lự → vẽ (FN > FP), không loại ảnh vì 1 vùng mờ.

### E5 — Lửa hình chữ L, vòng cung, đường lửa dài
- Bbox là hình chữ nhật axis-aligned nên luôn có khoảng trống. Chấp nhận.
- Áp R8: tính `fill`. fill ≥ 0.35 → **1 bbox bao trọn**, xong.
- fill < 0.35 → thử tách theo 2 nhánh. Chỉ tách nếu mỗi nhánh đạt fill ≥ 0.5 **và** mỗi nhánh có gốc riêng (R3). Chữ L từ một điểm cháy lan theo 2 hướng = **1 gốc → 1 bbox**, dù fill thấp.
- Đường lửa dài ngang (fire line): 1 bbox dài, fill thường cao vì lửa chiếm hết chiều ngang → 1 bbox.

### E6 — Khói bị lửa chiếu sáng đỏ/cam
- Đó là **smoke**, không phải fire. Test: vùng đó có tự phát sáng không, hay chỉ đang bị chiếu? Chỉ bị chiếu → smoke.
- Đây là nguồn lỗi `wrong_class` phổ biến nhất trong pool.

### E7 — Lửa nhìn qua cửa kính / cửa sổ
- Thấy **trực tiếp qua** kính trong suốt → vẽ bình thường (kính không phải occluder ngữ nghĩa).
- **Phản chiếu trên** mặt kính → không vẽ (R5).
- Phân biệt: phản chiếu thường bị lật, mờ, méo, và có object gốc ở đâu đó khác trong ảnh.

### E8 — Lửa trên TV, màn hình, poster, biển hiệu, tranh
- **Không vẽ.** Không phải sự kiện cháy — đây là loại FA đắt cho near-field indoor nếu model học nhầm.

### E9 — Nến, bật lửa, bếp gas, lò, đèn dầu, đốt rác có kiểm soát
- **Không vẽ.** Ảnh giữ nguyên với 0 bbox là hard negative có giá trị: model học nến ≠ báo cháy.
- Lưu ý mâu thuẫn cần bạn quyết nếu sản phẩm sau này muốn bắt cả lửa nhỏ trong nhà — nhưng theo định nghĩa hiện tại (fixed-camera cảnh báo cháy), nến không phải target.

### E10 — Đèn đường, đèn pha, mặt trời, chớp sáng, phản chiếu kim loại
- **Không vẽ.**

### E11 — Hơi nước, khí thải ống xả/ống khói nhà máy, bụi, mây thấp, sương mù
- **Không vẽ.** Khí thải ống khói công nghiệp là bẫy lớn nhất: có gốc, có hướng, giống khói cháy.
- Phân biệt: khí thải công nghiệp phát ra từ **miệng ống cố định**, đều, liên tục, thường trắng/xám nhạt đồng nhất. Khói cháy có gốc trên mặt đất/vật cháy, không đều, thường lẫn màu.
- Còn lưỡng lự → vẽ theo R nghiêng-về-vẽ; chỉ loại ảnh (`E`) nếu cả ảnh không dùng được.

### E12 — Khói đen và khói trắng trong cùng một cột
- **1 bbox.** R3 đã sửa: đổi màu theo độ cao không tách object. Chỉ tách khi có 2 gốc.

### E13 — Khói/lửa bị crop ở mép ảnh
- Clip bbox về biên (R10). Không đoán phần ngoài khung.
- Nếu phần thấy được <10×10 px và không rõ đặc trưng → bỏ (R4).
- 248 bbox trong pool hiện tại đang vượt biên (`box_crosses_image_boundary`) — sửa bằng cách kéo về biên, không xóa.

### E14 — Bbox lồng nhau / trùng nhau hoàn toàn
- Cùng class → gộp 1 (R12). Pool hiện có 2 `duplicate_box`, 3 `non_positive_box_size`.

### E15 — Ảnh ban đêm
- Lửa = điểm sáng nhỏ nhưng đặc trưng rõ → **vẽ** (R4 giữ).
- Khói ban đêm gần như không thấy → **không vẽ**, và ảnh không được dùng làm smoke-positive. Nếu ảnh có fire rõ, vẫn dùng làm fire-positive.

### E16 — Nhiều điểm cháy rời rạc trên một đường lửa
- Liên tục hoặc khoảng cách ≤1× chiều dài đốm → 1 bbox (R2).
- Rời rạc rõ, mỗi điểm có gốc riêng, tổng ≤6 box → tách (R3, R3b).

### E17 — Ảnh augment (flip/rotate/crop) của cùng ảnh gốc
- `annotated_fire_smoke_2025` có 2,214 nhóm tên lặp, `indoor_fire_smoke_2025` có 375 nhóm. Nhãn phải **nhất quán trong cùng nhóm**.
- Nếu bạn sửa 1 ảnh trong nhóm, các bản augment còn lại đang sai theo cùng cách. Ghi chú lại; đừng tự động lan quyết định (crop có thể làm object biến mất thật).

### E18 — Khói che gần hết ảnh
- Nếu khói thật phủ ≥60% ảnh → bbox lớn hợp lệ (R9).
- Nếu không → vẽ chặt hơn. Full-image box là lỗi, không phải "an toàn".

### E19 — Tàn lửa, than hồng, khói từ tàn
- Than hồng phát sáng, đặc trưng rõ → `fire`.
- Khói mỏng từ tàn có gốc rõ → `smoke`.
- Chỉ vệt đen/xám không phát sáng, không tỏa → không vẽ.

### E20 — Lửa rất nhỏ là nguồn của cột khói lớn
- 2 bbox: `fire` nhỏ (giữ dù <10×10 px nếu đặc trưng rõ, R4) + `smoke` lớn.
- Đây là case quan trọng nhất cho sản phẩm: bỏ fire nhỏ = mất fire recall, mà worst-class recall là metric chốt.

## 6. Chiến lược audit

- Queue gộp: **30,425 ảnh** (29,378 error + 1,047 pseudo).
- Bạn nói audit hết là cần thiết. Chấp nhận, nhưng phải hạ giá mỗi ảnh xuống, không phải bỏ ảnh:
  - Luồng 1 phím/ảnh: nhìn → `Enter` (nhãn dataset đúng, chốt thành GT) hoặc `W` (bbox model đúng hơn, chốt luôn) hoặc sửa tay rồi `Enter`.
  - Ở 4 s/ảnh = **~34 giờ**. Ở 10 s/ảnh = 85 giờ. Khoảng cách này nằm ở chỗ có phải sửa tay hay không.
  - Sắp xếp `ít bbox trước` để làm ảnh dễ trước, giữ nhịp.
  - Batch ≤150 ảnh/lượt rồi nghỉ (cách PYRONEAR làm). Audit mệt là lúc nhãn bắt đầu sai.
- **Thứ tự làm, theo giá trị với metric:**

| issue | số ảnh | ưu tiên | lý do |
|---|---|---|---|
| `static_dataset_issue` | 241 | 1 | box vượt biên/trùng/size≤0, sửa máy móc, nhanh nhất |
| `possible_wrong_class_bbox` | 47 | 1 | smoke↔fire đảo, rẻ và đắt giá |
| `possible_image_class_conflict` + `teacher_disagrees_with_image_class` | 299 | 1 | sai class cấp ảnh |
| `full_source_semantic_review` | 100 | 1 | quyết giữ/loại cả source FireAndSmoke v1 |
| `possible_missing_bbox` | 7,383 | 2 | thiếu object → FN, loại lỗi đắt nhất |
| `possible_pseudo_label` | 1,047 | 3 | mở thêm 1,047 positive có bbox từ DeepQuest |
| `possible_extra_bbox` | 11,804 | 4 | nhãn dương thừa; phần lớn do teacher conf 0.05 nên nhiều là báo động sai của chính teacher |
| `possible_geometry_mismatch` | 9,504 | 5 | thuần hình học — đúng cái metric không đo. Làm cuối, hoặc `Enter` chấp nhận nhanh |

- **Audit eval/held-out split trước train split.** Label noise ở eval làm **sai số đo**; ở train chỉ làm giảm chất lượng. Ưu tiên phần sẽ dùng làm eval.
- **Self-agreement 5%.** Audit lại mù 5% ảnh đã chốt sau vài ngày. Failure mode cụ thể bị chặn: nếu bạn không nhất quán với chính mình, mọi so sánh source/model dựa trên nhãn này đều không kết luận được, và không có cách nào khác phát hiện.

### Nút "lấy bbox model làm ground truth" — cảnh báo bắt buộc

- `W` = lấy bbox model + chốt GT + sang ảnh sau, 1 phím.
- Nhãn sinh từ model **không được dùng để đánh giá chính model đó**. Teacher hiện tại là `rfdetr_large_dfire` — đúng model đang là winner. GT lấy từ dự đoán của nó sẽ tự động ưu ái nó trong mọi so sánh sau này (circular evaluation).
- Tool ghi `label_source` cho từng quyết định: `dataset` (giữ nguyên nhãn gốc, không sửa gì) / `teacher` (bấm `W`) / `human` (có sửa tay). Manifest export đếm theo 3 loại, `eval_safe = label_source != teacher`.
- Nguyên tắc: **eval split chỉ dùng `dataset` + `human`.** `teacher` chỉ được vào train split. Vẫn phải xem ảnh trước khi bấm `W` — `W` tiết kiệm thao tác chuột, không phải để duyệt mù.

## 7. Công cụ

`label_audit_tool.py` + `label_audit_tool.html` (stdlib Python, không cần pip install, Windows + Linux). Chỉ giữ chức năng cần thiết: xem nhãn gốc, xem bbox model, vẽ/chọn/sửa/xóa bbox, xác nhận, loại ảnh.

```
python tasks/smoke_fire_detection/label_audit_tool.py import --bundle <zip hoặc thư mục queue>
python tasks/smoke_fire_detection/label_audit_tool.py serve --port 8760
python tasks/smoke_fire_detection/label_audit_tool.py status
python tasks/smoke_fire_detection/label_audit_tool.py export --link-images
```

- Windows local dùng `py`; server Linux dùng `.venv/bin/python`.
- Duyệt từ laptop qua LAN: `http://<LAN-IP>:8760/` hoặc `ssh -L 8760:127.0.0.1:8760 <server>`. Ảnh stream từ server.
- Label offline hoàn toàn (không cần server) — xem mục 7bis.

### Tương tác bbox

- Màu viền cố định, ngoài dải màu tự nhiên của khói/lửa: smoke = vàng-chanh `#C6FF00`, fire = magenta `#FF2BD6`. Lửa đỏ không còn trùng viền đỏ, khói xám không còn trùng nền trời. Viền có lớp casing đảo màu theo nền để luôn tương phản.
- Click vào vùng chồng lấn → chọn đúng bbox **nhỏ nhất** chứa điểm click (bbox con luôn ưu tiên hơn bbox mẹ). Kéo bbox đang chọn chỉ di chuyển đúng nó, không ảnh hưởng bbox khác đang trùm lên. `Shift` + kéo để buộc vẽ bbox mới ngay trong vùng đang có bbox chọn sẵn.
- Bbox model (teacher) vẽ nét đứt kèm confidence, mặc định lọc ≥0.30 (`[` `]` đổi ngưỡng), phím `T` ẩn/hiện.

### Đã audit = ground truth, không hỏi lại

- `decisions.jsonl` trong `artifacts/smoke_fire_detection/label_audit/` là nguồn sự thật duy nhất, append-only, key = đường dẫn ảnh tương đối trong `datasets/smoke_fire_detection/` (bền qua mọi lần model chạy lại).
- Ảnh `confirmed`/`excluded` bị loại khỏi queue mặc định. Chạy `import` lần sau tự gắn `carried_over_status` → không audit lại.
- App tự lưu ảnh đang làm dở vào `localStorage` trình duyệt; mở lại app nhảy thẳng về đúng ảnh đó.
- `export` ghi `datasets/smoke_fire_detection/near_field_curated_ground_truth/` với marker trong tên file `<stem>.audited.txt`; `--link-images` thêm symlink `<stem>.audited.<ext>`. Raw dataset không bao giờ bị rename/sửa.
- Status: `confirmed` (là GT), `excluded` (loại khỏi train/eval), `draft` (tự lưu khi rời ảnh giữa lúc sửa mà chưa bấm Enter/W/E).

## 7bis. Label offline trên máy local

Không cần server, không cần mạng, không cần Python cài thêm package.

```
python tasks/smoke_fire_detection/label_audit_tool.py pack \
  --output <thư mục pack> --zip \
  --issue static_dataset_issue --issue possible_wrong_class_bbox --issue possible_missing_bbox
```

- `pack` copy đúng ảnh cần audit + tool + launcher vào 1 thư mục tự chứa, nén thành `.zip`. Không cần cờ `--issue`/`--source` thì lấy toàn bộ ảnh còn `pending`/`draft` (theo `--limit`, mặc định không giới hạn — nên luôn lọc theo issue để pack vừa dùng).
- Mang `.zip` về máy local, giải nén, chạy `START_WINDOWS.bat` (Windows) hoặc `START_LINUX.sh` (Linux/mac) — mở `http://127.0.0.1:8760/` là dùng được, không đụng server, không đụng `datasets/` gốc.
- Label xong, copy `decisions.jsonl` trong thư mục pack về server, chạy:
  ```
  python tasks/smoke_fire_detection/label_audit_tool.py merge --decisions <đường dẫn>/decisions.jsonl
  ```
  Lệnh `merge` từ chối nếu `task_id` lạ (bảo vệ khỏi trộn nhầm workspace) và bỏ qua bản trùng y hệt.
- Nếu bạn đã audit vài ảnh trên server trước khi pack, quyết định đó theo luôn trong pack (`carried_over_status`) — không phải audit lại.

## 8. Nguồn

- FASDD annotation rule + QC: [ESSD preprint essd-2023-73](https://essd.copernicus.org/preprints/essd-2023-73/essd-2023-73.pdf)
- PYRONEAR-2025 protocol, lý do bỏ mAP: [arXiv 2402.05349](https://arxiv.org/html/2402.05349v2)
- FIgLib / SmokeyNet: [arXiv 2112.08598](https://arxiv.org/pdf/2112.08598)
- AI For Mankind wildfire smoke dataset: [github.com/aiformankind/wildfire-smoke-dataset](https://github.com/aiformankind/wildfire-smoke-dataset)
- Pyro-SDIS: [huggingface.co/datasets/pyronear/pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis)
- Open Images IsGroupOf: [github.com/openimages/dataset](https://github.com/openimages/dataset/blob/main/READMEV1.md)
- Bbox khói không duy nhất, NMS không dập được: [arXiv 2311.10116](https://arxiv.org/pdf/2311.10116)
- Khó khăn annotate khói (biên mờ, trong suốt, nhỏ): [AusSmoke / MultiNatSmoke](https://arxiv.org/pdf/2604.23542)
