# Annotation Quality: Vấn đề gốc không phải “label xấu”, mà là target không được định nghĩa ổn định

* **Kết luận chính**

  * Annotation không đơn thuần là thao tác vẽ box, mask, chọn class, chấm điểm hoặc nhập text.
  * Annotation là quá trình **chuyển một hiện tượng thực tế thành target có cấu trúc để model học và metric đánh giá**.
  * Vấn đề gốc thường không phải `annotator làm ẩu`.
  * Vấn đề gốc thường là:

    * **không rõ cần annotate cái gì**.
    * **không rõ một unit/instance/event/span/object là gì**.
    * **không rõ boundary nằm ở đâu**.
    * **không rõ trường hợp mơ hồ xử lý thế nào**.
    * **không rõ representation hiện tại có phù hợp task hay không**.
  * Khi target definition không ổn định:

    * cùng một pattern thực tế.
    * nhiều label hợp lệ theo các convention khác nhau.
    * model nhận supervision mâu thuẫn.
    * metric chủ yếu đo khả năng bắt chước convention của ground truth.
  * Vì vậy:

    * **annotation quality trước hết là quality của target specification**.
    * sau đó mới là quality của annotator execution.

# 1. Tách 5 tầng vấn đề

* **Tầng 1 — Task definition**

  * Model cuối cùng cần làm gì?
  * Ví dụ:

    * phát hiện có/không.
    * định vị.
    * đếm.
    * phân loại.
    * phân đoạn.
    * tracking.
    * ranking.
    * retrieval.
    * extraction.
    * alert/event detection.
  * Lỗi phổ biến:

    * deployment cần `event recall`.
    * dataset annotate object-level bbox.
    * benchmark dùng mAP.
  * Hậu quả:

    * tối ưu tốt benchmark.
    * không chắc tạo giá trị deployment.
  * Câu hỏi bắt buộc:

    * **Quyết định nào sẽ được đưa ra từ prediction?**

* **Tầng 2 — Ontology / target definition**

  * Xác định:

    * entity nào tồn tại.
    * class nào tồn tại.
    * quan hệ giữa class.
    * unit annotation là gì.
  * Unit có thể là:

    * object.
    * instance.
    * region.
    * event.
    * frame.
    * sequence.
    * text span.
    * intent.
    * attribute.
    * pair/relation.
  * Lỗi gốc:

    * con người nhìn cùng dữ liệu nhưng đang giải hai bài toán ontology khác nhau.
  * Ví dụ tổng quát:

    * `1 vùng lớn` ↔ `N vùng nhỏ`.
    * một hành vi liên tục ↔ nhiều event.
    * một text span dài ↔ nhiều span ngắn.
    * class cha ↔ class con.
  * Đây không nhất thiết là label sai.
  * Có thể là **instance definition chưa tồn tại hoặc chưa đủ rõ**.

* **Tầng 3 — Annotation policy**

  * Sau khi ontology rõ, cần rule cho case cụ thể.
  * Policy cần trả lời:

    * merge hay split?
    * boundary tới đâu?
    * minimum visibility/confidence là bao nhiêu?
    * partial object xử lý thế nào?
    * occlusion?
    * truncation/edge?
    * overlap?
    * multi-label?
    * ambiguous class?
    * uncertain case?
    * ignore region?
    * negative example?
  * Không có policy:

    * annotator tự tạo convention cá nhân.
    * annotation style phụ thuộc người, batch, vendor, thời gian.
  * Nguy hiểm nhất:

    * **cùng input pattern nhưng output convention thay đổi**.

* **Tầng 4 — Annotation execution**

  * Policy đúng nhưng annotator có thể thực thi sai.
  * Các lỗi:

    * missing label.
    * extra label.
    * wrong class.
    * inaccurate boundary.
    * duplicate annotation.
    * attribute sai.
    * typo/transcription error.
  * Đây mới là nhóm lỗi thường được gọi là `label noise`.
  * Không nên dùng `label noise` để mô tả toàn bộ vấn đề annotation.
  * Lý do:

    * ontology ambiguity không thể sửa chỉ bằng thuê annotator tốt hơn.
    * policy ambiguity không thể sửa chỉ bằng QC nhiều hơn.

* **Tầng 5 — Data/pipeline/evaluation integrity**

  * Không phải annotation semantics nhưng thường bị nhầm chung.
  * Ví dụ:

    * coordinate transform sai.
    * resize/crop không update label.
    * class mapping lệch.
    * duplicate data.
    * stale annotation version.
    * source-specific schema.
    * train/test leakage.
    * adjacent frame leakage.
    * subject/site/event leakage.
  * Fix:

    * pipeline validation.
    * lineage/versioning.
    * group-aware split.
  * Không nên gọi đây là `annotator inconsistency`.

# 2. Taxonomy lỗi annotation và hậu quả học máy

* **Merge/split inconsistency**

  * Cùng cấu trúc:

    * `1 annotation lớn` ↔ `N annotation nhỏ`.
  * Xuất hiện trong:

    * object detection.
    * segmentation.
    * tracking.
    * event annotation.
    * text span extraction.
    * clustering/entity resolution.
  * Hậu quả:

    * target count không ổn định.
    * matching không ổn định.
    * model không học được unit convention cố định.
    * duplicate prediction tăng.
    * post-processing nhạy.
  * Với detection:

    * AP matching.
    * NMS.
    * assignment.
  * Với NLP:

    * exact span match.
    * entity count.
  * Câu hỏi audit:

    * **Một entity bắt đầu/kết thúc khi nào?**
    * **Điều kiện nào khiến hai phần được xem là cùng một instance?**

* **Extent / boundary inconsistency**

  * Cùng target:

    * annotate core rõ nhất.
    * annotate toàn bộ extent.
  * Ví dụ:

    * box chặt ↔ box rộng.
    * mask core ↔ mask cả vùng mờ.
    * text phrase tối thiểu ↔ full noun phrase.
    * event onset rõ ↔ include transition.
  * Hậu quả:

    * localization target nhiễu.
    * model học nhiều background/context không mong muốn.
    * model bỏ phần yếu/mờ/khó.
    * IoU hoặc exact-match giảm dù semantic detection đúng.
  * Câu hỏi audit:

    * **Boundary là geometric, semantic hay operational boundary?**

* **Missing annotation**

  * Target có thật theo policy nhưng không được annotate.
  * Hậu quả phụ thuộc training setup.
  * Có thể:

    * positive bị coi là negative/background.
    * loss phạt prediction hợp lý.
    * prevalence estimate sai.
    * prediction đúng bị tính false positive khi evaluation.
  * Không nên khẳng định mọi missing label đều trở thành background supervision.
  * Cần kiểm tra:

    * sampling.
    * ignore mask.
    * loss.
    * negative mining.
    * partial-label support.
  * Câu hỏi audit:

    * **Dataset giả định exhaustive annotation hay partial annotation?**

* **Extra / hallucinated annotation**

  * Annotate target không tồn tại theo policy.
  * Hậu quả:

    * model học artifact.
    * tăng false positive.
    * class boundary méo.
  * Đặc biệt nguy hiểm:

    * rare class.
    * heavily imbalanced task.

* **Class inconsistency**

  * Cùng pattern:

    * class A ↔ class B.
  * Nguyên nhân:

    * class overlap.
    * hierarchy không rõ.
    * transition state.
    * insufficient context.
    * annotator suy đoán hidden state.
  * Hậu quả:

    * class supervision mâu thuẫn.
    * calibration xấu.
    * confusion tăng.
  * Overlap tự nó không phải lỗi.
  * **Policy xử lý overlap không ổn định mới là lỗi.**
  * Câu hỏi audit:

    * mutually exclusive hay multi-label?
    * class theo appearance hay cause?
    * class theo current state hay eventual outcome?

* **Temporal inconsistency**

  * Adjacent observations:

    * label nhảy.
    * xuất hiện/biến mất vô lý.
    * class đổi qua lại.
    * instance ID đổi.
    * split/merge không có rule.
  * Có hai khả năng:

    * hiện tượng thật thay đổi.
    * annotation instability.
  * Không được mặc định mọi temporal discontinuity là annotation error.
  * Audit cần so với expected dynamics của domain.
  * Hậu quả:

    * tracking identity instability.
    * temporal model học flicker.
    * event onset/duration target nhiễu.

* **Source / annotator / batch inconsistency**

  * Convention phụ thuộc:

    * annotator.
    * vendor.
    * source.
    * device.
    * site.
    * dataset version.
    * annotation batch.
  * Hậu quả:

    * model học source shortcut.
    * performance nội bộ tốt.
    * domain transfer kém.
  * Audit:

    * error rate theo group.
    * label distribution theo group.
    * annotation geometry/statistics theo group.
  * Câu hỏi:

    * **Có đoán được annotator/source chỉ từ pattern label không?**
  * Nếu có:

    * annotation process có thể đang để lại signature.

# 3. Điểm quan trọng: disagreement không đồng nghĩa annotator sai

* Hai annotator disagree có thể do:

  * một người sai.
  * cả hai sai.
  * guideline mơ hồ.
  * task intrinsically ambiguous.
  * input thiếu context.
  * representation ép hiện tượng liên tục thành discrete label.
* Vì vậy:

  * inter-annotator agreement thấp chỉ là symptom.
  * không tự động chứng minh `annotator quality thấp`.
* Cần phân biệt:

  * **avoidable disagreement**.
  * **irreducible ambiguity**.
* Avoidable:

  * guideline thiếu rule.
  * training annotator kém.
  * UI lỗi.
* Irreducible:

  * boundary thực tế không rõ.
  * multiple interpretations hợp lý.
  * latent information không quan sát được.
* Với irreducible ambiguity:

  * forcing single hard label có thể tạo ground truth giả.
* Giải pháp có thể là:

  * uncertain label.
  * soft label.
  * probability distribution.
  * multiple acceptable annotations.
  * ignore region.
  * hierarchical label.
  * adjudication with context.
  * đổi task representation.

# 4. Representation phải phù hợp bản chất target

* Không có annotation format tốt tuyệt đối.
* Có:

  * format phù hợp hoặc không phù hợp với task.
* Ví dụ:

  * bbox:

    * rẻ.
    * dễ annotate.
    * localization thô.
  * polygon/mask:

    * geometry tốt hơn.
    * chi phí cao.
    * boundary ambiguity vẫn tồn tại.
  * point:

    * phù hợp counting/rough localization.
  * tile/grid:

    * phù hợp region presence.
  * image-level label:

    * phù hợp classification/alert.
  * sequence/event label:

    * phù hợp temporal decision.
  * ranking/pairwise label:

    * phù hợp preference/relevance.
  * text span:

    * phù hợp extraction.
  * document-level label:

    * phù hợp global decision.
* Sai lầm:

  * thấy localization không tốt.
  * mặc định chuyển bbox → segmentation.
* Segmentation không giải quyết:

  * ontology ambiguity.
  * merge/split ambiguity.
  * unclear extent.
* Thậm chí:

  * representation càng chi tiết.
  * annotation variance có thể càng lộ rõ.
* Quy tắc:

  * **Chỉ tăng độ chi tiết annotation khi độ chi tiết đó tạo giá trị cho deployment.**

# 5. Ground truth không phải “sự thật tuyệt đối”

* Ground truth thường là:

  * observation.
  * interpretation.
  * policy.
  * representation.
  * annotation execution.
* Có thể viết:

  * `GT = f(real phenomenon, observable evidence, ontology, policy, representation, annotator)`
* Vì vậy:

  * GT là **operational reference target**.
  * không nhất thiết là geometric/semantic truth tuyệt đối.
* Điều này đặc biệt rõ với:

  * fuzzy boundaries.
  * subjective labels.
  * future-dependent outcomes.
  * latent intent.
  * ambiguous text.
  * overlapping phenomena.
* Hệ quả:

  * model disagreement với GT không luôn đồng nghĩa model sai về deployment objective.
  * nhưng cũng không được dùng ambiguity để bao biện model kém.
* Cần clean evaluation protocol để phân biệt.

# 6. Metric đo agreement với evaluation target, không tự động đo business/deployment value

* Metric luôn gắn với target representation.
* Ví dụ:

  * mAP đo detection matching theo class, IoU và matching convention.
  * exact match đo khớp chuỗi/span chính xác.
  * accuracy đo khớp discrete label.
* Metric không tự động đo:

  * time-to-detect.
  * event recall.
  * downstream utility.
  * human review cost.
  * false alarms theo thời gian.
  * safety risk.
* Ví dụ tổng quát:

  * GT dùng `1 unit`.
  * model trả `2 unit chi tiết`.
  * metric có thể phạt.
  * GT dùng `2 unit`.
  * model trả `1 unit bao phủ`.
  * metric có thể phạt.
* Không nên gọi metric là `bất công`.
* Cách diễn đạt chính xác:

  * **metric có thể không task-aligned**.
  * hoặc:
  * **evaluation target không đại diện đầy đủ deployment objective**.
* Quy tắc:

  * metric chính phải nối được với quyết định deployment.
  * representation metric dùng như diagnostic metric khi cần.

# 7. Audit annotation: không xem ảnh ngẫu nhiên rồi “cảm giác label xấu”

* Audit cần hypothesis-driven.
* Trước audit:

  * viết danh sách failure modes nghi ngờ.
* Ví dụ:

  * merge/split.
  * boundary.
  * missing.
  * extra.
  * class ambiguity.
  * overlap.
  * occlusion.
  * truncation.
  * low visibility.
  * hard negatives.
  * temporal inconsistency.
  * source/batch inconsistency.
* Sampling:

  * stratified.
  * không chỉ random.
* Strata nên dựa trên:

  * class.
  * rarity.
  * size.
  * visibility.
  * confidence.
  * source.
  * annotator.
  * time.
  * environment.
  * model error bucket.
* Random sample:

  * đo prevalence tổng quan.
* Targeted sample:

  * tìm failure mode.
* Hai loại không thay thế nhau.

# 8. Rubric phải được viết trước khi đo annotator

* Không nên:

  * đưa hai annotator label.
  * thấy disagreement.
  * kết luận dataset noisy.
* Trước double-label:

  * định nghĩa rubric.
* Rubric tối thiểu:

  * target/unit là gì?
  * positive condition?
  * negative condition?
  * merge/split rule?
  * boundary rule?
  * minimum evidence/visibility?
  * uncertain case?
  * overlap?
  * occlusion?
  * truncation/edge?
  * class precedence?
  * exhaustive hay partial annotation?
  * ignore case?
  * context nào annotator được xem?
* Mỗi rule cần:

  * positive examples.
  * negative examples.
  * hard examples.
  * counterexamples.
* Guideline chỉ có prose:

  * thường chưa đủ.
* Example bank:

  * quan trọng ngang rule text.

# 9. Double annotation phải đo disagreement theo loại

* Một con số agreement tổng:

  * quá ít thông tin.
* Cần tách:

  * presence disagreement.
  * count disagreement.
  * boundary disagreement.
  * class disagreement.
  * attribute disagreement.
  * temporal disagreement.
* Với detection/segmentation:

  * matched instance rate.
  * unmatched annotation rate.
  * count difference.
  * matched-box IoU.
  * mask IoU/boundary distance.
  * class confusion.
* Với classification:

  * confusion matrix.
  * per-class agreement.
  * Cohen's kappa hoặc weighted kappa khi phù hợp.
* Với text span:

  * exact span agreement.
  * token overlap.
  * boundary disagreement.
  * label disagreement.
* Không nên fetishize một agreement score.
* Mục tiêu:

  * xác định **disagreement đến từ rule nào**.

# 10. Adjudication không chỉ để tạo “gold label”

* Adjudication có hai output:

  * corrected annotation.
  * corrected guideline.
* Nếu cùng disagreement xuất hiện nhiều lần:

  * không nên adjudicate từng sample mãi.
  * phải sửa policy.
* Quy trình:

  * detect disagreement cluster.
  * identify ambiguity.
  * update rule.
  * add example.
  * relabel affected subset nếu cần.
* Annotation QA tốt:

  * feedback loop vào specification.
* Annotation QA kém:

  * chỉ sửa từng label.

# 11. Clean holdout cần độc lập với noisy training data

* Nếu evaluation GT có cùng ambiguity/noise:

  * model ranking có thể không đáng tin.
* Nên có:

  * clean holdout.
  * stricter annotation.
  * double-label.
  * adjudication.
  * version lock.
* Clean holdout không nhất thiết lớn.
* Cần:

  * representative.
  * high-quality.
  * leakage-safe.
* Split unit phải theo dependency structure thực tế.
* Có thể cần split theo:

  * subject.
  * patient.
  * site.
  * camera.
  * event.
  * sequence.
  * document.
  * conversation.
  * user.
* Không mặc định random image/row split.
* Nguyên tắc:

  * **samples có shared latent context không nên nằm hai phía train/test nếu deployment gặp context mới.**

# 12. Model-assisted audit: dùng model để tìm lỗi, không dùng model làm oracle

* Model hữu ích để rank:

  * high-loss samples.
  * persistent disagreement.
  * high-confidence model/GT conflict.
  * rare geometry.
  * distribution outliers.
  * temporal flicker.
* Nhưng:

  * model có bias riêng.
  * model có thể học đúng annotation convention và vẫn sai deployment.
* Không nên:

  * `model disagree → GT sai`.
* Nên:

  * `model disagree → candidate for review`.
* Đặc biệt hữu ích:

  * large dataset.
  * rare annotation errors.
  * iterative data cleaning.

# 13. Phân biệt label noise và concept ambiguity

* **Label noise**

  * có correct label rõ theo policy.
  * annotation khác correct label.
* **Concept ambiguity**

  * nhiều label hợp lý vì target chưa đủ xác định hoặc phenomenon mơ hồ.
* Fix khác nhau:

  * label noise:

    * QC.
    * relabel.
    * annotator training.
    * consensus.
  * concept ambiguity:

    * sửa ontology.
    * sửa policy.
    * thêm uncertain state.
    * đổi representation.
    * đổi metric.
* Dùng noise-robust loss để fix ontology ambiguity:

  * thường sai tầng vấn đề.

# 14. Checklist thực dụng trước khi train model

* **Task**

  * deployment decision là gì?
  * error nào đắt nhất?
  * latency/time dimension có quan trọng?
* **Target**

  * annotation unit là gì?
  * target observable trực tiếp hay cần suy đoán?
* **Policy**

  * merge/split rõ?
  * extent rõ?
  * ambiguous cases rõ?
  * exhaustive/partial rõ?
* **Representation**

  * cần box/mask/span thật không?
  * coarse label có đủ không?
* **Audit**

  * đã audit stratified chưa?
  * đã xem rare/hard cases chưa?
* **Agreement**

  * đã double-label subset chưa?
  * disagreement được phân loại chưa?
* **Pipeline**

  * transform label test?
  * class mapping test?
  * duplicate/leakage test?
* **Evaluation**

  * clean holdout?
  * split đúng dependency unit?
  * metric nối với deployment objective?
* Nếu nhiều câu trả lời là `không rõ`:

  * chưa nên dành phần lớn effort cho model architecture.

# 15. Protocol audit tối thiểu

* Audit `200–500` samples như điểm khởi đầu thực dụng.
* Không xem đây là magic number.
* Sample theo strata quan trọng.
* Bước 1:

  * xác định task và deployment KPI.
* Bước 2:

  * viết ontology + rubric draft.
* Bước 3:

  * audit sample.
* Bước 4:

  * tag từng issue theo taxonomy.
* Bước 5:

  * double-label subset.
* Bước 6:

  * đo disagreement theo loại.
* Bước 7:

  * adjudicate recurring ambiguity.
* Bước 8:

  * sửa guideline.
* Bước 9:

  * tạo clean holdout.
* Bước 10:

  * train baseline đơn giản.
* Bước 11:

  * dùng error analysis để quyết định:

    * clean data.
    * change representation.
    * change metric.
    * change model.
* Không mặc định:

  * error cao → cần model mạnh hơn.

# 16. Cách đọc model error đúng hơn

* Prediction khác GT có ít nhất 5 giả thuyết:

  * model sai.
  * GT execution sai.
  * guideline mơ hồ.
  * representation không phù hợp.
  * metric/matching không task-aligned.
* Error analysis nên kiểm tra từng giả thuyết.
* Không nên mặc định giả thuyết đầu tiên.
* Đây là lý do:

  * xem qualitative errors vẫn cần thiết.
  * nhưng phải xem theo taxonomy và sample design.
  * không xem gallery lỗi ngẫu nhiên.

# 17. Thứ tự ưu tiên sửa

* **Ưu tiên 1 — Task alignment**

  * đang giải đúng bài toán chưa?
* **Ưu tiên 2 — Ontology**

  * target unit/class rõ chưa?
* **Ưu tiên 3 — Policy**

  * hard cases có rule chưa?
* **Ưu tiên 4 — Evaluation integrity**

  * split và metric đúng chưa?
* **Ưu tiên 5 — Execution quality**

  * missing/wrong/boundary error bao nhiêu?
* **Ưu tiên 6 — Representation**

  * cần annotation chi tiết hơn hay đơn giản hơn?
* **Ưu tiên 7 — Model**

  * architecture/loss/post-processing.
* Sai lầm phổ biến:

  * nhảy thẳng tới ưu tiên 7.

# 18. Nguyên tắc cuối

* **Model học target được annotate, không học “ý định thật” của team.**
* Ý định không viết thành ontology/policy/metric:

  * model không biết.
* Ground truth inconsistent:

  * model bị yêu cầu fit một hàm target không ổn định.
* Annotation càng chi tiết:

  * không tự động càng tốt.
* Agreement cao:

  * không tự động task đúng.
* mAP/F1/accuracy cao:

  * không tự động deployment tốt.
* Dataset lớn:

  * không bù được target definition sai.
* Bước quan trọng nhất trước model optimization:

  * **định nghĩa chính xác model phải học convention nào, convention đó phục vụ quyết định nào, và metric nào chứng minh được giá trị đó.**