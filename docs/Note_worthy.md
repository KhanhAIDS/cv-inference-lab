Catastrophic forgetting là thật với sequential fine-tune không rehearsal


So sánh kiến trúc thực sự:
Cùng base weight family.
Cùng dataset split.
Cùng resolution, augmentation, epoch, seed.
Khác đúng một biến kiến trúc.


Pyro-SDIS không có test tách biệt — YOLO26x vừa chọn epoch vừa report mAP trên cùng 1 val, khác D-Fire (train/val/test 3 phần).
D-Fire train ở imgsz 640, YOLO26x train ở imgsz 1280 — so tuyệt đối 2 model nội bộ bị confound bởi resolution, chỉ so được qua slice/domain riêng như plan đã làm, không so trực tiếp số mAP.


Cùng giá trị seed trên 2 dataset/model khác nhau KHÔNG tạo ra "cùng randomness" hay tính so sánh gì — nó chỉ đảm bảo reproducibility từng run. Ý nghĩa fairness thật: cả hai đều là run 1-seed; plan đã quy định finalist cần ≥3 seed


Retrain trên Relabel D-Fire: Có ảnh hưởng ngầm: G0 (0.7017/0.7425), G1 (temporal cần ở FA≤1/tuần), tiling fail, breakdown silent/confused — đều là kết luận về detector CŨ. Detector retrain có FP structure khác (haze/glare) → kết luận G1/tiling không tự động áp sang


Resolution lúc train quan trọng: Model detection (YOLO-family, kể cả YOLOv8s) resize ảnh input về đúng imgsz lúc train => Có thể là một lợi thế chưa từng thấy lúc train (object to hơn quen thuộc) hay đang bị đẩy ra khỏi vùng hoạt động ổn định (feature map size lạ).





Việc cần làm:
Verify seed của quá trình train YOLO26x trên dataset pyro-sdis
Check dataset leakage với mô hình Pyronear trước khi benchmark dataset đó