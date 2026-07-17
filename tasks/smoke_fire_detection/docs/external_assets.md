# External Assets — smoke_fire_detection

- Vai trò: liệt kê file/thư mục nặng, KHÔNG track trong git (`datasets/` bị ignore toàn bộ; `*.pt`/`*.ckpt`/`*.pth`/`*.onnx`/`*.engine` bị ignore), nhưng cần có mặt để chạy lại pipeline trên máy/nền tảng mới. Không hash, không checksum, không cơ chế verify tự động — thuần liệt kê + cách lấy lại.
- Cập nhật: `2026-07-17`. Kiểm tra `du -sh` lại nếu nghi ngờ số liệu cũ.

## Dataset (`datasets/smoke_fire_detection/`, toàn bộ bị `.gitignore`)

| Path | Size | Nguồn / cách lấy lại |
|---|---|---|
| `D-Fire/` | 3.0G | **Không còn zip nguồn trong repo** (`D-Fire-train-ready.zip` đã xóa sau khi audit PASS 2026-07-16). Đây là bản train-ready cuối (GUI relabel đã chốt) — copy trực tiếp thư mục này khi chuyển máy, không có nơi khác để tải lại. Cấu trúc: `{train,valid,test}/{images,labels}` + `data.yaml`. |
| `pyro-sdis-yolo/` | 3.3G | Bản YOLO đã convert (29,537 train / 4,099 val). Lấy lại: (1) tar trên Modal volume `smoke-fire-step13-volume` path `datasets/smoke_fire_detection/pyro-sdis-yolo.tar`; (2) tải parquet gốc từ HF `pyronear/pyro-sdis` + converter `dataset.py pyro-sdis` trong git history (đã xóa 2026-07-17 cùng parquet local sau khi conversion xong việc). |
| `FIgLib/` | 35G | HPWREN public archive (attribution required, có bulk-download tool: hpwren.ucsd.edu). Archive sống, tiếp tục thêm sequence mới — tải lại nghĩa là crawl lại theo trạng thái archive hiện tại, không phải fetch 1 file cố định. |

## Checkpoint / weight (`artifacts/smoke_fire_detection/runs/`, `*.pt|*.ckpt|*.pth` bị `.gitignore`)

| Path | Size | Ghi chú |
|---|---|---|
| `dfire_yolo26n_baseline_full_vram/weights/best.pt` | 5.2M | E0 baseline YOLO26n/D-Fire. Không thể regenerate rẻ (cần train lại) — copy thủ công khi chuyển máy. Trước đây có track git làm ngoại lệ, đã untrack 2026-07-16 cho nhất quán — giờ nằm trong danh sách này như mọi weight khác. |
| `yolo26x_pyro_sdis_budget9/weights/best.pt` | 113M | YOLO26x/Pyro-SDIS, train xong 20/20 epoch (mAP50-95 best=0.4949 epoch 19). Không regenerate rẻ. |
| `rfdetr_large_pyro_sdis_gb10/checkpoint_7.ckpt` | 540M | RF-DETR-L/Pyro-SDIS, dừng ở epoch 8/20 — resume bằng `kaggle_resume_pyro_sdis_rfdetr.ipynb` (upload ckpt này + `pyro-sdis-yolo` lên Kaggle Input). |
| `rfdetr_large_pyro_sdis_gb10/checkpoint_best_ema.pth` | 135M | Best EMA weight (epoch 4, mAP50-95=0.4648). |
| `rfdetr_large_pyro_sdis_gb10/checkpoint_best_regular.pth` | 136M | Best regular weight (epoch 4, mAP50-95=0.4489). |
| `rfdetr_large_dfire_kaggle/checkpoint_19.ckpt` | 542M | RF-DETR-L/D-Fire xong 20/20 epoch (Kaggle 2×T4, 800px). Full-state epoch cuối. |
| `rfdetr_large_dfire_kaggle/checkpoint_best_regular.pth` | 136M | Best regular = epoch 18, val mAP50-95=0.49324 (best regular toàn 20 epoch). `checkpoint_best_total.pth` gốc = duplicate byte-identical của file này (verify 510/510 tensor) — đã xóa. |
| `rfdetr_large_dfire_kaggle/checkpoint_best_ema.pth` | 135M | Epoch 11 (đợt train đầu), EMA val mAP50-95=0.49614. Đã verify `torch.load` OK, 510/510 tensor, không NaN. |

Toàn bộ checkpoint trên: KHÔNG có cách "tải lại" rẻ — mất là phải train lại (giờ/ngày GPU). Backup thủ công (rsync/scp/cloud storage) khi chuyển máy hoặc trước khi dọn ổ đĩa.
