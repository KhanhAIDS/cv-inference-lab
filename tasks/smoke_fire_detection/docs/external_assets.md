# External assets — smoke_fire_detection

- Mục tiêu: `git clone` sang máy mới không mất dữ liệu/model cần thiết.
- Git clone: chỉ mang file đã commit.
- `datasets/`, checkpoint/weight: bị `.gitignore`; copy riêng.
- `.venv/`: giữ nguyên máy hiện tại; không copy/commit; tạo venv mới trên máy đích.
- Không hash/checksum.

## Bắt buộc copy riêng

- `datasets/smoke_fire_detection/D-Fire/` — 3.0G; bản train-ready cuối; không còn zip nguồn; giữ nguyên.
- `datasets/smoke_fire_detection/pyro-sdis-yolo/` — 3.3G; train/val đã convert.
- `datasets/smoke_fire_detection/FIgLib/` — 35G; archive HPWREN local.
- `artifacts/smoke_fire_detection/runs/dfire_yolo26n_baseline_full_vram/weights/best.pt` — 5.2M; baseline E0.
- `artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis_budget9/weights/best.pt` — 113M; YOLO26x/Pyro-SDIS best.
- `to_be_resolved/rf-detr_pyro-sdis/checkpoint_9.zip` — 540M; full-state mới nhất; resume RF-DETR/Pyro-SDIS.
- `to_be_resolved/rf-detr_pyro-sdis/checkpoint_best_ema.zip` — 135M; best EMA hiện tại.
- `to_be_resolved/rf-detr_pyro-sdis/checkpoint_best_regular.zip` — 136M; best regular hiện tại.
- `artifacts/smoke_fire_detection/runs/rfdetr_large_dfire_kaggle/checkpoint_best_ema.pth` — 130M; RF-DETR/D-Fire legacy best; chỉ đối chiếu same-family với rerun chuẩn.
- `to_be_resolved/yolo26x_d-fire/weights/best.pt` — 113M; YOLO26x/D-Fire legacy best.
- `artifacts/smoke_fire_detection/figlib_detector_cache_yolo26x_pyro_sdis_dev.jsonl` + `.errors.json` — 28M; cache FIgLib dev đã chạy 28,360 frame; không regenerate rẻ.

## File nhỏ cần commit trước clone

- `tasks/smoke_fire_detection/kaggle_train_dfire_rfdetr.ipynb` — fresh RF-DETR/D-Fire; 800px; native recipe.
- `to_be_resolved/yolo26x_d-fire/colab_train_dfire_yolo26x.ipynb` — fresh YOLO26x/D-Fire; 800px; native recipe.
- `to_be_resolved/yolo26x_d-fire/{args.yaml,results.csv,training_protocol.json,eval_report_test.json}` — bằng chứng legacy.
- `to_be_resolved/rf-detr_pyro-sdis/{rf-detr-pyro-sdis.ipynb,metrics.csv,resume_protocol.json}` — resume active.
- `agent_context.md`, `CHANGELOG.md`, research docs, RF-DETR legacy config/protocol — commit cùng thay đổi hiện tại.

## Đã xóa 2026-07-17; không copy sang máy mới

- `artifacts/smoke_fire_detection/runs/rfdetr_large_pyro_sdis_gb10/checkpoint_7.ckpt` + 2 best weight epoch cũ — bị checkpoint 9/best mới thay thế.
- `artifacts/smoke_fire_detection/runs/rfdetr_large_dfire_kaggle/checkpoint_19.ckpt` — full-state run legacy đã hoàn tất; không resume.
- `artifacts/smoke_fire_detection/runs/rfdetr_large_dfire_kaggle/checkpoint_best_regular.pth` — kém best EMA legacy; không cần inference.
- `to_be_resolved/yolo26x_d-fire/weights/{last.pt,epoch19.pt}` — run legacy hoàn tất; `best.pt` đủ inference.
- `to_be_resolved/rf-detr_pyro-sdis/state.db` — notebook serialize trùng nội dung; không phải checkpoint resume.
- `to_be_resolved/rf-detr_pyro-sdis/data.yaml` — notebook tự tạo.
- Plot/preview/prediction cache YOLO D-Fire — đã xóa 2026-07-17.
