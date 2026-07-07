# AGENTS.md

## 1. Bản chất dự án

- Dự án này là dự án phòng thí nghiệm.
- Mục đích chính: học, thử nghiệm, benchmark, tối ưu inference cho Computer Vision.
- Đây không phải production repo.
- Tool, dataset, model public / research / non-commercial đều có thể dùng nếu license cho phép.

## 2. Hai môi trường phát triển (Local & Server)

- Code cho dự án được phát triển song song trên hai môi trường khác biệt:
  - Máy local Windows.
  - Server SSH Linux.
- Bất kỳ đoạn code nào (tiện ích, data pipeline, script) cũng phải ưu tiên tương thích chéo với cả Windows và Linux.
- Nếu một phần không thể chạy chung trên cả hai platform:
  - Phải tách riêng theo từng trường hợp Windows / Linux.
  - Phải ghi rõ phần nào là platform-specific.
  - Tuyệt đối không để logic chỉ chạy được trên một platform rò rỉ vào core chung.
- Không hard-code path, OS, GPU, server, username, drive letter.

## 3. Target Platform: Modal.com

- **Target Platform:** GPU trên `modal.com`. Đây là môi trường chuẩn để thực thi và giải quyết triệt để sự khác biệt kỹ thuật giữa Windows Local và Linux Server, đảm bảo tính portable.
- **Vai trò:** Sử dụng `modal.com` cho **tất cả** các tác vụ liên quan đến mô hình (chạy thử, benchmark, triển khai).
- Kiến trúc hệ thống triển khai model cho các bài toán sẽ bám sát hoàn toàn vào kiến trúc trên `modal.com` (chi tiết sẽ phụ thuộc vào từng bài toán cụ thể).
- Code tương tác đặc thù với Modal phải được đóng gói vào các wrapper / script / config riêng biệt để không làm kẹt logic lõi (core logic).
- Dù Modal là đích đến chính, **Core inference logic** (thuật toán thuần) vẫn phải giữ được tính portable để có thể kiểm thử trực tiếp trên Windows Local hoặc Server Linux khi cần.

## 4. Mục tiêu tối thượng

- Mục tiêu tối thượng: xây dựng một lab để đo, so sánh, và tối ưu end-to-end Computer Vision inference.
- Mọi quyết định kỹ thuật phải phục vụ mục tiêu này.
- Không tối ưu mù.
- Không thêm complexity nếu chưa giúp đo, so sánh, hoặc tối ưu inference rõ ràng hơn.
