# AGENTS.md

## 1. Bản chất dự án

- Dự án này là dự án phòng thí nghiệm.
- Mục đích chính: học, thử nghiệm, benchmark, tối ưu inference cho Computer Vision.
- Đây không phải production repo.
- Tool, dataset, model public / research / non-commercial đều có thể dùng nếu license cho phép.

## 2. Hai môi trường phát triển

- Code được phát triển song song trên:
  - Máy local Windows.
  - Server SSH Linux.
- Code phải ưu tiên tương thích Windows và Linux.
- Nếu không thể chạy chung:
  - Tách riêng Windows / Linux.
  - Ghi rõ phần platform-specific.
  - Không để logic platform-specific rò vào core chung.
- Không hard-code path, OS, GPU, server, username, drive letter.
- Trên Windows local, chạy Python bằng `py`, không dùng `python`.

## 3. Target Platform: Modal.com

- GPU target chính: `modal.com`.
- Dùng Modal cho tác vụ liên quan mô hình:
  - chạy thử
  - benchmark
  - triển khai
- Code Modal phải nằm trong wrapper / script / config riêng.
- Core inference logic vẫn phải portable để test trên Windows local hoặc Linux server.

## 4. Tổ chức bài toán

- Mỗi bài toán nằm trong một folder riêng dưới `tasks/`.
- Dataset của mỗi bài toán nằm trong một folder riêng dưới `datasets/`.
- Tên folder phải nói rõ bài toán hoặc dataset đang chứa.
- Không để code, docs, dataset của bài toán cụ thể nằm rải ở root.

## 5. Tối giản codebase

- Mặc định giữ codebase nhỏ nhất có thể. Tuyệt đối không giữ lại file/thư mục không phục vụ trực tiếp cho quy trình hiện tại.
- Chỉ giữ file thật sự cần cho audit, train, eval, profile, test, report.
- Xóa ngay lập tức file/folder sinh tự động, cache, scaffold rỗng, placeholder thừa.
- Không được viết code thừa thãi, tạo file thừa thãi ví dụ như `__init__.py` khi không thực sự cần thiết.
- Không tạo folder hoặc ignore rule preemptive (đón đầu) khi chưa có file thực tế hoặc nhu cầu thực sự.
- Đặt tên file/folder tường minh theo nội dung.
- Không thêm comment trong code.
- Chỉ giữ hoặc thêm comment code khi user yêu cầu rõ.

## 6. Mục tiêu tối thượng

- Xây dựng lab để đo, so sánh, tối ưu end-to-end Computer Vision inference.
- Mọi quyết định kỹ thuật phải phục vụ đo, so sánh, hoặc tối ưu inference.
- Không tối ưu mù.
- Không thêm complexity nếu chưa giúp đo, so sánh, hoặc tối ưu inference rõ hơn.

## 7. Tracking và Reporting
- Bắt buộc duy trì file `CHANGELOG.md` ở thư mục root để kiểm soát gắt gao mọi thay đổi. Nếu file đã có nội dung, phải **append** (ghi tiếp) theo trình tự thời gian (timeline), tuyệt đối không xóa nội dung cũ.
- Phải ghi rõ mốc thời gian (Timeline) cho mỗi lượt tương tác/yêu cầu.
- Ghi chú đầy đủ các command đã chạy.
- Phải liệt kê RÕ RÀNG VÀ ĐẦY ĐỦ tất cả các file bị tác động (Thêm/Sửa/Xóa). Trong đó chia làm 2 loại:
  - **Thay đổi trực tiếp:** Do AI trực tiếp tạo hoặc sửa bằng tool (vd: viết code, sửa config).
  - **Thay đổi gián tiếp:** Các file/folder tự động sinh ra/thay đổi do chạy script/command (vd: script Python tự tạo file audit, tự sinh folder split). Đây là điều BẮT BUỘC phải ghi nhận.
