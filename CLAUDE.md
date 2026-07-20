# CLAUDE.md

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
- Agent không được giả định máy hiện tại là môi trường nào. Trước khi chạy lệnh phụ thuộc platform, agent phải tự phát hiện bằng command (vd `uname -a`, `nvidia-smi -L`) thay vì đoán hoặc hard-code vào tài liệu.
- Trên server Linux, project dùng virtualenv riêng tại `.venv` (root repo). Agent PHẢI kích hoạt venv này (`source .venv/bin/activate`, hoặc gọi trực tiếp `.venv/bin/python`/`.venv/bin/pip`) trước khi chạy bất kỳ lệnh `python`/`pip` nào liên quan tới project trên server đó. Không cài package vào Python hệ thống.

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
- Ngoại lệ: folder `docs/` ở root là tài liệu cá nhân của user, không thuộc quy tắc tối giản (mục 5). Agent không tự đọc, sửa, di chuyển, hay xóa bất kỳ file nào trong `docs/` (root) trừ khi user yêu cầu rõ ràng trong lượt đó.

## 5. Tối giản codebase

- Mặc định giữ codebase nhỏ nhất có thể. Tuyệt đối không giữ lại file/thư mục không phục vụ trực tiếp cho quy trình hiện tại.
- Chỉ giữ file thật sự cần cho audit, train, eval, profile, test, report.
- Xóa ngay lập tức file/folder sinh tự động, cache, scaffold rỗng, placeholder thừa.
- Không được viết code thừa thãi, tạo file thừa thãi ví dụ như `__init__.py` khi không thực sự cần thiết.
- Không tạo folder hoặc ignore rule preemptive (đón đầu) khi chưa có file thực tế hoặc nhu cầu thực sự.
- Không thêm layer quy trình mới (hash lock, checkpoint manifest, cost ledger, snapshot verify, gate phụ) khi chưa có failure mode thực tế mà layer đó ngăn chặn. Mặc định dùng cơ chế sẵn có: git commit, seed cố định, artifact JSON. Muốn thêm layer → phải nêu failure mode cụ thể và được user duyệt.
- Agent tuyệt đối không tự thêm hash, checksum, fingerprint cho dataset, split, checkpoint, artifact hoặc file list. Chỉ được thêm khi user yêu cầu rõ trong lượt hiện tại. Lý do chung chung như integrity, reproducibility, audit không phải phê duyệt.
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
- Mọi mốc thời gian (Timeline) ghi bằng múi giờ **GMT+7**, bất kể múi giờ hệ thống của máy đang chạy (server có thể chạy UTC).
- Ghi chú đầy đủ các command đã chạy.
- Phải liệt kê RÕ RÀNG VÀ ĐẦY ĐỦ tất cả các file bị tác động (Thêm/Sửa/Xóa). Trong đó chia làm 2 loại:
  - **Thay đổi trực tiếp:** Do AI trực tiếp tạo hoặc sửa bằng tool (vd: viết code, sửa config).
  - **Thay đổi gián tiếp:** Các file/folder tự động sinh ra/thay đổi do chạy script/command (vd: script Python tự tạo file audit, tự sinh folder split). Đây là điều BẮT BUỘC phải ghi nhận.
- User có thể tự xóa trắng nội dung `CHANGELOG.md` bất kỳ lúc nào để tiết kiệm token đọc, sau khi tự review việc agent đã làm. Đây là hành vi bình thường, có chủ đích — nếu thấy file rỗng hoặc thiếu lịch sử cũ, agent KHÔNG được tự ý khôi phục từ git hay coi đó là lỗi/vi phạm cần báo cáo.

## 8. Ngôn ngữ giao tiếp

- Agent luôn trả lời user bằng tiếng Việt (user đọc tiếng Việt nhanh hơn).
- Khi thực hiện web search, ưu tiên query bằng tiếng Anh để có nhiều kết quả hơn, rồi tổng hợp/trả lời lại bằng tiếng Việt.

## 9. Agent context và tối giản việc đọc file

- Trước khi đọc file khác trong repo, agent phải đọc `agent_context.md` (root) trước.
- Chỉ đọc thêm file cụ thể/chi tiết (code, docs, dataset file...) khi `agent_context.md` không cung cấp đủ thông tin cần thiết cho tác vụ đang làm.
- Mục đích: `agent_context.md` phải giúp giảm số lượng file agent cần đọc xuống mức tối thiểu, tránh quét/đọc tràn lan toàn bộ codebase mỗi lượt.
- Nếu phát hiện `agent_context.md` thiếu hoặc sai thông tin so với file gốc, phải cập nhật lại `agent_context.md` sau khi xác minh.

## 10. Văn phong agent và file context

- Mọi output chat của agent phải dùng tiếng Việt, 100% gạch đầu dòng, không câu chào, không câu mào đầu, không câu kết luận xã giao.
- Văn phong mặc định: keyword, câu ngắn, trực diện, thực dụng, phản biện; bỏ filler word, từ nối, câu hoa mỹ.
- Quy tắc này áp dụng cho mọi file agent tạo/sửa, trừ khi format kỹ thuật của file bắt buộc khác.
- `agent_context.md` chỉ dành cho agent đọc, không phải tài liệu cho người dùng.
- `agent_context.md` phải viết tối giản kiểu caveman: bullet ngắn, dữ kiện sống, schema, blocker, next step.
- `agent_context.md` không chứa log dài, transcript command, lịch sử đã lỗi thời, diễn giải câu đầy đủ nếu keyword đủ hiểu.
- Log command, timeline, file thay đổi phải ghi ở `CHANGELOG.md`, không nhồi vào `agent_context.md`.
- Phân tầng tài liệu: file static (vd `tasks/*/docs/*.md`) ít thay đổi; `agent_context.md` và `CHANGELOG.md` là file volatile (bị rút gọn/xóa trắng thường xuyên). File static TUYỆT ĐỐI không dẫn chiếu file volatile để chứa chi tiết — số liệu chốt phải ghi inline trong file static, chi tiết exploratory trỏ artifact JSON hoặc git history.