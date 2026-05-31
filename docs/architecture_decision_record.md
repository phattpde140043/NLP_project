# Architecture Decision Record (ADR)

Tài liệu này ghi nhận các quyết định kiến trúc cốt lõi trong dự án **NLP RAG Presentation Space** nhằm đảm bảo tính nhất quán và dễ dàng bảo trì hệ thống.

---

## ADR 1: Vector Database Selection

* **Trạng thái**: Đã nghiệm thu (Completed)
* **Ngữ cảnh**: Hệ thống cần lưu trữ và thực hiện tìm kiếm tương đồng ngữ nghĩa trên các vector nhúng (text embeddings) trích xuất từ tài liệu đã nạp. Hệ thống cần chạy hoàn toàn cục bộ trên máy Mac của học viên mà không yêu cầu cấu hình máy chủ cơ sở dữ liệu phức tạp.
* **Quyết định**: Chúng tôi chọn **Qdrant (Local Persistent Directory Mode)** sử dụng thư viện `qdrant-client` chạy nhúng cục bộ.
* **Các giải pháp thay thế đã xem xét**:
  - *FAISS*: Tối ưu cực hạn về tốc độ, nhưng thiếu khả năng lọc payload siêu dữ liệu nâng cao trực tiếp. Quản lý ánh xạ vector sang metadata thủ công làm tăng độ phức tạp của code.
  - *ChromaDB*: Khá mượt và dễ, nhưng tính năng lọc payload và quản lý phân đoạn cấu hình của Qdrant chuyên nghiệp, tường minh hơn cho việc nâng cấp lên Qdrant Cloud.
* **Phân tích Đánh đổi (Trade-offs)**:
  - *Điểm lợi*: Không cần cài đặt server (chạy trực tiếp dưới thư mục đĩa cục bộ `.qdrant_data`), hỗ trợ lọc payload mạnh mẽ cô lập theo Workspace UUID, đường đi chuyển đổi lên Qdrant Cloud sản xuất rất rõ ràng.
  - *Điểm hại*: Thêm dependency `qdrant-client`, nhưng cài đặt rất đơn giản qua `pip`.

---

## ADR 2: Metadata and Ingestion State Storage (Zero-Database Architecture)

* **Trạng thái**: Đã nghiệm thu (Completed - Hiệu chỉnh theo chỉ thị tinh gọn)
* **Ngữ cảnh**: Chúng tôi cần quản lý đăng ký tệp tin trong từng Workspace, trạng thái nạp tệp (`COMPLETED`, `PROCESSING`, `FAILED`), dung lượng chunk và mã băm chống trùng lặp.
* **Quyết định**: Chúng tôi quyết định sử dụng **Kiến trúc Zero-Database**, lưu trữ siêu dữ liệu bằng các tệp cấu trúc **JSON cục bộ** cô lập hoàn toàn dưới phân vùng thư mục `storage/workspaces/<uuid>/metadata.json`.
* **Các giải pháp thay thế đã xem xét**:
  - *SQLite Database*: Tốt cho quan hệ nhiều-nhiều, nhưng làm tăng độ phức tạp hệ thống không cần thiết trong phạm vi cá nhân (yêu cầu cài đặt driver, quản lý các tệp tin đĩa, migration).
  - *PostgreSQL*: Quá cồng kềnh, vi phạm ràng buộc chạy offline đơn giản của dự án.
* **Phân tích Đánh đổi (Trade-offs)**:
  - *Điểm lợi*: Giảm thiểu tuyệt đối độ phức tạp (Zero-DB setup), dữ liệu cô lập tuyệt đối theo cấu trúc thư mục UUID Workspace (xóa một thư mục là xóa sạch dữ liệu liên quan không để lại rác), dễ đọc hiểu và gỡ lỗi trực tiếp bằng mắt thường.
  - *Điểm hại*: Không hỗ trợ các phép toán JOIN quan hệ đắt đỏ, nhưng hoàn hảo cho mô hình Workspace độc lập của RAG.

---

## ADR 3: Embedding Interface Abstraction (Embedding Factory)

* **Trạng thái**: Đã nghiệm thu (Completed)
* **Ngữ cảnh**: Hệ thống cần hỗ trợ linh hoạt giữa các mô hình nhúng chất lượng cao trên đám mây (OpenAI) và các mô hình chạy cục bộ hoàn toàn miễn phí (MiniLM-L6-v2) để hỗ trợ học viên phát triển offline.
* **Quyết định**: Triển khai lớp máy chủ máy dịch vụ **`EmbeddingFactory`** cung cấp adapter động dựa trên cấu hình tệp `.env`.
* **Các giải pháp thay thế đã xem xét**:
  - *Cấu hình cứng OpenAI*: Ràng buộc chi phí và bắt buộc phải kết nối Internet.
  - *Cấu hình cứng SentenceTransformers*: Không tối ưu cho các bài toán đa ngôn ngữ nâng cao trên môi trường production.
* **Phân tích Đánh đổi (Trade-offs)**:
  - *Điểm lợi*: Cực kỳ linh hoạt, lớp xử lý trung tâm (Application) hoàn toàn không cần biết vector nhúng được tạo ra như thế nào.
  - *Điểm hại*: Tăng một chút boilerplate code ban đầu.
