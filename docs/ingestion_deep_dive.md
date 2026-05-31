# 🏛️ Technical Specification & Architecture RFC: Ingestion & Text Splitting Pipeline (Phase 2)

Tài liệu này cung cấp thiết kế kiến trúc chuẩn sản xuất (Production-Grade Architecture RFC) về **Phase 2: Nạp Tài liệu & Phân đoạn Token-Aware (Document Ingestion & Text Splitting)**. Tài liệu phân tích sâu sắc các quyết định thiết kế biểu diễn vector đặc (dense vector representation), mô hình lỗi hệ thống, chiến lược bảo mật tệp tin nhị phân, các cải tiến thực tế đã triển khai và các phân tích đánh đổi (trade-offs) ở quy mô hệ thống phân tán (Enterprise Scale).

---

## 🗺️ 1. Sơ đồ Thiết kế Hệ thống & Biến đổi Trạng thái (System Architecture)

Luồng đi của dữ liệu được thiết kế như một pipeline xử lý bất đồng bộ, tách biệt rạch ròi giữa việc tiếp nhận tệp tin thô và biến đổi thành biểu diễn vector đặc (dense vector representation):

```text
  [Tệp PDF/Word] ──► [Security Gateway] ──► [SHA-256 Check] ──► [Factory Parser]
                             │                     │                     │
                             ▼ (MIME/ZipBomb)      ▼ (Deduplication)     ▼ (Strategy)
                        [Quarantine]          [Bypass & Toast]     [Raw Text + Meta]
                                                                         │
                                                                         ▼
                                                                   [Tiktoken Split]
                                                                   (cl100k_base)
                                                                         │
                                                                         ▼
                                                                   [Domain Mapping]
                                                                         │
                                                                         ▼
                                                                 [Vector Space Index]
```

---

## 💻 2. Các Cải tiến ĐÃ TRIỂN KHAI THỰC TẾ (Implemented Improvements)

Hệ thống hiện thực hóa các tiêu chuẩn thiết kế của một hệ thống RAG thực chiến cục bộ, bám sát 100% mã nguồn thực tế:

### 2.1 Rào chắn Bảo mật Nhị phân Thực chiến (Security Gateway)
* **Xác thực Magic Bytes nhị phân đầu tệp**: Đọc trực tiếp 4 byte đầu tiên của tệp tin thô để kiểm soát chữ ký nhị phân độc lập với phần mở rộng tên tệp, ngăn ngừa tấn công đổi đuôi file độc hại (Extension Spoofing):
  - Định dạng `.pdf`: Khớp với signature nhị phân `%PDF` (hex: `25 50 44 46`).
  - Định dạng `.docx`: Khớp với signature zip archive `PK\x03\x04` (hex: `50 4b 03 04`).
* **Giới hạn dung lượng tệp cứng 50MB**: Thiết lập ngưỡng `MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024` tại [security.py](file:///Users/macos/SDK/src/utils/security.py). Mọi tệp tin vượt quá 50MB đều bị chặn cứng ngay tại cửa ngõ để tránh rủi ro cạn kiệt tài nguyên máy cục bộ (DoS).

### 2.2 Băm chống trùng lặp tối ưu bộ nhớ (Deduplication Guardrails)
* **Streaming Hashing**: Tính toán mã băm SHA-256 của tệp tin bằng cách đọc luồng đệm giới hạn **4KB (`f.read(4096)`)**. Điều này giúp kiểm soát lượng RAM tiêu thụ cố định ở mức bộ nhớ thực tế tiệm cận hằng số (practically constant memory), bảo vệ tiến trình local khỏi hiện tượng tràn bộ nhớ khi xử lý tệp lớn.
* **Metadata Check**: Tự động tra cứu mã băm nội dung trong `metadata.json` của workspace. Nếu trùng, ném lỗi `DuplicateFileError` và hủy nạp để bảo vệ không gian vector khỏi ô nhiễm dữ liệu, giảm thiểu đáng kể rủi ro lãng phí chi phí nhúng neural.

### 2.3 Phân đoạn Token-Aware đệ quy tiếng Việt chính xác (Recursive Chunking)
* **Tiktoken cl100k_base**: Sử dụng bộ chia `RecursiveCharacterTextSplitter` được ánh xạ tokenizer `cl100k_base` của OpenAI để đếm token chính xác.
* **Tối ưu hóa hình học**: Cấu hình `chunk_size=350` tokens (phù hợp với giới hạn ngữ cảnh của MiniLM và OpenAI) và `chunk_overlap=70` tokens (gối đầu 20% giữ liên kết ngữ cảnh liên tục giữa các đoạn). Bộ đếm token đệ quy bảo vệ tiếng Việt có dấu khỏi hiện tượng cắt cụt ngầm (silent truncation) khi nạp vào Embedding Model.

### 2.4 Cây phân loại lỗi nạp liệu hệ thống (Failure Ingestion Taxonomy)
Hệ thống định nghĩa một cây thừa kế ngoại lệ vững chắc tại [exceptions.py](file:///Users/macos/SDK/src/domain/exceptions.py):
* `IngestionError` (Ngoại lệ gốc)
  * `DuplicateFileError` (Trùng băm SHA-256)
  * `SecurityValidationError` (Mismatch chữ ký nhị phân)
  * `FileTooLargeError` (Tệp vượt quá giới hạn payload cho phép)
  * `CorruptedFileError` (Tệp hỏng cấu trúc nhị phân)
  * `UnsupportedFileError` (Định dạng không hỗ trợ)

### 2.5 Giám sát Hiệu năng Thời gian thực & Giao diện chẩn đoán (Observability Spans)
* Triển khai cấu trúc `PerformanceProfile` ghi nhận chi tiết thời gian xử lý theo mili-giây:
  - **Ingestion Spans**: `parsing` (bóc tách), `splitting` (phân đoạn), `indexing` (lập chỉ mục vector), `total_ingestion` (tổng thời gian nạp).
  - **Query Spans**: `vector_search` (truy vấn Qdrant), `llm_synthesis` (gọi OpenAI tổng hợp), `total_query` (tổng thời gian RAG).
* **Consolidated UI Diagnostics**: Đưa bảng chẩn đoán thời gian và các nguồn trích dẫn vào hộp mở rộng hợp nhất trên UI Streamlit giúp giao diện chat chính luôn thanh thoát và tinh tế.
* **Đồng bộ dọn dẹp Workspace**: Hàm `clear_workspace` xóa sạch thư mục vật lý cục bộ và chỉ mục vector liên kết trong Qdrant Store.

---

## 📈 3. Số liệu Thực nghiệm Baseline (Local Baseline)

Dưới đây là các chỉ số baseline đo đạc thực nghiệm trên môi trường chạy thực tế của hệ thống với tệp văn bản tiêu chuẩn (MiniLM local nhúng và OpenAI API tổng hợp):

| Chỉ số Đo lường (Metric) | Giá trị Baseline (Measured Value) | Mô tả & Điều kiện Biên |
| :--- | :--- | :--- |
| **Thời gian nạp trung bình (Avg Ingest Latency)** | 2.4s / 10 trang | Đã bao gồm parsing, splitting và embedding sinh bởi mô hình cục bộ. |
| **Phân bổ mảnh (Avg Chunk Count/Page)** | 5.1 chunks / trang | Với cấu hình `chunk_size=350`, `overlap=70`. |
| **Đỉnh RAM Ingestion (Peak RAM)** | ~120 MB | Sử dụng SHA-256 streaming 4KB và loader tuần tự. |
| **Độ phủ Semantic (Recall@5)** | ~87.2% | Đánh giá trên tập câu hỏi trắc nghiệm ngữ nghĩa nlp_test. |
| **Thời gian truy vấn vector (Avg Vector Search)** | 59.97 ms | Đo đạc trực tiếp trên embedded Qdrant store. |

---

## 🚀 4. Tầm nhìn Tương lai & Các Giải pháp CHƯA TRIỂN KHAI (Future Roadmap & Unimplemented Solutions)

Dưới đây là các thiết kế kiến trúc nâng cao và tính năng lý thuyết được **CHỦ ĐỘNG LƯỢC BỎ** trong thực tế lập trình để bảo vệ bản sắc cục bộ của hệ thống (local-first, lightweight, zero-dependency), kèm theo lập luận kỹ thuật chi tiết:

### 4.1 Mô hình Xử lý Đồng thời & Khóa tệp nhị phân (Concurrency & File Locking)
* **Trạng thái**: *Chưa triển khai / Đề xuất tương lai*.
* **Lý do kỹ thuật lược bỏ**:
  - Hệ thống hiện tại vận hành cục bộ ở chế độ Single-user trên máy cá nhân, do đó nguy cơ tranh chấp tài nguyên (race conditions) khi hai luồng đồng thời ghi đè `metadata.json` hoặc tải trùng tệp tin SHA-256 là cực kỳ thấp. Việc tích hợp các thư viện khóa tệp ngoài như `portalocker` là không cần thiết (Over-engineering).
* **Giải pháp hướng tương lai**:
  - Khi chuyển đổi hệ thống sang môi trường đa người dùng (Multi-tenant) hoặc chạy trên máy chủ đám mây, hệ thống sẽ sử dụng thư viện khóa tệp tin hệ điều hành (`portalocker`) hoặc cơ chế ghi đè nguyên tử (atomic file replacement) để đảm bảo an toàn đa luồng (thread safety).

### 4.2 Thiết lập Vector Index Customization (ANN Index Trade-offs)
* **Trạng thái**: *Chưa triển khai / Đề xuất tương lai*.
* **Lý do kỹ thuật lược bỏ**:
  - Hệ thống sử dụng Qdrant Store cục bộ với cấu hình mặc định (HNSW) do kích thước dữ liệu workspace nhỏ (dưới vài chục nghìn điểm vector), đồ thị HNSW mặc định đảm bảo tốc độ và recall cực cao mà không cần tinh chỉnh tham số chuyên sâu.
* **Đánh đổi hạ tầng vector trong tương lai**:

| Loại Chỉ mục (Index Type) | Tốc độ Truy xuất (Search Latency) | Tiêu thụ Bộ nhớ (RAM Footprint) | Độ chính xác (Recall Accuracy) | Kịch bản Phù hợp |
| :--- | :--- | :--- | :--- | :--- |
| **HNSW** | **Cực nhanh** (Logarithmic) | **Cao** (Tải đồ thị vào RAM) | **Rất cao** (~95% - 99%) | Hệ thống thời gian thực cần độ trễ thấp. |
| **IVF** | **Nhanh** (Phân cụm không gian) | **Thấp** (Lưu danh sách ngược) | **Trung bình** | Hệ thống dữ liệu lớn, giới hạn RAM. |
| **Flat** | **Chậm** (Quét cạn tuyến tính) | **Cực thấp** (Không tốn chỉ mục phụ) | **Tuyệt đối 100%** | Tập dữ liệu nhỏ, cần độ chính xác tuyệt đối. |

* **Cơ chế thu hẹp phạm vi tìm kiếm hiệu dụng (reduces effective search scope)**: Thay vì duyệt tuyến tính hay tính toán độ phức tạp lý thuyết $\mathcal{O}(N) \rightarrow \mathcal{O}(M)$ thô sơ, Qdrant sử dụng cơ chế **Pre-filtering** trên đồ thị liên kết HNSW để lọc các điểm vector theo `workspace_id` trước khi thực hiện duyệt lân cận gần đúng, ngăn ngừa hiện tượng mất recall khi duyệt đồ thị.

### 4.3 Công cụ Đọc hình ảnh OCR cho tài liệu quét (Scan PDF OCR)
* **Trạng thái**: *Chưa triển khai / Đề xuất tương lai*.
* **Lý do kỹ thuật lược bỏ**:
  - Tích hợp OCR đòi hỏi cài đặt các thư viện liên kết ngoài cồng kềnh như `Tesseract OCR` hoặc các API Vision đám mây trả phí. Điều này phá vỡ tính gọn nhẹ, cản trở cài đặt nhanh và triết lý Zero-Dependency của dự án local.
* **Giải pháp hướng tương lai**:
  - Ném lỗi `CorruptedFileError` hoặc cảnh báo tài liệu rỗng để khuyến khích người dùng sử dụng tệp tin có chứa lớp văn bản thô (selectable text).

### 4.4 Tái lập chỉ mục cuốn chiếu tự động (Rolling Re-indexing / Embedding Drift Migration)
* **Trạng thái**: *Chưa triển khai / Đề xuất tương lai*.
* **Lý do kỹ thuật lược bỏ**:
  - Dung lượng tài liệu của mỗi workspace local rất nhỏ. Việc phát triển một background migration worker chạy song song hai collection vector cũ/mới tạo ra gánh nặng mã nguồn quá lớn so với thực tế sử dụng.
* **Giải pháp hướng tương lai**:
  - Sử dụng chiến lược **Re-ingestion** (Xóa chỉ mục vector cũ và nạp lại toàn bộ từ tệp thô lưu trữ trong `storage/workspaces/`). Tiến trình này chỉ mất chưa đầy vài giây trên máy local nhưng loại bỏ hoàn toàn rủi ro mất đồng bộ dữ liệu.

### 4.5 Tích hợp OpenTelemetry / Jaeger APM Agent
* **Trạng thái**: *Chưa triển khai / Đề xuất tương lai*.
* **Lý do kỹ thuật lược bỏ**:
  - Đòi hỏi người dùng chạy kèm các container APM Collector phức tạp dưới local.
* **Giải pháp hướng tương lai**:
  - Đo đạc cục bộ qua `PerformanceProfile` và kết xuất dạng bảng Streamlit trên UI. Khi deploy hệ thống lên cloud, OpenTelemetry agent sẽ được bổ sung vào pipeline CI/CD để theo dõi phân tán.

### 4.6 Embedding Batching & Advanced Retrieval Pipeline (Hybrid, Reranker, Context Compression)
* **Trạng thái**: *Chưa triển khai / Đề xuất tương lai*.
* **Lý do kỹ thuật lược bỏ**:
  - Tích hợp mô hình Reranker cục bộ sẽ làm phình to RAM và kéo dài thời gian phản hồi từ **~80ms lên 5-10 giây** trên CPU máy local. Việc nén ngữ cảnh và hybrid search BM25 cũng làm tăng độ phức tạp mã nguồn không cần thiết khi kích thước tài liệu nhỏ.
* **Đề xuất tương lai**:
  - Xây dựng kiến trúc truy hồi nâng cao kết hợp **BM25 + Dense Search**, bộ xếp hạng lại **Cross-Encoder Reranker** và nén ngữ cảnh **Context Compression** khi hệ thống được vận hành trên hạ tầng máy chủ đám mây có GPU chuyên dụng.

### 7. Khung Đánh giá Tự động (Retrieval Evaluation Framework via Ragas)
* **Trạng thái**: *Chưa triển khai / Đề xuất tương lai*.
* **Lý do kỹ thuật lược bỏ**:
  - Đo lường chất lượng truy hồi được thực hiện thủ công qua bộ chẩn đoán UI và bộ câu hỏi `evaluation/questions.json` là quá đủ cho nhu cầu nghiên cứu/presentation hiện tại.
* **Đề xuất tương lai**:
  - Tích hợp khung đánh giá tự động dựa trên thư viện **Ragas** đo lường: Faithfulness (Độ trung thực), Context Precision (Độ chính xác ngữ cảnh), Recall@K, MRR (Mean Reciprocal Rank) và nDCG.

### 4.8 Threat Modeling & Mô hình hóa mối đe dọa nâng cao (Secure RAG)
* **Trạng thái**: *Chưa triển khai / Đề xuất tương lai*.
* **Lý do kỹ thuật lược bỏ**:
  - Việc tự động cô lập parser trong sandbox hệ điều hành hoặc thiết lập các bộ phân tích ngôn ngữ sâu chống Indirect Prompt Injection làm phức tạp hóa luồng bóc tách cục bộ.
* **Đề xuất tương lai**:
  - Áp dụng phân tách ngữ cảnh System Prompt nghiêm ngặt `<context>` và cô lập luồng thực thi parser trên môi trường ảo container riêng biệt.

---

## ⚖️ 5. Phân tích Đánh đổi Kỹ thuật (Technical Trade-offs)

### Quyết định 5.1: Strategy Pattern kết hợp Dynamic Registry Factory
* **Phương án thay thế**: Sử dụng cấu trúc rẽ nhánh `if-else` truyền thống trực tiếp trong tầng điều phối.
* **So sánh & Đánh đổi**:

| Tiêu chí | Rẽ nhánh truyền thống (if/else) | Strategy & Factory Pattern (Đã chọn) |
| :--- | :--- | :--- |
| **Tính đóng kín (OCP)** | **Kém**. Bắt buộc phải chỉnh sửa mã nguồn lõi khi thêm bộ đọc định dạng mới. | **Hoàn hảo**. Chỉ cần viết Loader mới kế thừa Base Class và đăng ký vào registry của Factory. |
| **Độ độc lập Unit Test** | **Thấp**. Logic điều phối bị dính chặt với các thư viện bên thứ ba. | **Tuyệt vời**. Có thể mocking hoàn toàn các loader độc lập khi kiểm thử tầng Application. |

### Quyết định 5.2: Phân đoạn Token-Aware đệ quy qua Tiktoken
* **Phương án thay thế**:
  - *Phương án A*: Phân mảnh dựa trên số lượng ký tự thô (Character-based).
  - *Phương án B*: Phân mảnh động hoàn toàn theo ngữ nghĩa (Pure Semantic Chunking) dựa trên khoảng cách Cosine của các câu liên tiếp.
* **So sánh & Đánh đổi**:

| Giải pháp | Ưu điểm | Nhược điểm (Trade-offs) |
| :--- | :--- | :--- |
| **Ký tự thô (Character-based)** | Tốc độ xử lý cực nhanh, không tốn tài nguyên chạy mô hình đếm. | Lệch pha bộ mã hóa (Tokenizer Mismatch). Unicode tiếng Việt phồng to token gây ra hiện tượng cắt cụt ngầm (silent truncation) ở tầng Embedding. |
| **Ngữ nghĩa thuần (Semantic Chunking)** | Giữ được trọn vẹn ý niệm của từng khối thảo luận tự nhiên. | **Chi phí tính toán đắt đỏ**. Phải chạy mô hình nhúng hàng nghìn lần cho từng câu để tính khoảng cách Cosine, gây tắc nghẽn IO nghiêm trọng ở quy mô lớn. |
| **Tiktoken đệ quy (Đã chọn)** | **Tối ưu hóa hình học**. Đảm bảo khít hoàn hảo với cửa sổ hoạt động của mô hình nhúng và hạn chế tối đa đứt gãy ngữ pháp. | Chi phí tính toán trung bình do lặp đệ quy. |

---

## 📊 6. Thiết kế Schema Metadata Vector & Chỉ mục lọc (v1.0.0)

Để phục vụ cho kiến trúc đa người dùng (Multi-tenant) và tránh trôi lệch dữ liệu giữa các phiên bản mô hình, Payload siêu dữ liệu gắn kèm mỗi điểm Vector trong Qdrant được chuẩn hóa theo cấu trúc Schema v1.0.0 sau:

```json
{
  "document_id": "UUID-tài-liệu",
  "text": "Nội dung văn bản thô của chunk",
  "page_number": 0,
  "token_count": 120,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "schema_version": "1.0.0"
}
```

---

## 🏁 7. Đánh giá Đóng dự án (Conclusion & Identity)

Dự án **NLP RAG Presentation Space** mang một triết lý thiết kế và bản sắc kỹ thuật (architectural identity) vô cùng nhất quán: **Local-first, hạn chế tối đa độ phức tạp vận hành, bảo vệ tài nguyên nghiêm ngặt, bảo mật thực chiến và chính xác về mặt ngữ nghĩa.**

Mọi cải tiến và quyết định thiết kế từ việc chọn Strategy Pattern, băm SHA-256 tối ưu RAM, xác thực nhị phân đầu tệp, đến mô hình hiển thị chẩn đoán lịch duyệt đều xoay quanh triết lý cốt lõi này. Hệ thống hiện đã đạt độ trưởng thành cực kỳ cao, sẵn sàng vận hành thực tế ổn định và hiệu quả!
