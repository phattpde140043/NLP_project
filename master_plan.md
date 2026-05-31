# 🏛️ Master Plan: Dự án NLP RAG Presentation Space

Tài liệu này đóng vai trò là **Bản đồ Tiến trình Tổng thể (Master Plan)**, phân chia dự án thành các Phase, Stage và Step chi tiết. Đây là công cụ quản lý và kiểm tra tiến trình chuẩn xác, phản ánh trạng thái thực hiện hiện tại của hệ thống RAG Clean Architecture hỗ trợ đa định dạng (PDF & DOCX).

---

## 📊 Bảng Tóm tắt Tiến trình (Overall Progress Summary)

| Phase | Mô tả mục tiêu | Tình trạng | Hoàn thành |
| :--- | :--- | :---: | :---: |
| **Phase 1** | Kiến trúc Phân tầng Sạch (Clean Architecture & Restructuring) | **ĐÃ HOÀN THÀNH** | 100% |
| **Phase 2** | Nạp Tài liệu & Phân đoạn Token-Aware (Ingestion & Text Splitting) | **ĐÃ HOÀN THÀNH** | 100% |
| **Phase 3** | Lưu trữ & Tìm kiếm Tương đồng (Vector Space Indexing & Retrieval) | **ĐÃ HOÀN THÀNH** | 100% |
| **Phase 4** | Tổng hợp LLM & Rào chắn ảo tưởng (RAG Synthesis & Guardrails) | **ĐÃ HOÀN THÀNH** | 100% |
| **Phase 5** | Giao diện Người dùng Streamlit (Aesthetic Streamlit UI) | **ĐÃ HOÀN THÀNH** | 100% |
| **Phase 6** | Đánh giá & Kiểm chuẩn (Evaluation & Verification) | **ĐÃ HOÀN THÀNH** | 100% |

---

## 📋 Chi tiết các Phase, Stage & Step

### 🏗️ Phase 1: Kiến trúc Phân tầng Sạch (Clean Architecture & Restructuring)
*Mục tiêu: Thiết lập cấu trúc mã nguồn theo chuẩn Clean Architecture, tách biệt rạch ròi giữa UI, Logic nghiệp vụ (Domain), và các Adapter Hạ tầng (Infrastructure).*

* **Stage 1.1: Tái cấu trúc Thư mục (Folder Re-architecting)**
  - `[x]` **Step 1.1.1**: Phân rã thư mục nguồn sang kiến trúc Clean Architecture: `src/config/`, `src/domain/models/`, `src/domain/services/`, `src/infrastructure/`, `src/application/`.
  - `[x]` **Step 1.1.2**: Tập trung hóa cấu hình dự án tại `src/config/settings.py` giải quyết tuyệt đối đường dẫn tĩnh `/Users/macos/SDK` và nạp biến môi trường.
  - `[x]` **Step 1.1.3**: Dọn dẹp triệt để các file legacy `main.py`, `src/config.py`, `src/models.py`, `src/database.py` và DB SQLite `notebook_lm.db` dư thừa.
* **Stage 1.2: Thiết lập Thực thể Miền (Domain Entities & Config)**
  - `[x]` **Step 1.2.1**: Thiết lập Domain Entity `Chunk` độc lập (`src/domain/models/chunk.py`).
  - `[x]` **Step 1.2.2**: Thiết lập Domain Entity `Document` độc lập (`src/domain/models/document.py`) tích hợp thuộc tính `content_hash`.
  - `[x]` **Step 1.2.3**: Triển khai `RetrievalConfig` dạng dataclass quản lý cấu hình truy hồi (`similarity_threshold=0.35` và `top_k=4`).

---

### 📄 Phase 2: Nạp Tài liệu & Phân đoạn Token-Aware (Document Ingestion & Text Splitting)
*Mục tiêu: Trích xuất văn bản thô chuẩn xác từ cả PDF và DOCX thông qua Strategy Pattern, phân đoạn văn bản thông minh theo token của mô hình ngôn ngữ.*

* **Stage 2.1: Bóc tách cấu trúc tài liệu trừu tượng (Document Parsing Abstraction)**
  - `[x]` **Step 2.1.1**: Định nghĩa giao diện trừu tượng chung `BaseDocumentLoader` (`src/infrastructure/loaders/base_loader.py`).
  - `[x]` **Step 2.1.2**: Triển khai `PDFLoader` kế thừa từ `BaseDocumentLoader`, bóc tách văn bản theo trang vật lý (`src/infrastructure/loaders/pdf_loader.py`).
  - `[x]` **Step 2.1.3**: Triển khai `DOCXLoader` kế thừa từ `BaseDocumentLoader`, bóc tách văn bản Word bằng `Docx2txtLoader` (`src/infrastructure/loaders/docx_loader.py`).
  - `[x]` **Step 2.1.4**: Xây dựng `DocumentLoaderFactory` tự động chọn chiến lược loader tương ứng theo đuôi file (`.pdf`, `.docx`).
* **Stage 2.2: Phân mảnh Văn bản Token-Aware (Token-Aware Chunking)**
  - `[x]` **Step 2.2.1**: Tích hợp bộ mã hóa `cl100k_base` thông qua `tiktoken` đo đạc chính xác số lượng tokens thực tế của từng chunk.
  - `[x]` **Step 2.2.2**: Triển khai chia nhỏ `RecursiveCharacterTextSplitter.from_tiktoken_encoder` với kích thước `chunk_size=350` và `chunk_overlap=70` tokens giúp khít tối đa vào cửa sổ MiniLM.
* **Stage 2.3: Băm Nội dung Chống Trùng lặp (SHA-256 Hashing & Incremental Ingestion)**
  - `[x]` **Step 2.3.1**: Xây dựng hàm băm nội dung SHA-256 độc lập (`src/utils/hash_util.py`).
  - `[x]` **Step 2.3.2**: Triển khai rào chắn kiểm tra trùng lặp thông minh (Incremental Update) dựa trên mã băm trong `rag_pipeline.py`.

---

### 🧠 Phase 3: Lưu trữ & Tìm kiếm Tương đồng (Vector Space Indexing & Retrieval)
*Mục tiêu: Biến văn bản thành không gian hình học đa chiều và thực hiện so khớp ngữ nghĩa Cosine Similarity tốc độ cao.*

* **Stage 3.1: Sinh Vector Nhúng (Semantic Embeddings)**
  - `[x]` **Step 3.1.1**: Tạo `EmbeddingFactory` phân phối linh hoạt giữa mô hình chạy local `sentence-transformers/all-MiniLM-L6-v2` (384 dim) và cloud `OpenAIEmbeddings` (1536 dim).
* **Stage 3.2: Không gian Lưu trữ Qdrant Local (Local Vector Store Indexing)**
  - `[x]` **Step 3.2.1**: Triển khai `QdrantStoreService` khởi tạo Collection cục bộ với cấu hình kích thước hình học và độ đo `COSINE` đồng bộ.
  - `[x]` **Step 3.2.2**: Lưu trữ vector nhúng gán siêu dữ liệu `page_number` và `document_id` cô lập theo `notebook_id` (Workspace UUID) trên đĩa phẳng.
  - `[x]` **Step 3.2.3**: Viết hàm tìm kiếm tương đồng ngữ nghĩa `search_similar_chunks` so khớp toán học góc Cosine với ngưỡng tối thiểu.

---

### 💬 Phase 4: Tổng hợp LLM & Rào chắn Ảo tưởng (RAG Synthesis & Guardrails)
*Mục tiêu: Đóng gói luồng điều phối, tích hợp LLM để tổng hợp câu trả lời chuẩn xác và rào chắn ảo tưởng thông tin.*

* **Stage 4.1: Tổng hợp Thông tin Chuẩn xác (Hallucination-Proof Synthesis)**
  - `[x]` **Step 4.1.1**: Triển khai lớp hạ tầng `ChatModelService` gọi OpenAI Chat API.
  - `[x]` **Step 4.1.2**: Cấu hình System Prompt rào chắn khắt khe buộc LLM chỉ được dùng ngữ cảnh đã cung cấp, trả về thông điệp fallback chuẩn xác nếu thiếu thông tin phù hợp.
* **Stage 4.2: Phân mảnh Workspace bằng UUID Phẳng (Workspace Segmentation & Zero-DB Tracking)**
  - `[x]` **Step 4.2.1**: Thiết lập phân vùng lưu trữ phẳng không dùng database: `storage/workspaces/<uuid>/pdfs/` và `metadata.json` quản lý trạng thái nạp.

---

### 🎨 Phase 5: Giao diện Người dùng Streamlit (Aesthetic Streamlit UI)
*Mục tiêu: Xây dựng giao diện Streamlit cao cấp, trực quan, hỗ trợ gỡ lỗi và quan sát đặc trưng hình học vector.*

* **Stage 5.1: Bố cục 2-8 Cao cấp (Layout & Glassmorphism)**
  - `[x]` **Step 5.1.1**: Triển khai bố cục 2 cột tinh tế với phong cách Sleek Dark Mode và Glassmorphism kính mờ.
  - `[x]` **Step 5.1.2**: Hiển thị UUID Workspace hiện tại dưới dạng Web3 short hash rút gọn (`📍 e7c8b21a...a30f`).
  - `[x]` **Step 5.1.3**: Tích hợp danh sách cuộn mượt mà các Workspace cũ cùng các nút tạo không gian mới.
* **Stage 5.2: Trạng thái Nạp Động (Dynamic Status Badges)**
  - `[x]` **Step 5.2.1**: Hiển thị danh sách tài liệu trong Popover gọn gàng, hỗ trợ cả PDF và DOCX.
  - `[x]` **Step 5.2.2**: Triển khai Badge trạng thái động dựa trên file metadata: `⚡` Completed (Hiển thị số chunks), `⏳` Processing, `❌` Failed.
* **Stage 5.3: Bộ chẩn đoán Truy hồi (Retrieval Diagnostics & Viewer)**
  - `[x]` **Step 5.3.1**: Tích hợp bảng thống kê Metric chẩn đoán truy hồi bên dưới mỗi câu trả lời: Kích thước Vector, Max Score, Avg Score, số lượng chunks.
  - `[x]` **Step 5.3.2**: Tích hợp hộp mở rộng `st.expander` hiển thị trực tiếp nội dung văn bản thô cùng tỉ lệ phần trăm tương đồng và trang trích dẫn.

---

### 📊 Phase 6: Đánh giá & Kiểm chuẩn (Evaluation & Verification)
*Mục tiêu: Giải quyết xung đột môi trường, đo đạc hiệu năng tìm kiếm ngữ nghĩa, xác nhận sự chính xác của pipeline bóc tách văn bản mới và hoàn thiện báo cáo.*

* **Stage 6.1: Khắc phục Xung đột Dependency Môi trường (Environment Alignment)**
  - `[x]` **Step 6.1.1**: Nhận diện lỗi import PyTorch/NumPy 2.x và thư viện `transformers` 5.x không tương thích.
  - `[x]` **Step 6.1.2**: Hạ cấp `transformers` xuống `4.44.2` và khóa phiên bản `numpy==1.26.4` để tương thích hoàn toàn với PyTorch 2.2.2 cục bộ.
  - `[x]` **Step 6.1.3**: Kiểm tra và xác nhận tất cả thư viện khoa học dữ liệu hoạt động trơn tru mà không có cảnh báo nghiêm trọng.
* **Stage 6.2: Bộ Dữ liệu Kiểm chuẩn Cục bộ & Lập kịch bản Test (Local Benchmarking)**
  - `[x]` **Step 6.2.1**: Thiết lập tệp benchmarks kiểm chuẩn `evaluation/questions.json` đa dạng câu hỏi cho cả PDF và DOCX.
  - `[x]` **Step 6.2.2**: Lập kịch bản kiểm thử tích hợp tự động (`scratch/test_rag_pipeline.py`) thực hiện nạp tệp Word, trích xuất text, băm SHA-256, chia chunk, sinh vector nhúng và tìm kiếm Cosine Similarity.
* **Stage 6.3: Kiểm thử Liên kết Cuối & Giao diện (E2E UI Verification)**
  - `[x]` **Step 6.3.1**: Thực hiện chạy kịch bản test tích hợp tự động qua CLI để đảm bảo các truy vấn vector trả về kết quả chính xác từ Qdrant Store (Đã hoàn thành 100% qua kịch bản `test_rag_pipeline.py`).
  - `[x]` **Step 6.3.2**: Kiểm tra cơ chế phát hiện trùng lặp SHA-256 (Incremental Ingestion) đối với cả tệp PDF và Word trên Streamlit UI (Đã xác minh hoạt động hoàn hảo).
  - `[x]` **Step 6.3.3**: Thực hiện câu hỏi nghiên cứu RAG trên giao diện Streamlit, kiểm tra kết quả truy hồi và gỡ lỗi qua bảng chẩn đoán Retrieval Diagnostics (Đã kiểm tra thành công).
* **Stage 6.4: Tài liệu Tổng duyệt & Báo cáo (Final Walkthrough Documentation)**
  - `[x]` **Step 6.4.1**: Hoàn thiện tài liệu tổng duyệt `walkthrough.md` thể hiện cấu trúc mới và kết quả kiểm tra (Báo cáo đã xuất bản tại `walkthrough.md`).
