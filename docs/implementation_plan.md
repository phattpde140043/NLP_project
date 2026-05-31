# 🏛️ Implementation Plan: Elevating RAG RFC to Enterprise Search Infrastructure & Retrieval Platform Standards

Bản kế hoạch này trình bày lộ trình nâng cấp toàn diện tài liệu kiến trúc **Technical Specification & Architecture RFC (Phase 3)** tại [vector_space_deep_dive.md](file:///Users/macos/SDK/docs/vector_space_deep_dive.md) và báo cáo tích hợp [walkthrough.md](file:///Users/macos/SDK/docs/walkthrough.md). Mục tiêu là chuyển đổi tài liệu từ cấp độ ứng dụng RAG thông thường lên chuẩn mực **Enterprise Retrieval Infrastructure & Platform** của các đội ngũ Search/ML Platform tại các hệ thống lớn (FAANG-level standards).

---

## 🗺️ 1. Các Nâng cấp Chi tiết sẽ Triển khai trong RFC

Chúng ta sẽ mở rộng và tái cấu trúc tài liệu RFC với **5 Chương lớn đột phá** và **9 Phân tích Kỹ thuật chuyên sâu**, bao gồm đầy đủ mô hình toán học và sơ đồ thiết kế:

### 🚀 (A) Retrieval Budget Optimization (Tối ưu hóa Ngân sách Truy hồi)
- **Mô hình hóa chi phí tổng thể**:
  $$Cost(q) = C_{\text{retrieval}} + C_{\text{rerank}} + C_{\text{LLM}}$$
- **Bài toán tối ưu ràng buộc (Constrained Latency Optimization)**:
  $$\text{minimize } Cost(q) \quad \text{subject to} \quad \text{Latency}(q) \le \text{SLA}_{\text{target}}$$
- **Cơ chế Ước lượng Độ phức tạp câu hỏi ($H_q$)**: Đo đạc độ hỗn loạn ngữ nghĩa (semantic ambiguity), độ hiếm từ vựng (lexical rarity thông qua phân phối IDF), và độ dài câu truy vấn để tự động điều chỉnh:
  $$K_{\text{candidate}} = f(H_q) \quad \text{và} \quad \text{ef\_search} = g(H_q)$$
- **Bảng phân loại trạng thái ngân sách**: So sánh hành vi và thông số giữa Low, Medium và High complexity queries.

### 📈 (B) Online Learning & Feedback Loops (Vòng phản hồi & Học máy Trực tuyến)
- **Hành vi Phản hồi Ngữ nghĩa Ngầm (Implicit Relevance Feedback)**:
  - *User Reformulation Rate (Tỷ lệ viết lại câu hỏi)*: Chỉ báo lỗi truy hồi ngữ nghĩa.
  - *Abandonment Rate (Tỷ lệ bỏ rơi)*: Chỉ báo kết quả không khớp/nhiễu.
  - *Citation Click-Through Rate (CTR)*: Proxy chính xác của mức độ liên quan của chunk.
  - *Answer Correction/Dislike Frequency*: Đo đạc tỷ lệ hallucination.
- **Data Loop Lifecycle**: Sơ đồ hóa cách thức thu thập các chỉ số phản hồi ngầm từ logs production, đóng gói thành bộ test-suite offline để định kỳ fine-tune mô hình embedding và hiệu chỉnh trọng số re-ranker.

### 🔄 (C) Embedding Lifecycle Management (Quản lý Vòng đời & Di chuyển Không gian Vector)
- **Chiến lược Cập nhật Mô hình Nhúng (Embedding Space Migration)**:
  - *Dual-Write Indexing*: Duy trì và ghi song song dữ liệu vào cả index cũ và index mới để đảm bảo tính sẵn sàng cao, tránh tham chiếu chéo không tương thích.
  - *Shadow Retrieval*: Luồng chạy ngầm tìm kiếm trên vector space mới để kiểm tra latency, recall, và stability trước khi đưa ra sử dụng.
  - *Retrieval A/B Testing & Phased Rollout*: Chuyển đổi lưu lượng người dùng dần dần (10% -> 50% -> 100%) dựa trên đo đạc CTR thực tế.
- **Ma trận Khôi phục lỗi & Chống lỗi phân tầng (Retrieval Fault Tolerance Matrix)**:
  - Thiết kế bảng fallback chi tiết cho các kịch bản lỗi: Dense retriever timeout, Reranker timeout, Qdrant node degraded, Embedding service overload.

### 🌐 (D) Distributed Search Architecture (Kiến trúc Tìm kiếm Phân tán & Lọc Payload)
- **Distributed ANN Federation**: Thiết kế cơ chế sharding (phân mảnh dữ liệu) dựa trên `tenant_id` làm partition key, shard routing, và replica consistency.
- **Filtered ANN Search Selectivity Collapse**: Phân tích toán học hiện tượng sụp đổ duyệt đồ thị HNSW khi lọc payload có tính chọn lọc quá cao (low selectivity).
- **Giải pháp Lọc cứng**: Phân tích so sánh 3 chiến lược: Pre-filtering (quét tuyến tính fallback của Qdrant), Post-filtering (lọc sau truy hồi), và Partitioned Collections (chia bộ sưu tập vật lý độc lập).

### 🧮 (E) Advanced Retrieval Models (Các Mô hình Truy hồi Tiên tiến)
- **Late Interaction (ColBERT)**: Cơ chế khớp mịn mức token sử dụng phép nhân vô hướng max-sim.
- **Sparse Expansion (SPLADE)**: Kỹ thuật mở rộng từ vựng để khắc phục lỗi lệch từ khóa.
- **HyDE (Hypothetical Document Embeddings)**: Tạo văn bản giả định qua LLM để di chuyển truy vấn về gần manifold tài liệu đích.
- **Dynamic Query Router**: Nâng cấp từ Regex/TF-IDF sang distilled intent classifier hoặc lightweight Transformer-based routing policy network để tối ưu hóa đường đi truy hồi.

---

## 🛠️ 2. Các Tệp Sẽ Cập nhật (Proposed Changes)

#### [MODIFY] [vector_space_deep_dive.md](file:///Users/macos/SDK/docs/vector_space_deep_dive.md)
- Tái cấu trúc và mở rộng sâu sắc 11 chương hiện tại để lồng ghép 5 chương lớn và 9 khoảng trống kiến trúc trên.
- Đảm bảo các công thức toán học LaTeX được định dạng rõ ràng, chính xác.

#### [MODIFY] [walkthrough.md](file:///Users/macos/SDK/docs/walkthrough.md)
- Cập nhật báo cáo kỹ thuật tổng quan để đồng bộ với cấu trúc nâng cấp của RFC.

---

## ⚖️ 3. Trade-off Analysis & Key Design Decisions

- **Scalability vs Latency vs Cost**: Việc áp dụng Budget Controller giúp tối ưu hóa chi phí API Cloud và tài nguyên CPU local, trong khi việc lượng tử hóa (Quantization) và phân vùng vật lý (Partitioning) đảm bảo khả năng mở rộng quy mô mà không đánh đổi latency.
- **Complexity vs Reliability**: Tích hợp các tầng fallback và graceful degradation tăng tính phức tạp trong code nhưng đảm bảo SLA uptime đạt mức 99.99% của doanh nghiệp lớn.

---

## 🔬 4. Verification Plan

1. **Kiểm tra Định dạng tài liệu**: Dùng trình đọc Markdown kiểm tra tính hợp lệ của cú pháp LaTeX và sơ đồ Mermaid.
2. **Kiểm tra Tích hợp**: Chạy lại test suite Phase 3 bằng pytest để đảm bảo mã nguồn hiện tại không bị ảnh hưởng bởi việc thay đổi/nâng cấp tài liệu lý thuyết.
   ```bash
   ./.venv/bin/python -m pytest tests/test_phase3_integration.py -v
   ```
