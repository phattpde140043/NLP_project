# 🏛️ Technical Specification & Implementation Plan: Query Processing, Adaptive Budgeting, and Intent Routing Layer (Phase 5 - Advanced Query Platform)

Tài liệu này cung cấp bản thiết kế kỹ thuật chi tiết (**Technical Specification & Implementation Plan**) cho **Tầng Xử lý, Phân tích & Định tuyến Truy vấn Thích ứng (Query Processing, Adaptive Budgeting & Dynamic Routing Layer)**. Hệ thống được nâng cấp toàn diện lên mô hình **Retrieval Control Plane / Adaptive Retrieval Operating Layer** cấp sản xuất lớn (Enterprise-Grade & Research-Grade Standards) kết hợp các phản hồi xuất sắc về tối ưu hệ thống từ Expert Review.

Hệ thống phân rã rạch ròi quy trình thành cấu trúc **8-Stage Control Plane Pipeline**, tách biệt tuyệt đối **Mặt phẳng Quyết định (Decision Plane)** khỏi **Mặt phẳng Thực thi (Execution Plane)** để triệt tiêu các lỗi về circular decision dependencies, QPS amplification, entropy instability, routing oscillation, semantic cache poisoning, và tail-latency explosion.

---

## 🗺️ 1. Quy trình Xử lý Chi tiết 8 Giai đoạn (8-Stage Control Plane Specification)

```text
  [ User Query ]
        │
        ▼
 ┌───────────────┐      ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
 │ Stage 1:      │      │ Stage 2:      │      │ Stage 3:      │      │ Stage 4:      │
 │ Cheap Query   │ ──🪶►│ Parallel      │ ──🪶►│ Uncertainty   │ ──🪶►│ Budget        │ ──┐
 │ Profiling     │      │ Hedged Search │      │ Diagnostics   │      │ Allocation    │   │
 └───────────────┘      └───────────────┘      └───────────────┘      └───────────────┘   │
                                                                                          │
 ┌───────────────┐      ┌───────────────┐      ┌───────────────┐      ┌───────────────┐   │
 │ Stage 8:      │◄──🪶─│ Stage 7:      │◄──🪶─│ Stage 6:      │◄──🪶─│ Stage 5:      │◄──┘
 │ LLM Dispatch  │      │ 2-Stage       │      │ Rerank Benefit│      │ Progressive   │
 │               │      │ Semantic Cache│      │ Predictor     │      │ Refinement    │
 └───────────────┘      └───────────────┘      └───────────────┘      └───────────────┘
```

### 🚀 1.1 Stage 1: Cheap Query Profiling & Calibrated Intent Routing
Stage 1 đưa ra **Hồ sơ Truy vấn (Query Profile)** độc lập với kết quả tìm kiếm thực tế với độ trễ cực thấp **$<1.5$ms**.
*   **Learned Lightweight Router**: Bộ phân loại Logistic Regression dựa trên trích xuất đặc trưng đặc thù:
    $$\mathbf{x} = \Big[ F_{\text{regex\_match}}, F_{\text{symbol\_density}}, F_{\text{caps\_ratio}}, F_{\text{token\_rarity}}, F_{\text{query\_length}}, F_{\text{lexical\_dominance}} \Big]$$
*   **Percentile-Based Online Calibration (Self-Tuning Temperature)**:
    Thay vì gán cứng các nhiệt độ $T$ gây ra routing oscillation, hệ thống tự động xác định nhiệt độ hiệu chuẩn $T$ dựa trên khoảng phân phối bách phân vị thực tế của các scores trong active corpus:
    $$T = \max\Big(0.05, \frac{P_{90}(\mathbf{s}) - P_{10}(\mathbf{s})}{2.0}\Big)$$
    Sử dụng $T$ này để làm mềm phân phối routing thô:
    $$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
*   **Temporal Routing Hysteresis & EMA Smoothing (Chống dao động)**:
    Để chống lại hiện tượng dao động định tuyến đột ngột (routing oscillation) do nhiễu score cực tiểu hoặc ANN randomness, hệ thống áp dụng bộ lọc làm mịn Exponential Moving Average (EMA) cho vector routing affinity theo thời gian:
    $$\hat{\mathbf{p}}_t = \beta \cdot \hat{\mathbf{p}}_{t-1} + (1-\beta)\mathbf{p}_t \quad (\text{Mặc định: } \beta = 0.7)$$
    Quyết định định tuyến sẽ chọn đường đi tối ưu thông qua $\operatorname{argmax}(\hat{\mathbf{p}}_t)$ nếu vượt ngưỡng độ chênh lệch tối thiểu $\delta = 0.1$, ngược lại giữ nguyên định tuyến trước đó để bảo toàn tính ổn định.

---

### 🔍 1.2 Stage 2: Parallel Hedged Search & Progressive ANN (Triệt tiêu QPS Amplification)
Để bảo vệ hệ thống khỏi sự bùng nổ QPS (QPS amplification) và triệt tiêu lỗi thắt nút cổ chai băng thông bộ nhớ của đồ thị HNSW:
*   **Parallel Hedged Retrieval**: Khởi chạy song song cả tìm kiếm vector (ANN Search trên Qdrant) và tìm kiếm từ khóa cục bộ (Local BM25) ngay từ đầu. Thay vì chờ đợi ANN thất bại rồi mới fallback (gây cộng dồn độ trễ), hệ thống sẽ theo dõi và lấy kết quả trả về trước từ hai luồng.
*   **Search Path Bias Control trong Progressive ANN**:
    Để chống lại hiện tượng kẹt vào local minima do đồ thị ANN bị aging hoặc các vùng ngữ nghĩa mật độ cao:
    $$\mathbf{frontier}_{\text{new}} = \alpha \cdot \mathbf{frontier}_{\text{reused}} + (1-\alpha) \cdot \mathbf{entrypoints}_{\text{fresh\_random}} \quad (\alpha = 0.7)$$
    Phối trộn đỉnh biên cũ từ Probe Search với các entry points ngẫu nhiên mới để tối đa hóa graph exploration diversity.

---

### 🧮 1.3 Stage 3: Multi-Signal Uncertainty Diagnostics Plane (Chẩn đoán Đa Chỉ số)
Hệ thống áp dụng chuẩn hóa robust và chẩn đoán đa chiều:
*   **Robust Normalization using Median & MAD**:
    Quy chuẩn hóa điểm số tương đồng sử dụng trung vị (Median) và độ lệch tuyệt đối trung vị (Median Absolute Deviation) để chống lại các giá trị biên dị biệt (outliers):
    $$s''_i = \mathrm{clip}\left(\frac{s_i - \mathrm{median}(\mathbf{s})}{\text{MAD}(\mathbf{s}) + \epsilon}, -4, 4\right)$$
*   **Multi-Signal Uncertainty Vector ($U_q$)**:
    Tích hợp 5 tín hiệu mạnh mẽ đại diện cho độ bất định ngữ nghĩa:
    $$U_q = \Big[ H_q, \text{Gap}_{1,2}, \text{ScoreVariance}, \text{SparseDenseDisagreement}, \text{RetrievalStability} \Big]$$
    Trong đó $\text{Gap}_{1,2} = s''_1 - s''_2$ là proxy phản ánh confidence thực sự của ranking frontier.

---

### ⏳ 1.4 Stage 4: Dynamic Budget Allocation & Circuit Breaker
Hệ thống Orchestrator quản lý ngân sách thời gian thực bảo vệ SLA:
*   **Tail-Latency Circuit Breaker**: Nếu khâu ANN kéo dài quá **40ms** (P95 threshold), circuit breaker lập tức ngắt mạch:
    *   Hủy bỏ toàn bộ tiến trình Reranking nặng nề phía sau.
    *   Sử dụng ngay kết quả từ luồng **Local BM25 đã chạy song song** ở Stage 2 làm kết quả hồi đáp tức thì.
    *   Bảo vệ độ trễ P99 luôn dưới mục tiêu **80ms**.

---

### 🛡️ 1.5 Stage 5: Progressive Refinement & Expected Utility HyDE Guardrails
HyDE là tác vụ đắt đỏ. Để kiểm soát chi phí tính toán (HyDE Cost Explosion Risk), hệ thống tích hợp **Expected Utility Gate**:
*   **HyDE Expected Utility Gate**:
    Hệ thống chỉ kích hoạt HyDE khi và chỉ khi $H_q \ge 3.5$ VÀ thỏa mãn điều kiện hữu dụng dự báo dương:
    $$\text{Utility}_{\text{hyde}} = \text{EstimatedRecallGain} - (\text{LatencyPenalty} + \text{TokenCost}) > 0$$
*   **Double-Lock Lexical Anchor Preservation & Custom Entity Weighting**:
    Bản dịch giả định bắt buộc phải bảo toàn tối thiểu **85%** trọng số neo từ vựng kỹ thuật:
    $$\text{WeightedRetention} = \frac{\sum_{a_i \in \text{Anchors}(q) \cap \text{Anchors}(\text{HyDE})} W(a_i)}{\sum_{a_j \in \text{Anchors}(q)} W(a_j)} \ge 0.85$$
    Hệ thống phân lớp thực thể và gán trọng số tối ưu (Expert-Recommended Weights):
    *   *Stack traces*: 3.5
    *   *API names, Class/function names, Error codes*: 3.0
    *   *SQL table names, Config keys, Environment vars, File paths, K8s resources*: 2.5
    *   *RFC/Spec identifiers, Version identifiers*: 2.0

---

### 📊 1.6 Stage 6: Rerank Benefit Predictor & Contextual Bandit Exploration
Reranker là một compute sink khổng lồ. Để tránh lãng phí compute và triệt tiêu lỗi vòng lặp phản hồi tự củng cố tiêu cực (Self-Reinforcing Feedback Loop Collapse):
*   **Feature-Rich Marginal Gain Predictor**:
    Ước lượng sự cải thiện thứ hạng dự kiến ($\text{Gain}_{\text{expected}}$) dựa trên $\text{Gap}_{1,2}$, $\text{ScoreVariance}$, $\text{RerankDisagreement}$, và $\text{ANN\_Depth}$. Nếu $\text{Gain}_{\text{expected}} < 0.05$, rerank sẽ bị bỏ qua để tiết kiệm compute.
*   **Contextual Bandit Exploration Budget**:
    Để tránh việc predictor bỏ qua rerank liên tục làm mất đi tính đa dạng của nhãn huấn luyện ngoại tuyến, hệ thống duy trì **Exploration Budget** cố định: **5-10%** lưu lượng truy cập ngẫu nhiên luôn được kích hoạt Rerank đầy đủ để thu thập các nhãn unbiased dữ liệu huấn luyện.

---

### 💾 1.7 Stage 7: 2-Stage Vector Semantic Cache & Corpus Centroid Drift Monitoring
Hệ thống nâng cấp bộ nhớ đệm chống ngộ độc ngữ nghĩa (poisoning) và trôi lệch khái niệm (concept drift):
1.  **2-Stage Cache Indexing**: Exact match lookup Layer 1 kết hợp Qdrant cache collection độc lập `nlp_semantic_cache` Layer 2, kèm công thức suy hao thời gian và hit decay:
    $$\text{CacheScore} = \text{Similarity} \times e^{-\lambda \cdot \Delta t} \times \log(1 + \text{HitFrequency})$$
2.  **Corpus Centroid Drift Monitoring (Chống trôi lệch dữ liệu cục bộ)**:
    Khi tài liệu mới được thêm/bớt liên tục, ý nghĩa ngữ nghĩa toàn cục của không gian vector thay đổi dẫn đến cache bị drift. Hệ thống đo đạc khoảng cách dịch chuyển của trọng tâm không gian vector (Centroid Drift):
    $$\text{Drift} = \|\boldsymbol{\mu}_t - \boldsymbol{\mu}_{t-1}\|_2$$
    Nếu $\text{Drift} > \theta$ (với $\theta = 0.08$), hệ thống tự động **invalidate cục bộ các phân vùng cache bị ảnh hưởng** thay vì chờ đợi epoch hết hạn.

---

### 📊 1.8 Stage 8: LLM Dispatch & Per-Model Calibration Profile
Hệ thống loại bỏ hoàn toàn các hằng số hardcode, thay vào đó mỗi mô hình nhúng đăng ký một **Calibration Profile** đặc thù trong `EmbeddingFactory` được đồng bộ hóa với bách phân vị thực tế.

---

## 📈 2. Thiết lập Khung Nghiên cứu Nâng cao (Phase 5.5 - Research Spec)

1.  **Online HNSW Aging & Navigation Entropy**: Theo dõi sự suy thoái đồ thị dựa trên **Navigation Entropy** đo lường mức độ đa dạng của các traversal paths:
    $$H_{\text{navigation}} = -\sum \pi_i \log \pi_i$$
    Nếu traversal entropy suy giảm liên tục, đồ thị đang collapse và hệ thống tự động trigger background reindexing.
2.  **Jaccard Retrieval Stability Index**: Đo lường tính ổn định của hệ thống bằng độ chồng lặp Jaccard giữa hai lượt chạy tìm kiếm liên tiếp:
    $$\text{Stability} = \frac{|\mathcal{R}_1 \cap \mathcal{R}_2|}{|\mathcal{R}_1 \cup \mathcal{R}_2|}$$

---

## 🛠️ 3. Kế hoạch Phát triển Chi tiết (Proposed Code Modifications)

#### 1. [NEW] [query_config.py](file:///Users/macos/SDK/src/domain/services/query_config.py)
*   Định nghĩa `CalibrationProfile`, `MABRouterState`, và các tham số mặc định của 8-Stage Control Plane.

#### 2. [NEW] [query_analyzer.py](file:///Users/macos/SDK/src/application/query_analyzer.py)
*   Triển khai khâu trích xuất đặc trưng đặc thù cho truy vấn.
*   Cơ chế **Lightweight Logistic Classifier** định tuyến ý định, kết hợp hiệu chuẩn nhiệt độ động percentile-based.
*   Bộ lọc **Temporal Routing Hysteresis** làm mịn EMA để chống oscillation.
*   Khâu chuẩn hóa nâng cao **Robust MAD Normalization** và **Softmax Entropy**.
*   Tính toán véc-tơ chẩn đoán bất định đa chỉ số ($U_q$) và đo đạc Jaccard Stability.
*   Công thức HNSW Aging Check và Navigation Entropy.

#### 3. [NEW] [query_expander.py](file:///Users/macos/SDK/src/application/query_expander.py)
*   Tích hợp tạo giả thuyết HyDE qua OpenAI hoặc local mockup fallback.
*   Trích xuất **Entity-Weighted Lexical Anchors** bằng biểu thức chính quy nâng cao và dải trọng số định danh kỹ thuật (Stack traces, API names, SQL, v.v.).
*   Áp dụng **Double-Lock Expected Utility Guardrail validation**.

#### 4. [NEW] [query_cache.py](file:///Users/macos/SDK/src/application/query_cache.py)
*   Quản lý bộ nhớ đệm 2 tầng thông qua cấu trúc lookup map song song với Qdrant collection `nlp_semantic_cache`.
*   Triển khai công thức suy hao Cache theo thời gian và lượt gọi.
*   Tính toán **Corpus Centroid Drift** để tự động invalidate cache phân mảnh.

#### 5. [MODIFY] [rag_pipeline.py](file:///Users/macos/SDK/src/application/rag_pipeline.py)
*   Tích hợp bộ điều phối 8-Stage Control Plane trong `query_workspace`.
*   Triển khai **Hedged Search** chạy song song ANN và BM25 cục bộ.
*   Cơ chế ghép biên đồ thị **Search Path Bias Blending**.
*   Ghi nhận Trace Graph chi tiết cùng Latency breakdown phục vụ trực quan hóa.

#### 6. [NEW] [test_query_analysis.py](file:///Users/macos/SDK/tests/test_query_analysis.py)
*   Viết suite kiểm thử tự động toàn diện kiểm chứng: routing calibration, MAD robustness, parallel hedged query execution, lexical retention rate, và semantic cache lookup.
