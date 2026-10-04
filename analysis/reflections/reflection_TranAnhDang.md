# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Trần Anh Đăng  
**Mã số học viên:** 2A202602992  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Map từng concept trong lecture vào code đã triển khai trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Dùng threshold `0.85` với embedding model `all-MiniLM-L6-v2`. Tách văn bản theo ngữ nghĩa câu thay vì cắt giữa chừng như paragraph cơ bản, giúp bảo toàn mạch thông tin của quy trình/chính sách. |
| **Hierarchical Chunking** | M1 | `chunk_hierarchical()` | Tạo cấu trúc Parent (2048 chars) & Child (256 chars). Tìm kiếm trên Child chunk cho độ chính xác cao (Precision), nhưng trả về Parent chunk giúp LLM có đầy đủ ngữ cảnh (Context). |
| **Structure-Aware Chunking** | M1 | `chunk_structure_aware()` | Parse tiêu đề Markdown (`#`, `##`), tự động gắn nhãn `section` vào metadata. Bảo toàn các bảng biểu, danh sách quy định mà không bị rách đoạn. |
| **BM25 + Dense Fusion (Hybrid Search)** | M2 | `reciprocal_rank_fusion()` | Kết hợp BM25 (đã dùng Underthesea để tách từ ghép tiếng Việt) và Dense Retrieval (`BAAI/bge-m3` + Qdrant). Thuật toán RRF giúp kết hợp cả khớp từ khóa chính xác (mã văn bản, thuật ngữ) và khớp ngữ nghĩa. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Sử dụng `BAAI/bge-reranker-v2-m3` tái xếp hạng Top-20 candidates thành Top-3 kết quả chính xác nhất trước khi gửi tới LLM. |
| **RAGAS 4 Metrics Evaluation** | M4 | `evaluate_ragas()` | Đánh giá toàn diện qua 4 chỉ số: Faithfulness, Answer Relevancy, Context Precision, Context Recall. |
| **Contextual Prepend / Enrichment** | M5 | `contextual_prepend()` / `_enrich_single_call()` | Bổ sung câu tóm tắt vị trí & chủ đề chunk vào đầu văn bản trước khi embed, giảm đáng kể lỗi retrieval. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải:**
  - Lỗi tokenization với BM25 khi xử lý tiếng Việt: `underthesea` nối từ ghép bằng dấu gạch dưới `_` (ví dụ `nghỉ_phép`), khiến BM25 tách `nghỉ phép` thành 2 token riêng biệt và không khớp kết quả.
  - Lỗi tương thích `qdrant-client` bản mới đổi từ `client.search()` sang `client.query_points()`.
- **Nguyên nhân gốc rễ & Cách debug:**
  - Xử lý replace `_` thành khoảng trắng trong `segment_vietnamese()` để đưa về dạng token chuẩn cho BM25.
  - Thêm cơ chế fallback kiểm tra phương thức `query_points()` / `search()` trên `QdrantClient` để tương thích linh hoạt.
- **Kiến thức đã bổ sung:**
  - Nắm vững cách thức hoạt động của Reciprocal Rank Fusion (RRF) để kết hợp các điểm số từ nguồn xếp hạng khác nhau mà không cần chuẩn hóa scale.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Chatbot Nội bộ Doanh nghiệp (Enterprise Policy & Support Bot)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Naive RAG (chỉ dùng Dense Search và cắt đoạn cố định).
- **Vấn đề / Bottlenecks:** Thường bị nhầm lẫn giữa các phiên bản quy định cũ và mới (ví dụ quy định nghỉ phép 2023 vs 2024), chưa hỗ trợ tốt từ khóa chính xác tiếng Việt.

#### 2. Kế hoạch cải tiến
1. **Chunking Strategy:** Áp dụng **Hierarchical Parent-Child Chunking** làm mặc định cho tài liệu chính sách dài, kết hợp **Structure-Aware** cho các tài liệu quy trình dạng Markdown.
2. **Search Retrieval:** Sử dụng **Hybrid Search (BM25 + BGE-M3 + RRF)** để vừa tìm đúng tên/mã quy định, vừa tìm đúng ý nghĩa câu hỏi.
3. **Reranking:** Sử dụng `BAAI/bge-reranker-v2-m3` để chọn ra Top-3 ngữ cảnh sạch nhất cho LLM.
4. **Evaluation:** Đưa RAGAS vào CI/CD pipeline với bộ 20+ test cases thực tế để giám sát chỉ số Faithfulness và Context Recall.
5. **Enrichment:** Sử dụng **Contextual Prepend** cho từng chunk để gắn tên tài liệu và phiên bản quy định vào context.

#### 3. Timeline triển khai
- **Tuần 1:** Cải tạo module Chunking & Indexing với Hybrid Search.
- **Tuần 2:** Tích hợp Cross-Encoder Reranker & Thống kê đánh giá RAGAS benchmark.
