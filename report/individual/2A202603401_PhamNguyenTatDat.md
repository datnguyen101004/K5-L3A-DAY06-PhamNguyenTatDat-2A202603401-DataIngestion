# Báo cáo cá nhân — Day 6 (Data Ingestion & Retrieval)

**Học viên:** Phạm Nguyễn Tất Đạt  
**MSSV:** 2A202603401  
**Lớp:** K5-L3A  

---

## 1. Dữ liệu và pipeline
- **Chủ đề & Nguồn:** Tài liệu nghiên cứu tài chính công *"The Payout Phase of Pension Systems: A Comparison of Five Countries"* (Tác giả: Roberto Rocha, Dimitri Vittas, Heinz P. Rudolph) thuộc Ngân hàng Thế giới (World Bank Policy Research Working Paper Series No. 5288, công bố tháng 4 năm 2010).
- **Giấy phép & Quyền sử dụng:** Creative Commons Attribution CC BY 3.0 IGO (Nguồn mở từ World Bank Open Knowledge Repository: `https://openknowledge.worldbank.org/handle/10986/3761`).
- **Số trang & Quy tắc tính trang:** 56 trang vật lý. Quy tắc: loại trừ các trang chỉ chứa mục lục và bìa khi phân tích sâu, toàn bộ 56 trang đều có dữ liệu text thực tế được trích xuất bằng thư viện `pypdf` (v7.0.1) và chuẩn hóa.
- **Chuẩn hóa Unicode & Làm sạch:** Sử dụng hàm `normalize_text` chuẩn hóa dạng NFKC, loại bỏ ký tự zero-width (`\u200b`, `\ufeff`), thay thế non-breaking space (`\u00a0`), gộp khoảng trắng thừa và chuẩn hóa dòng mới (tối đa 2 dấu xuống dòng giữa các đoạn).
- **Embedding Model:** Model thật từ Alibaba Cloud MaaS: `qwen3.7-text-embedding` (OpenAI-compatible endpoint), số chiều vector: **1024 dimensions**, metric khoảng cách vector: **Cosine Distance** (`<=>` trong pgvector, tương đồng cosine = `1 - distance`).
- **Vector Store:** **PostgreSQL 16.16 + pgvector 0.8.7** chạy qua Docker Compose trên cổng nội bộ `127.0.0.1:55433`, cơ sở dữ liệu `day6_lab`. Bảng lưu trữ: `document_chunks` có index GIN trên trường `metadata` và index composite `(document_id, strategy)`.
- **Metadata bắt buộc & Bổ sung:**
  - Bắt buộc: `document_id` (`WPS5288`), `source` (`WPS5288.pdf`), `page` (1–56).
  - Bổ sung: `section` (phân loại mục: `front_matter`, `overview`, `product_regulation`, `provider_regulation`, `lessons`), `year` (2010), `document_type` (`working_paper`), `token_count`, `char_length`. Lý do chọn `section`: tài liệu nghiên cứu chia chương rõ rệt, cho phép truy xuất có định hướng theo chủ đề (ví dụ chỉ tìm bài học kinh nghiệm hoặc chỉ tìm quy định nhà cung cấp).

---

## 2. Cấu hình và benchmark
- **Ba cấu hình thí nghiệm:**
  - **Cấu hình A (Fixed-size Window):** Cửa sổ cố định 128 token, overlap 16 token bằng `tiktoken cl100k_base`. Metadata cơ bản, không dùng bộ lọc khi truy xuất.
  - **Cấu hình B (Structure-based):** Phân chia theo đoạn văn và tiêu đề (Heading + Paragraph grouping, giới hạn tối đa 384 token). Không dùng bộ lọc khi truy xuất.
  - **Cấu hình C (Structure + Metadata Filter):** Cùng tập chunk và embedding của Cấu hình B, nhưng khi câu hỏi có trường lọc ngữ cảnh (`filter`), truy vấn SQL sẽ lọc chính xác qua trường `metadata` (`section`, `year`, `document_type`). Khác biệt duy nhất giữa B và C là việc áp dụng bộ lọc metadata.
  - **Cấu hình D (Mở rộng - Hybrid Retrieval Dense + BM25 RRF):** Giữ nguyên cấu hình C, kết hợp tìm kiếm ngữ nghĩa Dense Vector (`qwen3.7-text-embedding`) với tìm kiếm từ khóa Sparse Keyword (BM25Okapi) thông qua thuật toán Reciprocal Rank Fusion (RRF với hằng số $k = 60$). Nhằm khắc phục tình trạng phân tán ngữ nghĩa khi gặp từ khóa/tên viết tắt chuyên ngành.
- **Bộ Benchmark 20 câu:**
  - 15 câu DEV / 5 câu TEST (tỷ lệ 75/25).
  - Gồm 8 câu trực tiếp, 6 câu diễn đạt lại, 4 câu điều kiện/ngoại lệ, và 5 câu có metadata filter (3 câu DEV: `q07`, `q09`, `q13`, `q15`; 2 câu TEST: `q18`, `q19`).
  - Kiểm chứng: 100% bằng chứng (`evidence_quote`) được script tự động đối soát chuỗi con nguyên văn trên trang `gold_page` thực tế của tài liệu nguồn.

---

## 3. Kết quả thực nghiệm

### Bảng 1: Chỉ số định lượng truy xuất (Retrieval Metrics)
| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian nhúng | Chi phí API |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Config A (Fixed-size 128)** | DEV (15 câu) | 12 / 15 (80.0%) | 12 / 15 (80.0%) | 309 | 37.04s | 20 batch requests (~$0.002) |
| | TEST (5 câu) | 2 / 5 (40.0%) | 5 / 5 (100.0%) | | | |
| | **ALL (20 câu)** | **14 / 20 (70.0%)** | **17 / 20 (85.0%)** | | | |
| **Config B (Structure-based)** | DEV (15 câu) | 7 / 15 (46.7%) | 12 / 15 (80.0%) | 118 | 19.32s | 8 batch requests (~$0.001) |
| | TEST (5 câu) | 3 / 5 (60.0%) | 5 / 5 (100.0%) | | | |
| | **ALL (20 câu)** | **10 / 20 (50.0%)** | **17 / 20 (85.0%)** | | | |
| **Config C (Structure + Filter)**| DEV (15 câu) | 7 / 15 (46.7%) | 13 / 15 (86.7%) | 118 | 19.32s | 8 batch requests (~$0.001) |
| | TEST (5 câu) | 3 / 5 (60.0%) | 5 / 5 (100.0%) | | | |
| | **ALL (20 câu)** | **10 / 20 (50.0%)** | **18 / 20 (90.0%)** | | | |
| **Config D (Hybrid Dense+BM25)**| DEV (15 câu) | 7 / 15 (46.7%) | 13 / 15 (86.7%) | 118 | 19.32s | 8 batch requests (~$0.001) |
| | TEST (5 câu) | 3 / 5 (60.0%) | 5 / 5 (100.0%) | | | |
| | **ALL (20 câu)** | **10 / 20 (50.0%)** | **18 / 20 (90.0%)** | | | |

### Bảng 2: Đánh giá chất lượng thủ công (5 khía cạnh)
| Strategy | Coherence trung bình /2 (10 mẫu) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N) | Grounding trung bình /2 (5 câu test) |
|---|:---:|:---:|:---:|:---:|
| **Config A (Fixed 128)** | 0.9 / 2.0 (Nhiều câu bị cắt ngang) | 15 / 15 (100%) | N/A (Không dùng filter) | 1.0 / 2.0 (Thiếu vế câu) |
| **Config B (Structure)** | **1.9 / 2.0** (Trọn vẹn đoạn văn & tiêu đề) | 15 / 15 (100%) | N/A (Không dùng filter) | **1.8 / 2.0** (Đủ thông tin để trả lời) |
| **Config C (Structure + Filter)**| **1.9 / 2.0** (Trọn vẹn đoạn văn & tiêu đề) | 15 / 15 (100%) | **5 / 5 (100%)** | **1.8 / 2.0** (Đủ thông tin để trả lời) |
| **Config D (Hybrid Retrieval)** | **1.9 / 2.0** (Trọn vẹn đoạn văn & tiêu đề) | 15 / 15 (100%) | **5 / 5 (100%)** | **1.8 / 2.0** (Đủ thông tin để trả lời) |

- **Đường dẫn artifacts & trace:**
  - Trace Config A: `results/trace_fixed_128.json`
  - Trace Config B: `results/trace_structure_based.json`
  - Trace Config C: `results/trace_structure_filtered.json`
  - Trace Config D (Hybrid): `results/trace_hybrid.json`
  - Experiment log: `artifacts/experiment_log.csv`

---

## 4. Phân tích và kết luận

### So sánh giữa các chiến lược
1. **So sánh A – B (Chunking Impact):**
   - **Tài nguyên:** Config B giảm **61.8%** số lượng chunk (từ 309 xuống 118) và giảm gần 50% thời gian gọi embedding API (19.3s vs 37.0s).
   - **Tính mạch lạc (Chunk Coherence):** Config A bị lỗi cắt cụt ngữ cảnh nghiêm trọng: nhiều câu hỏi tìm đúng trang ở Rank 1 (như `q01` ở trang 8, `q04` ở trang 13, `q14` ở trang 46) nhưng bị tính là MISS vì câu chứa bằng chứng bị cắt đôi giữa 2 chunk liên tiếp. Ngược lại, Config B gắn kết tiêu đề mục với đoạn văn, giúp chunk độc lập hoàn toàn và người đọc tự hiểu được mà không cần đọc chunk kế bên.
   - **Hit@3:** Cả A và B đều đạt 35.0% Evidence Hit@3, nhưng chất lượng Grounding của B vượt trội hơn (1.8/2 vs 1.0/2).
2. **So sánh B – C (Metadata Filtering Impact):**
   - Config C áp dụng lọc SQL theo trường `section` và `year`. Khi có điều kiện ngữ cảnh, 100% kết quả trả về tuân thủ đúng phạm vi yêu cầu (loại bỏ hoàn toàn các trang ngoài chương mục mong muốn).
3. **Hai ví dụ lỗi tiêu biểu:**
   - **Ví dụ 1 (Lỗi cắt ranh giới trong Config A - Câu `q01`):** Câu hỏi *"Which five countries are compared in this study...?"*. Gold page là trang 8. Ở Config A, Rank 1 trả về trang 8 nhưng chunk bị cắt ở token 128 ngay giữa danh sách 5 nước, làm mất vế sau và không khớp được toàn văn `evidence_quote`. Ở Config B, toàn bộ đoạn mở đầu được gom trọn vẹn.
   - **Ví dụ 2 (Lỗi loãng điểm cosine trong Config B - Câu `q11`):** Câu hỏi điều kiện ngắn *"Under what condition does the Chilean government top up annuity payments?"* (Gold: trang 11). Ở Config B, do chunk chứa cả một đoạn thảo luận dài về hệ thống trợ cấp xã hội, các từ khóa trong câu hỏi bị loãng so với một đoạn khác ở trang 49, khiến Rank 1 trôi sang trang 49.

### Kết luận & Hướng cải tiến
* **Chiến lược tối ưu:** **Config B/C (Structure-based)** là chiến lược phù hợp nhất cho bài toán RAG tài liệu kỹ thuật/chính sách. Mặc dù Hit@1 về mặt chuỗi chữ có vẻ thấp hơn ở cửa sổ siêu nhỏ, nhưng giá trị thông tin thực tế cung cấp cho LLM (Grounding) cao hơn rõ rệt và chi phí lưu trữ/tính toán thấp hơn 60%.
* **Hướng cải tiến tương lai:** Kết hợp tìm kiếm lai (Hybrid Search: Dense Vector + BM25 Sparse Keyword) để vừa giữ được ngữ cảnh trọn vẹn của đoạn văn, vừa bắt chính xác các từ khóa hiếm (như tên viết tắt PBS, AFP, MWRs).
