# Báo cáo Lab Day 19: GraphRAG với Tech Company Corpus

## 1. Nghiên cứu và chuẩn bị

### Entity Extraction

LLM phân biệt entity và relation bằng cách đọc ngữ cảnh quanh cụm từ. Entity thường là danh từ riêng hoặc đối tượng có vai trò rõ trong câu: công ty, người sáng lập, sản phẩm, địa điểm, năm, tổ chức đầu tư. Relation được suy ra từ động từ hoặc cấu trúc ngữ nghĩa nối hai entity, ví dụ `founded by`, `headquartered in`, `developed`, `acquired`, `invested in`.

Ví dụ câu "OpenAI was founded by Sam Altman and Elon Musk in 2015" có thể sinh ra các triples:

- `(OpenAI, FOUNDED_BY, Sam Altman)`
- `(OpenAI, FOUNDED_BY, Elon Musk)`
- `(OpenAI, FOUNDED_IN, 2015)`

### Graph Construction

Deduplication quan trọng vì cùng một entity có thể xuất hiện với nhiều biến thể: `OpenAI`, `OpenAI Global, LLC`, `OpenAI Inc.` hoặc `Google`, `Google LLC`. Nếu không chuẩn hóa, graph sẽ bị tách node giả, làm traversal mất liên kết và tăng nhiều cạnh trùng lặp. Pipeline trong repo chuẩn hóa whitespace, relation label và loại triple trùng key `(subject, relation, object)`.

### Query Answering

Vector search thông thường tìm các chunk gần câu hỏi theo ngữ nghĩa. Cách này tốt khi câu trả lời nằm trong một vài đoạn văn bản gần nhau. BFS trên graph khác ở chỗ nó duyệt trực tiếp từ entity seed sang các node liên quan nhiều hop. GraphRAG phù hợp hơn với câu hỏi cần nối nhiều quan hệ, ví dụ "OpenAI liên quan Microsoft như thế nào?" hoặc "so sánh các công ty theo lĩnh vực".

## 2. Công cụ

- NetworkX: dùng để build `MultiDiGraph`, lưu node/entity và edge/relation kèm evidence.
- Neo4j: phù hợp khi cần lưu graph lớn, query bằng Cypher và trực quan hóa bằng Browser/Bloom. Lab này ưu tiên NetworkX để chạy nhanh local.
- NodeRAG: là hướng all-in-one cho GraphRAG, nhưng pipeline trong repo tự cài đặt các bước cốt lõi để thấy rõ indexing, graph construction, retrieval và evaluation.

## 3. Thiết kế pipeline

1. Load `data/corpus.json`.
2. Chunk mỗi document theo word window có overlap.
3. Trích xuất triples:
   - Có API key: gọi OpenAI chat model để extract JSON triples.
   - Không có API key: dùng rule-based fallback để lab vẫn chạy offline.
4. Deduplicate triples.
5. Build NetworkX `MultiDiGraph`.
6. Save graph, triples, chunks và visualization.
7. Query:
   - Flat RAG: retrieve top chunks bằng embedding OpenAI, fallback lexical.
   - GraphRAG: detect entity seed, BFS 2-hop, textualize graph facts, kết hợp supporting chunks.
8. Benchmark 20 câu hỏi, ghi time/token/context size.

## 4. Kết quả chạy bằng API

Kết quả trong `artifacts/index_metrics.json`:

- Documents: 5
- Chunks: 39
- Triples: 397
- Graph nodes: 337
- Graph edges: 397
- Indexing time: 658.477 giây
- Indexing token usage: 28,563 prompt tokens + 24,792 completion tokens = 53,355 tokens

Kết quả benchmark trong `artifacts/benchmark_results.csv`:

| Hệ thống | Số câu | Thời gian trung bình | Tổng token | Context trung bình |
|---|---:|---:|---:|---:|
| Flat RAG | 20 | 5.338 giây | 516,024 | 7,336.6 ký tự |
| GraphRAG | 20 | 4.757 giây | 528,407 | 9,802.1 ký tự |

Tổng benchmark có 40 dòng vì mỗi câu hỏi được chạy trên cả hai hệ thống.

## 5. Phân tích chi phí

Chi phí xây dựng graph tập trung ở bước LLM extraction. Với 39 chunks, pipeline dùng 53,355 tokens để tạo 397 triples. Thời gian indexing khoảng 11 phút do phải gọi LLM tuần tự cho từng chunk và retry khi network chậm.

Chi phí benchmark cao hơn indexing vì mỗi câu hỏi chạy hai lần: Flat RAG và GraphRAG. Flat RAG dùng nhiều token embedding/chat cho retrieval và answer. GraphRAG có context dài hơn vì textualize graph facts rồi ghép thêm supporting text, nên tổng token benchmark nhỉnh hơn Flat RAG.

Trong thực tế production, có thể giảm chi phí bằng cách cache embeddings, cache triples theo chunk hash, chạy extraction batch/async, và giới hạn số facts đưa vào context theo độ liên quan.

## 6. Nhận xét

Flat RAG thường ổn với câu hỏi có keyword rõ và câu trả lời nằm trong một chunk, ví dụ câu hỏi về headquarters hoặc founder. GraphRAG hữu ích hơn với câu hỏi cần nối nhiều entity/relation hoặc multi-hop, ví dụ quan hệ OpenAI-Microsoft, Google-Alphabet, hoặc so sánh các công ty theo ngành. Trường hợp Flat RAG lấy nhầm chunk gần keyword nhưng thiếu relation, GraphRAG có lợi thế vì facts đã được cấu trúc hóa.

## 7. Artifact nộp bài

- Mã nguồn: `src/graphrag_lab.py`
- Notebook: `notebooks/day19_graphrag_lab.ipynb`
- Ảnh đồ thị: `artifacts/knowledge_graph.png`
- Triples: `artifacts/triples.csv`
- Benchmark 20 câu: `artifacts/benchmark_results.csv`
- Metrics chi phí: `artifacts/index_metrics.json`
