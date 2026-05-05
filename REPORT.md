# Bao cao Lab Day 19: GraphRAG voi Tech Company Corpus

## 1. Nghien cuu va chuan bi

### Entity Extraction

LLM phan biet entity va relation bang cach doc context quanh cum tu. Entity thuong la danh tu rieng hoac doi tuong co vai tro ro rang trong cau: cong ty, nguoi sang lap, san pham, dia diem, nam, to chuc dau tu. Relation duoc suy ra tu dong tu/cau truc ngu nghia ket noi hai entity, vi du `founded by`, `headquartered in`, `developed`, `acquired`, `invested in`.

Vi du cau "OpenAI was founded by Sam Altman and Elon Musk in 2015" sinh ra cac triples:

- `(OpenAI, FOUNDED_BY, Sam Altman)`
- `(OpenAI, FOUNDED_BY, Elon Musk)`
- `(OpenAI, FOUNDED_IN, 2015)`

### Graph Construction

Deduplication quan trong vi cung mot entity co the xuat hien voi nhieu bien the: `OpenAI`, `OpenAI Global, LLC`, `OpenAI Inc.` hoac `Google`, `Google LLC`. Neu khong chuan hoa, graph se bi tach node gia, lam traversal mat lien ket va tang nhieu canh trung lap. Pipeline trong repo chuan hoa whitespace, relation label va loai triple trung key `(subject, relation, object)`.

### Query Answering

Vector search thong thuong tim cac chunk gan cau hoi theo ngu nghia. Cach nay tot khi cau tra loi nam trong mot vai doan van ban gan nhau. BFS tren graph khac o cho no duyet truc tiep tu entity seed sang cac node lien quan nhieu hop. GraphRAG phu hop hon voi cau hoi can noi nhieu quan he, vi du "OpenAI lien quan Microsoft nhu the nao?" hoac "so sanh cac cong ty theo linh vuc".

## 2. Cong cu

- NetworkX: dung de build `MultiDiGraph`, luu node/entity va edge/relation kem evidence.
- Neo4j: phu hop khi can luu graph lon, query bang Cypher va truc quan hoa bang Browser/Bloom. Lab nay uu tien NetworkX de chay nhanh local.
- NodeRAG: la huong all-in-one cho GraphRAG, nhung pipeline trong repo tu cai dat cac buoc cot loi de sinh vien thay ro indexing, graph construction, retrieval va evaluation.

## 3. Thiet ke pipeline

1. Load `data/corpus.json`.
2. Chunk moi document theo word window co overlap.
3. Trich xuat triples:
   - Co API key: goi OpenAI chat model de extract JSON triples.
   - Khong co API key: dung rule-based fallback de lab van chay offline.
4. Deduplicate triples.
5. Build NetworkX `MultiDiGraph`.
6. Save graph, triples, chunks va visualization.
7. Query:
   - Flat RAG: retrieve top chunks bang embedding OpenAI, fallback lexical.
   - GraphRAG: detect entity seed, BFS 2-hop, textualize graph facts, ket hop supporting chunks.
8. Benchmark 20 cau hoi, ghi time/token/context size.

## 4. Cach danh gia

Chay:

```bash
python -m src.graphrag_lab all
```

Sau do mo:

- `artifacts/knowledge_graph.png` de xem do thi tri thuc.
- `artifacts/benchmark_results.csv` de so sanh tung cau hoi.
- `artifacts/index_metrics.json` de xem chi phi indexing.

Cot nen phan tich trong benchmark:

- `seconds`: thoi gian tra loi.
- `total_tokens`: tong token chat + embedding.
- `context_chars`: do dai context dua vao answer model.
- `answer`: dung de cham thu cong xem Flat RAG hay GraphRAG chinh xac hon.

## 5. Ky vong ket qua

Flat RAG thuong nhanh va on voi cau hoi co keyword ro nhu "Where is Qualcomm headquartered?". GraphRAG thuong tot hon voi cau hoi can noi ket entity, relation va multi-hop, vi du quan he OpenAI-Microsoft, Google-Alphabet, hoac so sanh cac cong ty theo nganh. Truong hop Flat RAG lay nham chunk gan keyword nhung thieu relation, GraphRAG co loi the vi facts da duoc cau truc hoa.

