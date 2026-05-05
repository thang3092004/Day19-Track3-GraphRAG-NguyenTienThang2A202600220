# Lab Day 19: Xay dung he thong GraphRAG voi Tech Company Corpus

Repo nay da hoan thien pipeline cho lab GraphRAG tren corpus 5 cong ty cong nghe trong `data/corpus.json`: OpenAI, Qualcomm, Vingroup, Google va Viettel.

## 1. Muc tieu

- Tach van ban tho thanh cac chunk co kich thuoc on dinh.
- Trich xuat thuc the va quan he thanh triple `(subject, relation, object)`.
- Xay dung do thi tri thuc bang NetworkX.
- Truy van bang hai cach:
  - Flat RAG: tim chunk lien quan bang embedding hoac lexical fallback.
  - GraphRAG: nhan dien entity trong cau hoi, duyet hang xom tren graph, textualize facts roi dua vao LLM.
- Benchmark 20 cau hoi va ghi lai thoi gian, token, kich thuoc context.

## 2. Cai dat

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Mo file `.env` va dien:

```bash
OPENAI_API_KEY=sk-...
```

Neu chua co API key, code van chay duoc bang fallback rule-based extraction va lexical retrieval, nhung chat answer se la cau tra loi extractive don gian.

## 3. Chay pipeline

Index corpus va tao do thi:

```bash
python -m src.graphrag_lab index
```

Hoi thu bang GraphRAG:

```bash
python -m src.graphrag_lab ask --question "What is the relationship between OpenAI and Microsoft?"
```

Chay benchmark 20 cau cho ca Flat RAG va GraphRAG:

```bash
python -m src.graphrag_lab benchmark
```

Chay tat ca:

```bash
python -m src.graphrag_lab all
```

## 4. Artifact dau ra

Sau khi chay, folder `artifacts/` se co:

- `chunks.jsonl`: cac chunk da cat tu corpus.
- `triples.csv`: danh sach triples da trich xuat.
- `knowledge_graph.json`: graph dang node-link JSON.
- `knowledge_graph.png`: anh chup/visualization do thi tri thuc.
- `index_metrics.json`: so document, chunk, triple, node, edge, time, token.
- `benchmark_results.csv`: bang so sanh Flat RAG va GraphRAG tren 20 cau hoi.

## 5. File nen nop

- Ma nguon: `src/graphrag_lab.py` hoac notebook `notebooks/day19_graphrag_lab.ipynb`.
- Anh do thi: `artifacts/knowledge_graph.png`.
- Bang benchmark: `artifacts/benchmark_results.csv`.
- Bao cao ngan: `REPORT.md`.

