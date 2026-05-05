from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_QUESTIONS = [
    "Who founded OpenAI and when was it founded?",
    "What is the relationship between OpenAI and Microsoft?",
    "Which products or model families did OpenAI develop?",
    "What governance change did OpenAI make in 2025?",
    "What lawsuits or controversies are associated with OpenAI?",
    "Where is Qualcomm headquartered and what does it make?",
    "Who founded Qualcomm?",
    "How does Qualcomm relate to Snapdragon?",
    "What are Qualcomm's major business areas?",
    "What legal or antitrust issues has Qualcomm faced?",
    "Who founded Google?",
    "What products and services is Google known for?",
    "How does Google relate to Alphabet?",
    "What acquisitions or subsidiaries are connected to Google?",
    "What AI-related products or research are connected to Google?",
    "What is Vingroup and who founded it?",
    "What are Vingroup's major business sectors?",
    "What is Viettel and who owns or operates it?",
    "What markets or countries are connected to Viettel?",
    "Compare OpenAI, Google, Qualcomm, Vingroup, and Viettel by industry focus.",
]


@dataclass
class Config:
    data_path: Path = Path("data/corpus.json")
    artifact_dir: Path = Path("artifacts")
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size_words: int = 450
    chunk_overlap_words: int = 80
    max_chunks_per_doc: int = 8
    graph_depth: int = 2
    graph_max_facts: int = 40
    use_llm_extraction: bool = True

    @classmethod
    def from_env(cls, env_path: str | Path = ".env") -> "Config":
        load_env_file(env_path)
        return cls(
            data_path=Path(os.getenv("DATA_PATH", "data/corpus.json")),
            artifact_dir=Path(os.getenv("ARTIFACT_DIR", "artifacts")),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            chunk_size_words=int(os.getenv("CHUNK_SIZE_WORDS", "450")),
            chunk_overlap_words=int(os.getenv("CHUNK_OVERLAP_WORDS", "80")),
            max_chunks_per_doc=int(os.getenv("MAX_CHUNKS_PER_DOC", "8")),
            graph_depth=int(os.getenv("GRAPH_DEPTH", "2")),
            graph_max_facts=int(os.getenv("GRAPH_MAX_FACTS", "40")),
            use_llm_extraction=os.getenv("USE_LLM_EXTRACTION", "true").lower() == "true",
        )


@dataclass
class Document:
    id: str
    title: str
    source: str
    url: str
    content: str


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    source: str
    url: str
    ordinal: int


@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    evidence: str
    doc_id: str
    chunk_id: str


class UsageMeter:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.embedding_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.embedding_tokens

    def add_chat_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    def add_embedding_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        self.embedding_tokens += int(getattr(usage, "total_tokens", 0) or 0)

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "total_tokens": self.total_tokens,
        }


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_corpus(path: Path) -> list[Document]:
    raw_docs = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for item in raw_docs:
        docs.append(
            Document(
                id=str(item.get("id", item.get("title", len(docs) + 1))),
                title=str(item.get("title", "")),
                source=str(item.get("source", "")),
                url=str(item.get("url", "")),
                content=normalize_text(str(item.get("content", ""))),
            )
        )
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size_words: int,
    overlap_words: int,
    max_chunks_per_doc: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max(1, chunk_size_words - overlap_words)
    for doc in docs:
        words = doc.content.split()
        doc_chunks = []
        for start in range(0, len(words), step):
            piece = words[start : start + chunk_size_words]
            if len(piece) < 80:
                break
            ordinal = len(doc_chunks)
            chunk = Chunk(
                id=f"doc{doc.id}_chunk{ordinal:03d}",
                doc_id=doc.id,
                title=doc.title,
                text=" ".join(piece),
                source=doc.source,
                url=doc.url,
                ordinal=ordinal,
            )
            doc_chunks.append(chunk)
            if len(doc_chunks) >= max_chunks_per_doc:
                break
        chunks.extend(doc_chunks)
    return chunks


def get_openai_client(api_key: str) -> Any | None:
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    return OpenAI(api_key=api_key)


def parse_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        cleaned = match.group(0)
    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError("LLM output is not a JSON array")
    return [item for item in data if isinstance(item, dict)]


def extract_triples_llm(
    chunks: list[Chunk],
    cfg: Config,
    usage: UsageMeter,
) -> list[Triple]:
    client = get_openai_client(cfg.openai_api_key)
    if client is None:
        return []

    triples: list[Triple] = []
    system_prompt = (
        "You extract knowledge graph triples from technology-company text. "
        "Return only valid JSON array items with keys: subject, relation, object, evidence. "
        "Use concise entity names and uppercase relation labels such as FOUNDED_BY, HEADQUARTERED_IN, "
        "DEVELOPED, ACQUIRED, INVESTED_IN, OWNED_BY, PARTNERED_WITH, CEO_OF, SUBSIDIARY_OF, LOCATED_IN. "
        "Do not invent facts."
    )
    for chunk in chunks:
        user_prompt = (
            f"Document title: {chunk.title}\n"
            f"Chunk id: {chunk.id}\n"
            "Extract up to 12 important company/person/product/location/date triples from this text.\n\n"
            f"{chunk.text[:5000]}"
        )
        try:
            response = client.chat.completions.create(
                model=cfg.openai_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            usage.add_chat_usage(response)
            content = response.choices[0].message.content or "[]"
            for item in parse_json_array(content):
                subject = normalize_entity(str(item.get("subject", "")))
                relation = normalize_relation(str(item.get("relation", "")))
                obj = normalize_entity(str(item.get("object", "")))
                evidence = normalize_text(str(item.get("evidence", "")))[:400]
                if subject and relation and obj:
                    triples.append(Triple(subject, relation, obj, evidence, chunk.doc_id, chunk.id))
        except Exception as exc:
            print(f"[warn] LLM extraction failed for {chunk.id}: {exc}")
    return deduplicate_triples(triples)


def normalize_entity(value: str) -> str:
    value = normalize_text(value)
    value = re.sub(r"^[^\w]+|[^\w]+$", "", value)
    return value[:120]


def normalize_relation(value: str) -> str:
    value = normalize_text(value).upper()
    value = re.sub(r"[^A-Z0-9]+", "_", value).strip("_")
    return value[:80] or "RELATED_TO"


def extract_triples_rules(chunks: list[Chunk]) -> list[Triple]:
    triples: list[Triple] = []
    patterns = [
        (r"(?P<s>[A-Z][A-Za-z0-9&.\- ]{1,80}) was founded .*? by (?P<o>[A-Z][A-Za-z0-9&.,\- ]{1,160})", "FOUNDED_BY"),
        (r"(?P<s>[A-Z][A-Za-z0-9&.\- ]{1,80}) (?:is|was) headquartered in (?P<o>[A-Z][A-Za-z0-9&.,\- ]{1,120})", "HEADQUARTERED_IN"),
        (r"(?P<s>[A-Z][A-Za-z0-9&.\- ]{1,80}) developed (?P<o>[A-Z][A-Za-z0-9&.,\- ]{1,120})", "DEVELOPED"),
        (r"(?P<s>[A-Z][A-Za-z0-9&.\- ]{1,80}) acquired (?P<o>[A-Z][A-Za-z0-9&.,\- ]{1,120})", "ACQUIRED"),
        (r"(?P<s>[A-Z][A-Za-z0-9&.\- ]{1,80}) invested .*? in (?P<o>[A-Z][A-Za-z0-9&.,\- ]{1,120})", "INVESTED_IN"),
        (r"(?P<s>[A-Z][A-Za-z0-9&.\- ]{1,80}) (?:owns|owned) (?P<o>[A-Z][A-Za-z0-9&.,\- ]{1,120})", "OWNS"),
        (r"(?P<s>[A-Z][A-Za-z0-9&.\- ]{1,80}) (?:is|was) (?:a|an|the)? ?subsidiary of (?P<o>[A-Z][A-Za-z0-9&.,\- ]{1,120})", "SUBSIDIARY_OF"),
        (r"(?P<o>[A-Z][A-Za-z0-9&.\- ]{1,80}) (?:served as|became|is|was) CEO of (?P<s>[A-Z][A-Za-z0-9&.,\- ]{1,120})", "CEO_OF"),
    ]
    fallback_relations = {
        "founded": "MENTIONS_FOUNDING",
        "headquartered": "MENTIONS_HEADQUARTERS",
        "acquired": "MENTIONS_ACQUISITION",
        "subsidiary": "MENTIONS_SUBSIDIARY",
        "lawsuit": "MENTIONS_LAWSUIT",
        "antitrust": "MENTIONS_ANTITRUST",
        "artificial intelligence": "MENTIONS_AI",
        "telecommunications": "MENTIONS_TELECOM",
    }

    for chunk in chunks:
        sentences = split_sentences(chunk.text)
        for sent in sentences[:80]:
            for pattern, relation in patterns:
                for match in re.finditer(pattern, sent):
                    subject = normalize_entity(match.group("s"))
                    obj = normalize_entity(match.group("o"))
                    if subject and obj and len(subject.split()) <= 8:
                        triples.append(Triple(subject, relation, obj, sent[:300], chunk.doc_id, chunk.id))
            low = sent.lower()
            for keyword, relation in fallback_relations.items():
                if keyword in low:
                    triples.append(Triple(chunk.title, relation, keyword, sent[:300], chunk.doc_id, chunk.id))
        triples.append(Triple(chunk.title, "HAS_SOURCE", chunk.source or "Unknown source", chunk.title, chunk.doc_id, chunk.id))
    return deduplicate_triples(triples)


def split_sentences(text: str) -> list[str]:
    compact = normalize_text(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", compact) if len(s.strip()) > 20]


def deduplicate_triples(triples: list[Triple]) -> list[Triple]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Triple] = []
    for triple in triples:
        key = (triple.subject.lower(), triple.relation, triple.object.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(triple)
    return deduped


def build_graph(triples: list[Triple]) -> Any:
    import networkx as nx

    graph = nx.MultiDiGraph()
    for triple in triples:
        graph.add_node(triple.subject, label=triple.subject)
        graph.add_node(triple.object, label=triple.object)
        graph.add_edge(
            triple.subject,
            triple.object,
            relation=triple.relation,
            evidence=triple.evidence,
            doc_id=triple.doc_id,
            chunk_id=triple.chunk_id,
        )
    return graph


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks


def save_triples(triples: list[Triple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(triples[0]).keys()) if triples else list(Triple.__annotations__.keys()))
        writer.writeheader()
        for triple in triples:
            writer.writerow(asdict(triple))


def load_triples(path: Path) -> list[Triple]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [Triple(**row) for row in csv.DictReader(f)]


def save_graph(graph: Any, path: Path) -> None:
    import networkx as nx
    from networkx.readwrite import json_graph

    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_graph.node_link_data(graph, edges="edges")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_graph(path: Path) -> Any:
    import networkx as nx
    from networkx.readwrite import json_graph

    data = json.loads(path.read_text(encoding="utf-8"))
    return json_graph.node_link_graph(data, edges="edges")


def visualize_graph(graph: Any, path: Path, top_n: int = 35) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    path.parent.mkdir(parents=True, exist_ok=True)
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot visualize an empty graph")
    degree = dict(graph.degree())
    top_nodes = [node for node, _ in sorted(degree.items(), key=lambda item: item[1], reverse=True)[:top_n]]
    subgraph = graph.subgraph(top_nodes).copy()
    pos = nx.spring_layout(subgraph, seed=42, k=0.8)
    plt.figure(figsize=(15, 10))
    node_sizes = [500 + degree.get(node, 1) * 90 for node in subgraph.nodes()]
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color="#5DADE2", alpha=0.88)
    nx.draw_networkx_edges(subgraph, pos, arrows=True, edge_color="#6E6E6E", alpha=0.35, width=1.2)
    nx.draw_networkx_labels(subgraph, pos, font_size=9)
    edge_labels = {}
    for u, v, data in subgraph.edges(data=True):
        edge_labels[(u, v)] = data.get("relation", "RELATED_TO")
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]{1,}", text.lower())


def lexical_rank(query: str, chunks: list[Chunk], top_k: int = 5) -> list[tuple[Chunk, float]]:
    query_terms = Counter(tokenize(query))
    scores = []
    for chunk in chunks:
        terms = Counter(tokenize(chunk.title + " " + chunk.text))
        overlap = sum(min(query_terms[t], terms[t]) for t in query_terms)
        title_bonus = 2.0 if chunk.title.lower() in query.lower() else 0.0
        score = overlap / math.sqrt(max(1, len(terms))) + title_bonus
        if score > 0:
            scores.append((chunk, score))
    return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]


def embed_texts(texts: list[str], cfg: Config, usage: UsageMeter) -> list[list[float]] | None:
    client = get_openai_client(cfg.openai_api_key)
    if client is None:
        return None
    try:
        response = client.embeddings.create(model=cfg.embedding_model, input=texts)
        usage.add_embedding_usage(response)
        return [item.embedding for item in response.data]
    except Exception as exc:
        print(f"[warn] OpenAI embedding failed, falling back to lexical retrieval: {exc}")
        return None


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def flat_retrieve(query: str, chunks: list[Chunk], cfg: Config, usage: UsageMeter, top_k: int = 5) -> list[tuple[Chunk, float]]:
    texts = [query] + [chunk.title + "\n" + chunk.text for chunk in chunks]
    vectors = embed_texts(texts, cfg, usage)
    if vectors is None:
        return lexical_rank(query, chunks, top_k)
    query_vec = vectors[0]
    scored = [(chunk, cosine(query_vec, vector)) for chunk, vector in zip(chunks, vectors[1:])]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


def entity_candidates(query: str, graph: Any) -> list[str]:
    query_low = query.lower()
    terms = set(tokenize(query))
    scored = []
    for node in graph.nodes():
        node_low = str(node).lower()
        node_terms = set(tokenize(str(node)))
        score = 0
        if node_low in query_low:
            score += 5
        score += len(terms & node_terms)
        if score:
            scored.append((str(node), score))
    return [node for node, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:5]]


def graph_facts(query: str, graph: Any, cfg: Config) -> list[str]:
    seeds = entity_candidates(query, graph)
    if not seeds:
        return []

    facts: list[str] = []
    visited = set(seeds)
    queue = deque((seed, 0) for seed in seeds)
    while queue and len(facts) < cfg.graph_max_facts:
        node, depth = queue.popleft()
        if depth >= cfg.graph_depth:
            continue
        neighbors = set(graph.successors(node)) | set(graph.predecessors(node))
        for neighbor in neighbors:
            edge_datas = []
            if graph.has_edge(node, neighbor):
                edge_datas.extend(graph.get_edge_data(node, neighbor).values())
            if graph.has_edge(neighbor, node):
                edge_datas.extend(graph.get_edge_data(neighbor, node).values())
            for data in edge_datas[:3]:
                rel = data.get("relation", "RELATED_TO")
                evidence = data.get("evidence", "")
                if graph.has_edge(node, neighbor):
                    fact = f"{node} --{rel}--> {neighbor}. Evidence: {evidence}"
                else:
                    fact = f"{neighbor} --{rel}--> {node}. Evidence: {evidence}"
                facts.append(fact)
                if len(facts) >= cfg.graph_max_facts:
                    break
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
            if len(facts) >= cfg.graph_max_facts:
                break
    return facts


def build_context_from_chunks(scored_chunks: list[tuple[Chunk, float]]) -> str:
    lines = []
    for rank, (chunk, score) in enumerate(scored_chunks, start=1):
        excerpt = chunk.text[:1400]
        lines.append(f"[{rank}] {chunk.title} | score={score:.3f} | {chunk.url}\n{excerpt}")
    return "\n\n".join(lines)


def build_context_from_graph(query: str, graph: Any, chunks: list[Chunk], cfg: Config, usage: UsageMeter) -> str:
    facts = graph_facts(query, graph, cfg)
    fallback_chunks = flat_retrieve(query, chunks, cfg, usage, top_k=3)
    chunk_context = build_context_from_chunks(fallback_chunks)
    fact_context = "\n".join(f"- {fact}" for fact in facts)
    return f"GRAPH FACTS:\n{fact_context}\n\nSUPPORTING TEXT:\n{chunk_context}".strip()


def answer_with_llm(question: str, context: str, cfg: Config, usage: UsageMeter) -> str:
    client = get_openai_client(cfg.openai_api_key)
    if client is None:
        return extractive_answer(question, context)
    try:
        response = client.chat.completions.create(
            model=cfg.openai_model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the provided context. If the context is insufficient, say so. "
                        "Be concise and cite entity relationships when relevant."
                    ),
                },
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context[:12000]}"},
            ],
        )
        usage.add_chat_usage(response)
        return normalize_text(response.choices[0].message.content or "")
    except Exception as exc:
        print(f"[warn] LLM answer failed, using extractive fallback: {exc}")
        return extractive_answer(question, context)


def extractive_answer(question: str, context: str) -> str:
    question_terms = set(tokenize(question))
    sentences = split_sentences(context)
    scored = []
    for sentence in sentences:
        terms = set(tokenize(sentence))
        score = len(question_terms & terms)
        if score:
            scored.append((sentence, score))
    if not scored:
        return "Khong tim thay bang chung ro rang trong context."
    best = [sentence for sentence, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:4]]
    return " ".join(best)


def index_pipeline(cfg: Config) -> dict[str, Any]:
    started = time.perf_counter()
    usage = UsageMeter()
    docs = load_corpus(cfg.data_path)
    chunks = chunk_documents(docs, cfg.chunk_size_words, cfg.chunk_overlap_words, cfg.max_chunks_per_doc)
    if cfg.use_llm_extraction and cfg.openai_api_key:
        triples = extract_triples_llm(chunks, cfg, usage)
        if not triples:
            triples = extract_triples_rules(chunks)
    else:
        triples = extract_triples_rules(chunks)
    graph = build_graph(triples)

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    save_chunks(chunks, cfg.artifact_dir / "chunks.jsonl")
    save_triples(triples, cfg.artifact_dir / "triples.csv")
    save_graph(graph, cfg.artifact_dir / "knowledge_graph.json")
    visualize_graph(graph, cfg.artifact_dir / "knowledge_graph.png")

    metrics = {
        "documents": len(docs),
        "chunks": len(chunks),
        "triples": len(triples),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "index_seconds": round(time.perf_counter() - started, 3),
        **usage.as_dict(),
    }
    (cfg.artifact_dir / "index_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_or_build_artifacts(cfg: Config) -> tuple[list[Chunk], list[Triple], Any]:
    chunks_path = cfg.artifact_dir / "chunks.jsonl"
    triples_path = cfg.artifact_dir / "triples.csv"
    graph_path = cfg.artifact_dir / "knowledge_graph.json"
    if not chunks_path.exists() or not triples_path.exists() or not graph_path.exists():
        index_pipeline(cfg)
    return load_chunks(chunks_path), load_triples(triples_path), load_graph(graph_path)


def ask_flat(question: str, chunks: list[Chunk], cfg: Config) -> dict[str, Any]:
    usage = UsageMeter()
    started = time.perf_counter()
    scored = flat_retrieve(question, chunks, cfg, usage, top_k=5)
    context = build_context_from_chunks(scored)
    answer = answer_with_llm(question, context, cfg, usage)
    return {
        "question": question,
        "mode": "Flat RAG",
        "answer": answer,
        "context_chars": len(context),
        "seconds": round(time.perf_counter() - started, 3),
        **usage.as_dict(),
    }


def ask_graph(question: str, graph: Any, chunks: list[Chunk], cfg: Config) -> dict[str, Any]:
    usage = UsageMeter()
    started = time.perf_counter()
    context = build_context_from_graph(question, graph, chunks, cfg, usage)
    answer = answer_with_llm(question, context, cfg, usage)
    return {
        "question": question,
        "mode": "GraphRAG",
        "answer": answer,
        "context_chars": len(context),
        "seconds": round(time.perf_counter() - started, 3),
        **usage.as_dict(),
    }


def run_benchmark(cfg: Config, questions: list[str] | None = None) -> list[dict[str, Any]]:
    chunks, _, graph = load_or_build_artifacts(cfg)
    rows: list[dict[str, Any]] = []
    for question in questions or DEFAULT_QUESTIONS:
        rows.append(ask_flat(question, chunks, cfg))
        rows.append(ask_graph(question, graph, chunks, cfg))
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.artifact_dir / "benchmark_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["question", "mode", "answer", "context_chars", "seconds", "prompt_tokens", "completion_tokens", "embedding_tokens", "total_tokens"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_metrics(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab Day 19 - GraphRAG for Tech Company Corpus")
    parser.add_argument("command", choices=["index", "ask", "benchmark", "all"])
    parser.add_argument("--question", default="")
    parser.add_argument("--env", default=".env")
    args = parser.parse_args()

    cfg = Config.from_env(args.env)
    if args.command in {"index", "all"}:
        print_metrics(index_pipeline(cfg))
    if args.command == "ask":
        if not args.question:
            raise SystemExit("--question is required for ask")
        chunks, _, graph = load_or_build_artifacts(cfg)
        print(json.dumps(ask_graph(args.question, graph, chunks, cfg), ensure_ascii=False, indent=2))
    if args.command in {"benchmark", "all"}:
        rows = run_benchmark(cfg)
        print(f"wrote {len(rows)} benchmark rows to {cfg.artifact_dir / 'benchmark_results.csv'}")


if __name__ == "__main__":
    main()
