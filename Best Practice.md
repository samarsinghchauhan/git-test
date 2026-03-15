Here are the practices that separate production-grade RAG systems from proof-of-concept ones — organised by the phase in which they apply.---

## Phase 1 — Preparation: what you embed matters more than how

**Clean before you chunk, not after.** Boilerplate text — headers, footers, navigation links, repeated disclaimers, table formatting artefacts — pollutes every chunk it appears in. A policy document where every page starts with *"CONFIDENTIAL — Internal Use Only — ABC Bank © 2024"* will embed that phrase into the semantic neighbourhood of every rule, dragging unrelated content closer together. Strip it at the document level before chunking, not per-chunk.

**Preserve document structure as metadata, not content.** The heading hierarchy (`Section 3 > 3.2 > 3.2.1`) tells you where a chunk lives semantically, but embedding it as raw text inside the chunk competes with the actual content for vector space. Instead, store it as metadata alongside the vector and use it as a filter or re-ranking signal at retrieval time.

```python
{
  "chunk_id": "bcbs239_s3_2_1_chunk_4",
  "text": "Data lineage records must be maintained for a minimum of...",
  "metadata": {
    "document": "BCBS 239",
    "section": "3.2.1",
    "heading": "Data lineage retention",
    "doc_type": "regulatory",
    "effective_date": "2024-01-01",
    "embedding_model": "text-embedding-3-small",
    "embedded_at": "2025-03-01"
  }
}
```

**Match chunk size to your query type, not to convention.** The commonly repeated advice of "200–300 tokens" is a starting point, not a universal truth. If your users ask short factual questions ("what is the null threshold for column X?"), small dense chunks win — the answer fits in one chunk and retrieval precision is high. If they ask open-ended synthesis questions ("summarise all credit risk rules for premium customers"), larger chunks that preserve more context produce better LLM answers because fewer retrieved chunks are needed to cover the topic. Benchmark both on your actual query distribution before committing.

**Use parent-child chunking for the best of both worlds.** Index small child chunks (100–150 tokens) for high-precision retrieval, but return the full parent chunk (400–600 tokens) to the LLM for context-rich generation. This is sometimes called the "small-to-big" pattern and it is consistently one of the highest-impact structural changes you can make to a RAG system.

```
Parent chunk (400 tokens) → stored for LLM context
    ├── Child chunk A (120 tokens) → indexed for retrieval
    ├── Child chunk B (130 tokens) → indexed for retrieval
    └── Child chunk C (110 tokens) → indexed for retrieval
```

When child chunk B is retrieved, the system returns its parent to the LLM — giving it the surrounding context that makes B interpretable.

---

## Phase 2 — Encoding: the model choice is a contract

**Domain mismatch is your biggest quality risk.** General-purpose models (`text-embedding-3-small`, `all-MiniLM`) are trained on web text. Banking DQ corpora contain dense regulatory language, SQL, column name conventions, and acronyms (`BCBS`, `LGD`, `NPA`, `CUST_ACCT_BAL`) that these models have never encountered in meaningful training volume. The model will embed `NPA` (Non-Performing Asset) in the same neighbourhood as general uses of the acronym. Fine-tuning on even a few thousand domain-specific query-document pairs measurably closes this gap.

**Always use asymmetric encoding when the model supports it.** As covered in the previous session — queries and documents have different linguistic structure and should be encoded differently. With `E5` and `BGE` family models this is a prefix convention. With bi-encoder fine-tuning it means training separate query and document tower weights.

**Embed the question, not just the answer.** For FAQ-style or structured knowledge bases, embed synthetic questions that the chunk answers, not just the chunk text itself. A chunk describing Rule R07 embeds well as a rule description, but it embeds even better if you also store a vector for *"which accounts are flagged for credit limit breach?"* pointing to that chunk. This technique — called **HyDE (Hypothetical Document Embeddings)** in reverse — dramatically improves retrieval for question-type queries.

```python
# At index time: generate synthetic questions for each chunk
synthetic_qs = llm.generate(f"""
    Generate 3 questions that this text directly answers:
    {chunk_text}
""")
# Embed and index both the chunk and its synthetic questions
# pointing to the same chunk_id
```

---

## Phase 3 — Indexing: your index is a versioned artifact

**Treat the vector index like a database schema — version it.** When you update the embedding model, change chunking strategy, or re-clean the corpus, the index changes fundamentally. Every vector must be stamped with the model ID and version that produced it. Your retrieval layer must refuse to mix vectors from different model versions in the same search. This sounds obvious but is skipped in almost every initial implementation.

**Separate hot and cold index tiers.** Not all documents are queried equally. Frequently accessed rules, active regulatory documents, and recent failure patterns deserve a hot tier with a fast in-memory index (HNSW in Qdrant, Pinecone, or pgvector). Archival content — historical rule versions, old test results — can live in a cold tier with higher latency and lower cost. Query routing logic decides which tier or both to hit based on metadata filters.

```
Hot tier  → HNSW index, in-memory, low latency
              active rules · current regulations · recent failures

Cold tier → IVF-PQ index, on-disk, higher latency
              historical versions · archived docs · old test runs
```

**Choose your ANN algorithm based on your update pattern.** HNSW (Hierarchical Navigable Small World) gives the best query-time performance but expensive index updates — every new vector insertion modifies graph edges. IVF (Inverted File Index) is cheaper to update but requires periodic full re-clustering as the corpus grows. For a DQ knowledge base that ingests new rules frequently, a hybrid: HNSW for the hot tier, IVF-PQ for the archive.

**Plan re-embedding capacity upfront.** Re-indexing 10 million chunks is a multi-hour Spark job. Budget for it at every model upgrade. The re-embedding cadence should be part of your MLOps runbook, not a surprise.

---

## Phase 4 — Retrieval: single-vector search is almost never enough

**Hybrid search is the baseline, not an optimisation.** Dense vector search alone misses exact-match queries. BM25 alone misses semantic queries. In production systems, hybrid search (dense + sparse, fused with Reciprocal Rank Fusion) consistently outperforms either alone by 10–20% on recall@10 across benchmark datasets. This was covered in the retrieval methods session — but the key practice point is: measure your system *without* BM25 before assuming you don't need it.

**Query expansion before encoding.** Short queries lose information when compressed into a single vector. Expanding the query before embedding — either via LLM paraphrase, HyDE (generate a hypothetical ideal answer and embed that), or synonym expansion — gives the embedding more surface area to match against. For the DQ assistant, a steward asking *"why is txn_amount failing?"* expands to a richer representation that catches rules, lineage paths, and historical failures simultaneously.

```python
def expand_query(query: str, llm) -> list[str]:
    expansions = llm.generate(f"""
        Rephrase this question in 3 different ways that preserve meaning:
        {query}
    """)
    return [query] + expansions  # embed all, average or take best-scoring vector
```

**Re-rank after retrieval, always.** The top-k from ANN search is a coarse shortlist, not the final answer. A cross-encoder re-ranker (models like `cross-encoder/ms-marco-MiniLM-L-6-v2`) reads the full (query, chunk) pair together — not as separate vectors — and produces a much more precise relevance score. Apply it to your top-50 ANN results and return the top-5 to the LLM. The cost is worthwhile: cross-encoders are too slow for full-corpus search but fast enough for 50 candidates.

**Apply metadata filters before vector search, not after.** Filtering after retrieval wastes ANN capacity — you retrieve 100 candidates, 60 fail the filter, and the LLM gets 40 instead of 100. Pre-filtering by metadata (document type, date range, domain, rule severity) before the ANN search shrinks the search space and improves both precision and latency. Most modern vector databases (Qdrant, Weaviate, Pinecone) support pre-filtered ANN natively.

```python
results = vector_db.search(
    query_vector = query_vec,
    top_k        = 20,
    filters      = {
        "doc_type":   {"$in": ["rule", "regulation"]},
        "domain":     "credit_risk",
        "effective_date": {"$gte": "2023-01-01"}
    }
)
```

**Contextual compression before passing to the LLM.** Even a well-retrieved chunk contains sentences irrelevant to the specific query. Passing the full chunk to the LLM wastes context window and dilutes the signal. A compression step — using a small LLM to extract only the query-relevant sentences from each retrieved chunk — can reduce context window usage by 40–60% with no loss in answer quality. LangChain calls this `ContextualCompressionRetriever`; it is underused.

---

## Phase 5 — Evaluation and operations: what you can't measure, you can't improve

**Measure retrieval and generation separately.** Most teams measure end-to-end answer quality (did the LLM say something correct?) but not retrieval quality in isolation. A bad LLM answer might be caused by bad retrieval, bad generation, or both — and you cannot fix what you haven't isolated. The key retrieval metrics to track are Recall@k (is the ground-truth chunk in the top-k?), MRR (Mean Reciprocal Rank), and NDCG (Normalized Discounted Cumulative Gain).

```
Retrieval metrics         Generation metrics
────────────────          ──────────────────
Recall@5, Recall@10       Faithfulness (is answer grounded in chunks?)
MRR                       Answer relevance
NDCG@10                   Context precision (how much retrieved context was used?)
Mean similarity score     Hallucination rate
```

**Monitor embedding drift in production.** As your corpus grows and users interact with the system, track the distribution of retrieved chunk similarity scores over time. A drop in average top-1 similarity score signals either query distribution shift (users are asking about things not well-represented in the index) or corpus gap (new content is poorly covered). Set alerts on this metric the same way you would on API latency.

**Build a golden evaluation set on day one.** Before going to production, assemble 100–200 representative (query, expected chunk, expected answer) triples covering your key use cases. Run this set automatically on every index change, model update, or chunking strategy change. Without a golden set, you are flying blind — every change is a guess about whether things got better or worse.

**Log what was retrieved, not just what was answered.** Every LLM response should carry the `chunk_ids` that grounded it. This gives you three capabilities: citation display for the user, post-hoc debugging when an answer is wrong, and a feedback loop for improving retrieval (if users mark an answer as wrong, you know exactly which chunks misled the model).

---

Bringing this together into a single decision hierarchy:

| Priority | Practice | Impact if skipped |
|---|---|---|
| 1 | Parent-child chunking | Poor recall on synthesis queries |
| 2 | Asymmetric query/doc encoding | ~10–15% retrieval quality loss |
| 3 | Hybrid search (dense + BM25) | Exact-match queries fail silently |
| 4 | Cross-encoder re-ranking | Top-5 to LLM is noisy |
| 5 | Metadata pre-filtering | Wasted ANN capacity, slower search |
| 6 | Vector versioning with model ID | Silent corruption after model upgrade |
| 7 | Golden eval set | No way to know if changes help |
| 8 | Separate retrieval metrics | Can't diagnose whether problem is retrieval or generation |

The first four are foundational — a RAG system without them is fragile regardless of which embedding model or vector database you choose. The last four are what separate systems that degrade gracefully from those that fail mysteriously.
