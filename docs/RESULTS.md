# Results

96 configurations, 114 questions, 14 documents. Recall@k and MRR use the top-k list; MRR is MRR@k. Answer metrics use the offline generator/judge unless noted (generator: `extractive`, judge: `heuristic`).


## Best configuration per retrieval method (k = 5)

| Retrieval | Chunking | Embedding | Recall@5 | MRR | Recall@1 | Correctness | Latency (ms) | Index (s) |
|---|---|---|---|---|---|---|---|---|
| Dense | paragraph | bge-small | 0.912 | 0.789 | 0.702 | 0.310 | 46.6 | 33.1 |
| BM25 | fixed-1000-o0 | none | 0.991 | 0.906 | 0.842 | 0.332 | 9.1 | 0.1 |
| Hybrid | fixed-200-o15 | bge-small | 0.991 | 0.919 | 0.868 | 0.341 | 43.9 | 41.6 |
| Hybrid + rerank | fixed-200-o15 | bge-small | 0.965 | 0.901 | 0.851 | 0.351 | 2298.3 | 41.6 |

## Chunking strategy (k = 5)

| Chunking | Chunks | Avg words | BM25 R@5 | BM25 MRR | Hybrid R@5 | Hybrid MRR | Hybrid R@5 factoid | Hybrid R@5 explanatory | Correctness |
|---|---|---|---|---|---|---|---|---|---|
| fixed-200-o0 | 186 | 193.8 | 0.965 | 0.859 | 0.983 | 0.873 | 0.986 | 0.977 | 0.344 |
| fixed-200-o15 | 219 | 192.6 | 0.983 | 0.903 | 0.991 | 0.910 | 1.000 | 0.977 | 0.347 |
| fixed-500-o0 | 79 | 456.2 | 0.991 | 0.890 | 0.983 | 0.896 | 0.971 | 1.000 | 0.338 |
| fixed-500-o15 | 91 | 459.5 | 1.000 | 0.902 | 0.991 | 0.899 | 0.986 | 1.000 | 0.344 |
| fixed-1000-o0 | 43 | 838.1 | 0.991 | 0.906 | 0.991 | 0.896 | 0.986 | 1.000 | 0.327 |
| fixed-1000-o15 | 47 | 872.1 | 0.991 | 0.898 | 0.991 | 0.909 | 0.986 | 1.000 | 0.323 |
| heading | 224 | 160.9 | 0.965 | 0.890 | 0.974 | 0.895 | 0.971 | 0.977 | 0.335 |
| paragraph | 156 | 231.0 | 0.991 | 0.879 | 0.965 | 0.879 | 0.957 | 0.977 | 0.327 |

## Embedding model (fixed-200-o15, k = 5)

| Embedding | Dense R@5 | Dense MRR | Hybrid R@5 | Hybrid MRR | Index (s) | Query (ms) | $ / 1k q |
|---|---|---|---|---|---|---|---|
| tfidf | 0.974 | 0.852 | 0.983 | 0.871 | 0.4 | 1.8 | $0.0 |
| minilm | 0.860 | 0.704 | 0.991 | 0.910 | 34.2 | 29.2 | $0.0 |
| bge-small | 0.939 | 0.774 | 0.991 | 0.919 | 41.5 | 38.2 | $0.0 |
| e5-base | 0.912 | 0.718 | 0.991 | 0.908 | 121.2 | 114.2 | $0.0 |

## Top-k (fixed-200-o15 · bge-small · Hybrid)

| k | Recall@k | MRR | Correctness | Latency (ms) |
|---|---|---|---|---|
| 3 | 0.974 | 0.915 | 0.341 | 43.2 |
| 5 | 0.991 | 0.919 | 0.341 | 43.9 |
| 10 | 1.000 | 0.921 | 0.341 | 45.9 |

## Does the reranker help? (hybrid vs hybrid + rerank, k = 5)

| Chunking | Embedding | Hybrid MRR | +Rerank MRR | Δ MRR | Hybrid R@1 | +Rerank R@1 | Hybrid ms | +Rerank ms |
|---|---|---|---|---|---|---|---|---|
| fixed-200-o15 | bge-small | 0.919 | 0.901 | -0.018 | 0.868 | 0.851 | 44 | 2298 |
| paragraph | bge-small | 0.904 | 0.885 | -0.019 | 0.851 | 0.825 | 42 | 2311 |
