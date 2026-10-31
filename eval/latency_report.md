# Latency breakdown report

Runs per question: 1. Total runs: 26.

## Overall mean latency by stage (seconds)

| Stage | Mean seconds |
|---|---|
| router | 4.02 |
| sql_tool | 21.15 |
| rag_tool | 10.31 |
| synthesizer | 18.66 |
| total | 46.95 |

### Share of total time per stage

- **router**: 4.02s (8.6% of total)
- **sql_tool**: 21.15s (45.1% of total)
- **rag_tool**: 10.31s (22.0% of total)
- **synthesizer**: 18.66s (39.7% of total)

## Mean latency by route type (seconds)

| Route | Router | SQL tool | RAG tool | Synthesizer | Total |
|---|---|---|---|---|---|
| sql | 4.02 | 20.87 | n/a | 15.16 | 40.06 |
| both | 4.06 | 21.47 | 10.67 | 25.52 | 61.74 |
| rag | 3.86 | n/a | 8.99 | 7.48 | 20.34 |

## Notes on the sequential "both" route

The 'both' route (sql_tool then rag_tool sequentially, per the Phase 2 design decision) averages 61.74s total, versus 40.06s for sql-only and 20.34s for rag-only. This is the concrete cost of the sequential-not-parallel choice made in backend/graph.py — worth citing directly in the report's discussion of that design trade-off.