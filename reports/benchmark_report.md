# Benchmark Report

## Summary

Recorded 8 runs; 8 completed without errors. Best quality: `multi-agent: You are designing a research exp`. Fastest: `baseline: Compare single-agent and multi-agent`.

## Trace Artifact

The latest multi-agent trace is available at [trace_latest.json](trace_latest.json).

## Metrics

| Run | Latency (s) | Cost (USD) | Quality | Citation Coverage | Errors | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline: You are designing a research experim | 0.00 | 0.0000 | 3.0 |  | 0 | routes: none |
| multi-agent: You are designing a research exp | 1.86 | 0.0000 | 10.0 | 100% | 0 | 4 sources; routes: researcher, analyst, writer, done |
| baseline: Research GraphRAG state-of-the-art a | 0.00 | 0.0000 | 3.0 |  | 0 | routes: none |
| multi-agent: Research GraphRAG state-of-the-a | 1.22 | 0.0000 | 10.0 | 100% | 0 | 5 sources; routes: researcher, analyst, writer, done |
| baseline: Compare single-agent and multi-agent | 0.00 | 0.0000 | 3.0 |  | 0 | routes: none |
| multi-agent: Compare single-agent and multi-a | 1.16 | 0.0000 | 10.0 | 100% | 0 | 5 sources; routes: researcher, analyst, writer, done |
| baseline: Summarize production guardrails for  | 0.00 | 0.0000 | 3.0 |  | 0 | routes: none |
| multi-agent: Summarize production guardrails  | 1.47 | 0.0000 | 10.0 | 100% | 0 | 5 sources; routes: researcher, analyst, writer, done |

## Methodological Notes

- The current demo baseline answers directly from the query, while the multi-agent workflow may retrieve sources. Treat quality differences as a lab signal, not a definitive architecture claim.
- A stricter experiment should give both systems the same cached source pack, model, answer length, and total token budget.
- To separate decomposition gains from extra inference time, add a controlled multi-call baseline with the same number of calls and token budget as the multi-agent workflow.
- Local fallback responses are useful for reproducible tests, but real provider runs should report provider mode and actual token/cost usage.
