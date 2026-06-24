# Failure Modes And Fixes

This note summarizes the main failure modes observed or expected in the
multi-agent research lab implementation, plus the mitigation used in the code.

## Trace Artifact

The latest multi-agent trace is available at [trace_latest.json](trace_latest.json).

## 1. Missing Or Invalid LLM API Key

Impact:

- Real provider calls may fail before the system can produce an answer.
- A classroom demo could become blocked by local environment setup.

Current mitigation:

- `LLMClient` uses a deterministic local fallback when no OpenAI-style key is
  configured.
- Provider calls retry before fallback.

Recommended improvement:

- Record provider fallback explicitly in `state.errors` or agent metadata so the
  benchmark does not confuse fallback output with a successful real-LLM run.

## 2. Search Provider Failure

Impact:

- Researcher may fail to collect evidence.
- Final answer may become unsupported or too generic.

Current mitigation:

- `SearchClient` uses a local source corpus when `TAVILY_API_KEY` is missing.
- If the provider fails, `SearchClient` falls back to the local source corpus so
  the workflow can still complete.

Recommended improvement:

- Cache successful search results per benchmark query to make runs reproducible.

## 3. Multi-Agent Benchmark Advantage

Impact:

- The multi-agent system currently has access to retrieved sources, while the
  single-call baseline answers directly from the query.
- A higher multi-agent score may reflect better source access rather than better
  decomposition.

Current mitigation:

- The benchmark report states this limitation.

Recommended improvement:

- Add a controlled baseline that receives the same source pack.
- Compare four conditions: single-call no-source, single-call with sources,
  controlled multi-call with sources, and multi-agent with sources.

## 4. Infinite Or Repeated Routing

Impact:

- A Supervisor bug could loop forever or waste tokens.

Current mitigation:

- `MAX_ITERATIONS` bounds the route loop.
- If the limit is reached, the Supervisor routes to Writer fallback and records
  an error.

Recommended improvement:

- Add an explicit route validation test for unknown routes and max-iteration
  fallback.

## 5. Citation Illusion

Impact:

- The final answer may include references without proving that each claim is
  actually supported by those sources.
- Citation coverage can be gamed by appending a reference list.

Current mitigation:

- Writer appends source references and benchmark measures citation index
  coverage.

Recommended improvement:

- Add a Critic or evaluator pass that checks claim-to-source support, not just
  presence of citations.

## 6. Fallback Output Looks Too Successful

Impact:

- Local fallback produces stable output with zero cost, which is useful for
  tests but can make benchmark results look better than a real provider run.

Current mitigation:

- Fallback behavior is deterministic and documented.

Recommended improvement:

- Add a `provider_mode` field to each agent result, such as `openai`, `tavily`,
  or `local_fallback`, and include it in the benchmark report.

## 7. Context Loss Between Agents

Impact:

- Researcher may collect useful details that Analyst or Writer fails to use.
- Final answer may be generic even when sources are strong.

Current mitigation:

- Shared state stores sources, research notes, analysis notes, and agent
  results.

Recommended improvement:

- Make Analyst and Writer prompts require explicit reference to source indexes
  and key claims.
