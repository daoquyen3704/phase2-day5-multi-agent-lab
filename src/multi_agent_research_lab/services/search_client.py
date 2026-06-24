"""Search client abstraction for ResearcherAgent.

Uses a mock search source with realistic content for the lab.
Students with a Tavily API key can switch to live search.
"""

import logging
import os

from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock knowledge base – realistic research snippets covering common AI topics
# ---------------------------------------------------------------------------

_MOCK_KB: list[dict] = [
    {
        "title": "GraphRAG: Unlocking LLM discovery on narrative private data",
        "url": "https://microsoft.com/research/blog/graphrag",
        "snippet": (
            "GraphRAG is a structured, hierarchical approach to Retrieval Augmented Generation "
            "(RAG), as opposed to naive semantic-search approaches using plain text snippets. "
            "GraphRAG uses an LLM to build a knowledge graph from private data, then uses that "
            "graph at query time to reason across community summaries and local entity clusters. "
            "Benchmarks show 2-3× improvements on complex multi-hop questions compared to vector RAG."
        ),
        "tags": ["graphrag", "rag", "knowledge graph", "microsoft", "llm"],
    },
    {
        "title": "From RAG to GraphRAG – Architecture Deep Dive",
        "url": "https://arxiv.org/abs/2404.16130",
        "snippet": (
            "The paper introduces a graph-based indexing phase that extracts entities and "
            "relationships, groups them into communities via Leiden algorithm, and pre-computes "
            "summaries. At query time, a two-level retrieval (global community summaries + local "
            "entity subgraphs) dramatically improves holistic sensemaking tasks vs classic RAG. "
            "Evaluation uses datasets from the TREC question answering benchmark."
        ),
        "tags": ["graphrag", "rag", "community detection", "leiden", "arxiv"],
    },
    {
        "title": "Multi-Agent Systems: State of the Art 2024",
        "url": "https://arxiv.org/abs/2402.01680",
        "snippet": (
            "Modern multi-agent LLM frameworks employ orchestration patterns such as "
            "Supervisor→Worker, Hierarchical, and Collaborative. Key challenges include "
            "shared state management, avoiding agent hallucination loops, and enforcing "
            "guardrails (max_iterations, timeouts). LangGraph, CrewAI, and AutoGen represent "
            "the main open-source frameworks in 2024."
        ),
        "tags": ["multi-agent", "langgraph", "crewai", "autogen", "orchestration"],
    },
    {
        "title": "Retrieval Augmented Generation for Knowledge-Intensive NLP Tasks",
        "url": "https://arxiv.org/abs/2005.11401",
        "snippet": (
            "The original RAG paper (Lewis et al., 2020) combines a dense retriever (DPR) with "
            "a seq2seq generator (BART). Retrieval is done over Wikipedia passages stored in a "
            "FAISS index. The approach outperforms parametric-only models on open-domain QA, "
            "dialogue, and fact verification tasks. Foundational for all modern RAG variants."
        ),
        "tags": ["rag", "retrieval", "nlp", "dpr", "faiss"],
    },
    {
        "title": "LangGraph: Building Stateful Multi-Agent Applications",
        "url": "https://langchain-ai.github.io/langgraph/concepts/",
        "snippet": (
            "LangGraph extends LangChain with a graph-based execution model supporting cycles, "
            "branching, and persistence. Nodes are Python functions or agents; edges are either "
            "static or conditional. StateGraph allows shared typed state across all nodes. "
            "Supports interrupts, human-in-the-loop, and streaming out of the box."
        ),
        "tags": ["langgraph", "multi-agent", "graph", "state", "langchain"],
    },
    {
        "title": "Agentic AI Design Patterns – Anthropic Engineering Blog",
        "url": "https://www.anthropic.com/engineering/building-effective-agents",
        "snippet": (
            "Anthropic identifies five core agentic patterns: prompt chaining, routing, "
            "parallelization, orchestrator-subagent, and evaluator-optimizer. The blog "
            "recommends starting simple and only adding multi-agent complexity when a single "
            "agent demonstrably underperforms. Guardrails (max_steps, tool limits, human "
            "checkpoints) are essential for production deployments."
        ),
        "tags": ["agents", "anthropic", "design patterns", "routing", "orchestrator"],
    },
    {
        "title": "Vector Databases Comparison 2024: Pinecone vs Weaviate vs Qdrant",
        "url": "https://benchmark.vectordb.com/2024",
        "snippet": (
            "ANN benchmarks show Qdrant achieving the best QPS/accuracy trade-off at high "
            "dimensions. Pinecone leads in managed offering UX. Weaviate supports hybrid "
            "BM25+vector natively. Chroma is preferred for local development. Choice depends "
            "on scale, hosting preference, and need for hybrid retrieval."
        ),
        "tags": ["vector database", "pinecone", "weaviate", "qdrant", "chroma", "rag"],
    },
    {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "url": "https://arxiv.org/abs/2210.03629",
        "snippet": (
            "ReAct interleaves reasoning traces (Thought) with action execution (Act) and "
            "observation collection (Obs) in a loop. This allows LLMs to dynamically plan, "
            "use tools, and update plans based on new information. ReAct outperforms "
            "chain-of-thought on HotpotQA, FEVER, and ALFWorld benchmarks."
        ),
        "tags": ["react", "reasoning", "agents", "tool use", "hotpotqa"],
    },
    {
        "title": "Hallucination in LLMs: Causes and Mitigations",
        "url": "https://arxiv.org/abs/2309.01219",
        "snippet": (
            "LLM hallucinations stem from training data gaps, distribution shift, and "
            "over-confident sampling. Mitigations include RAG (grounding in retrieved facts), "
            "self-consistency voting, confidence calibration, and critic agents that "
            "fact-check outputs before delivery to users. Multi-agent critic patterns "
            "reduce factual errors by 20-40% on benchmarks."
        ),
        "tags": ["hallucination", "rag", "critic", "reliability", "fact-check"],
    },
    {
        "title": "Prompt Engineering Best Practices for Production LLMs",
        "url": "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering",
        "snippet": (
            "Effective prompts specify role, task, context, format, and constraints explicitly. "
            "System prompts should be stable; user prompts dynamic. Chain-of-thought (XML tags "
            "for scratchpad) improves reasoning quality. Output schemas (JSON/Pydantic) enforce "
            "structured responses. Length calibration reduces cost without sacrificing quality."
        ),
        "tags": ["prompt engineering", "system prompt", "chain-of-thought", "anthropic"],
    },
]


class SearchClient:
    """Provider-agnostic search client.

    Defaults to a curated mock knowledge base so the lab works offline.
    Set TAVILY_API_KEY in environment to enable live Tavily search.
    """

    def __init__(self) -> None:
        self._tavily_key = os.getenv("TAVILY_API_KEY")
        if self._tavily_key:
            logger.info("SearchClient: using Tavily live search")
        else:
            logger.info("SearchClient: using mock knowledge base (set TAVILY_API_KEY for live search)")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self._tavily_key:
            return self._tavily_search(query, max_results)
        return self._mock_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Keyword-based mock search over the built-in knowledge base."""
        query_words = set(query.lower().split())
        scored: list[tuple[int, dict]] = []

        for doc in _MOCK_KB:
            score = 0
            text = (doc["title"] + " " + doc["snippet"]).lower()
            for word in query_words:
                if word in text:
                    score += 1
            for tag in doc.get("tags", []):
                if any(word in tag for word in query_words):
                    score += 2
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = scored[:max_results]

        if not results:
            # Return first max_results as fallback
            results = [(0, doc) for doc in _MOCK_KB[:max_results]]

        return [
            SourceDocument(
                title=doc["title"],
                url=doc.get("url"),
                snippet=doc["snippet"],
                metadata={"mock": True, "score": score, "tags": doc.get("tags", [])},
            )
            for score, doc in results
        ]

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Live search via Tavily API."""
        try:
            from tavily import TavilyClient  # type: ignore[import-untyped]

            client = TavilyClient(api_key=self._tavily_key)
            resp = client.search(query=query, max_results=max_results)
            return [
                SourceDocument(
                    title=r.get("title", ""),
                    url=r.get("url"),
                    snippet=r.get("content", ""),
                    metadata={"score": r.get("score", 0)},
                )
                for r in resp.get("results", [])
            ]
        except Exception as exc:
            logger.warning("Tavily search failed (%s), falling back to mock", exc)
            return self._mock_search(query, max_results)