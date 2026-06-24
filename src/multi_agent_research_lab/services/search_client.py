"""Search client abstraction for ResearcherAgent."""

import json
from urllib.error import URLError
from urllib.request import Request, urlopen

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client skeleton."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Uses Tavily when `TAVILY_API_KEY` is configured. Otherwise returns a
        deterministic local corpus that keeps the workflow testable offline.
        """

        if not self.settings.tavily_api_key:
            return self._local_search(query, max_results)

        try:
            return self._search_with_tavily(query, max_results)
        except (OSError, URLError, TimeoutError, json.JSONDecodeError):
            return self._local_search(query, max_results)

    def _search_with_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        request = Request(
            "https://api.tavily.com/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.settings.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = data.get("results", [])
        documents: list[SourceDocument] = []
        for item in results[:max_results]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "Untitled source")
            url = item.get("url")
            snippet = str(item.get("content") or item.get("snippet") or "")
            documents.append(
                SourceDocument(
                    title=title,
                    url=str(url) if url else None,
                    snippet=snippet,
                    metadata={"provider": "tavily"},
                )
            )
        return documents or self._local_search(query, max_results)

    @staticmethod
    def _local_search(query: str, max_results: int) -> list[SourceDocument]:
        corpus = [
            SourceDocument(
                title="Building effective agent workflows",
                url="https://www.anthropic.com/engineering/building-effective-agents",
                snippet=(
                    "Agent workflows work best when each step has a clear responsibility, "
                    "simple control flow, and measurable outputs."
                ),
                metadata={"provider": "local"},
            ),
            SourceDocument(
                title="LangGraph workflow concepts",
                url="https://langchain-ai.github.io/langgraph/concepts/",
                snippet=(
                    "Graph-based orchestration models agent state transitions as nodes, "
                    "edges, conditional routing, and explicit termination."
                ),
                metadata={"provider": "local"},
            ),
            SourceDocument(
                title="Production guardrails for LLM systems",
                url=None,
                snippet=(
                    "Useful guardrails include input validation, bounded iterations, timeout, "
                    "retry policy, fallbacks, trace logs, and benchmark-driven evaluation."
                ),
                metadata={"provider": "local"},
            ),
            SourceDocument(
                title="Single-agent versus multi-agent trade-offs",
                url=None,
                snippet=(
                    "Single-agent systems are simpler and cheaper; multi-agent systems can "
                    "improve separation of concerns, reviewability, and task decomposition."
                ),
                metadata={"provider": "local"},
            ),
        ]
        ranked = sorted(
            corpus,
            key=lambda doc: SearchClient._keyword_overlap(query, f"{doc.title} {doc.snippet}"),
            reverse=True,
        )
        return ranked[:max_results]

    @staticmethod
    def _keyword_overlap(query: str, text: str) -> int:
        query_terms = {term.lower().strip(".,:;!?()[]") for term in query.split()}
        text_terms = {term.lower().strip(".,:;!?()[]") for term in text.split()}
        return len(query_terms & text_terms)
