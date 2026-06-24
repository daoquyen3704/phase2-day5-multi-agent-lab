from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


def test_llm_client_fallback_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()

    response = LLMClient().complete("You are a writer.", "Explain agent workflows")

    assert response.content
    assert response.cost_usd == 0.0

    get_settings.cache_clear()


def test_search_client_fallback_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "")
    get_settings.cache_clear()

    results = SearchClient().search("multi-agent guardrails", max_results=2)

    assert len(results) == 2
    assert all(result.title for result in results)

    get_settings.cache_clear()
