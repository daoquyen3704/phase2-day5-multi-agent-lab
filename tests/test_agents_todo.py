import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_is_student_todo() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    with pytest.raises(StudentTodoError):
        SupervisorAgent().run(state)
