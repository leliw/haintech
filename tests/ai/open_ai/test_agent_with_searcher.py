from typing import List

from haintech.ai import AIChatSession, BaseRAGSearcher, RAGItem
from haintech.ai.open_ai.open_ai_agent import OpenAIAgent


class FileRAGSearcher(BaseRAGSearcher):
    def search_sync(self, query: str) -> List[RAGItem]:
        if "Who is a father of two?" == query:
            return [
                RAGItem(
                    title="Ferdynand Kiepski",
                    content="""# The Kiepski family
                         
* Ferdynand Kiepski: Head of the family, almost continuously unemployed father of two, who has a penchant for canned beer and trying to prove that he is an unappreciated genius.
* Halina Kiepska: Ferdek's wife, who works at a nearby hospital as a nurse. Every day she has to solve problems created by other members of the family. Addicted to cigarettes. 
* Waldemar Kiepski: the adult son of Ferdynand and Halina. Like his father he is unemployed.
* Mariola Kiepska (Barbara Mularczyk): Waldek’s younger sister. In the first several seasons she is an average teenager and likes to spend time outside the house - this usually involves going somewhere with her boyfriend Łysy ('Bald').
""",
                )
            ]
        else:
            return []


def test_agent_one_question():
    # Given: An agent with session and searcher
    searcher = FileRAGSearcher()
    session = AIChatSession()
    ai_agent = OpenAIAgent(searcher=searcher, session=session)
    # And: A simple question
    q = "Who is a father of two?"
    # When: I ask model
    response = ai_agent.get_text_response(q)
    # Then: I should get answer rturned by searcher
    assert "Ferdynand" in response
