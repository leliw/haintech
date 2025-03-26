from pathlib import Path

import pytest
from haintech.ai.open_ai import OpenAIChat


@pytest.fixture
def file_path(tmp_path: Path) -> Path:
    file_name = "test.md"
    file_path = tmp_path / file_name
    file_path.write_text(
        """# The Kiepski family
                         
* Ferdynand Kiepski: Head of the family, almost continuously unemployed father of two, who has a penchant for canned beer and trying to prove that he is an unappreciated genius.
* Halina Kiepska: Ferdek's wife, who works at a nearby hospital as a nurse. Every day she has to solve problems created by other members of the family. Addicted to cigarettes. 
* Waldemar Kiepski: the adult son of Ferdynand and Halina. Like his father he is unemployed.
* Mariola Kiepska (Barbara Mularczyk): Waldek’s younger sister. In the first several seasons she is an average teenager and likes to spend time outside the house - this usually involves going somewhere with her boyfriend Łysy ('Bald').
"""
    )
    return file_path

@pytest.mark.skip(reason="Next tests covers it.")
def test_chat_one_question(ai_chat: OpenAIChat, file_path: Path):
    # Given: A file added to a chat
    ai_chat.add_file(file_path)
    # And: A simple question
    q = "Who is a father of two?"
    # When: I ask model
    response = ai_chat.get_text_response(q)
    # Then: I should get answer
    assert "Ferdynand" in response
