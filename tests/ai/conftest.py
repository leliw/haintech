import logging
from typing import Iterator

import pytest
from ampf.local import LocalFactory

from haintech.ai.model import AIChatSession, AIMultiagentSession


@pytest.fixture
def tm_session() -> Iterator[AIMultiagentSession]:
    factory = LocalFactory("./tests/data/test_sessions")
    storage = factory.create_storage("ai_log", AIMultiagentSession, "datetime")
    session = AIMultiagentSession()
    yield session
    # storage.save(session)
    # storage.drop()


@pytest.fixture
def ch_session() -> Iterator[AIChatSession]:
    factory = LocalFactory("./tests/data/test_sessions")
    storage = factory.create_storage("ai_log", AIMultiagentSession, "datetime")
    session = AIChatSession()
    yield session
    # storage.save(session)
    # storage.drop()


@pytest.fixture
def log() -> logging.Logger:
    logging.getLogger("haintech").setLevel(logging.DEBUG)
    ret = logging.getLogger("tests")
    ret.setLevel(logging.DEBUG)
    return ret
