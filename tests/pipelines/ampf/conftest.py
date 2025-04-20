import pytest
from ampf.in_memory import InMemoryFactory


@pytest.fixture
def factory():
    return InMemoryFactory()
