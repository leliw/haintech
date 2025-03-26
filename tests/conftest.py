import logging
import pytest


@pytest.fixture
def log():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("").setLevel(logging.INFO)
    logging.getLogger("haintech.pipeline").setLevel(logging.DEBUG)
    logging.getLogger("haintech.base_processor").setLevel(logging.DEBUG)
    logging.getLogger("markdown_pages_pre_processor").setLevel(logging.DEBUG)
