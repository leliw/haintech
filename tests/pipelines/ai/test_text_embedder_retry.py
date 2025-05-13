import asyncio
import pytest # Changed from unittest
import httpx

# Using mocker for patching and creating some mock objects
# from unittest.mock import AsyncMock, MagicMock # No longer needed if using mocker's versions

from haintech.pipelines.ai.text_embedder import TextEmbedder, Vector
from haintech.ai.base import BaseAITextEmbeddingModel


# Common test data
TEST_DATA_INPUT: str = "This is a test sentence."
EXPECTED_VECTOR_OUTPUT: Vector = [0.1, 0.2, 0.3, 0.4, 0.5]
BASE_WAIT_TIME = 0.1
# Default max_retries for most tests, can be overridden in specific tests
DEFAULT_MAX_RETRIES = 2


@pytest.fixture
def mock_ai_model(mocker):
    """Fixture to create a mock AI model."""
    return mocker.AsyncMock(spec=BaseAITextEmbeddingModel)


@pytest.fixture
def text_embedder_instance(mock_ai_model):
    """Fixture to create a TextEmbedder instance with default settings for testing."""
    embedder = TextEmbedder(ai_model=mock_ai_model, max_concurrent=1)
    # Override random wait_time_seconds for predictable testing
    embedder.wait_time_seconds = BASE_WAIT_TIME
    embedder.max_retries = DEFAULT_MAX_RETRIES  # Total 3 attempts (1 initial + 2 retries)
    return embedder


@pytest.mark.asyncio
async def test_process_item_success_first_try(text_embedder_instance, mock_ai_model):
    """Test successful embedding on the first attempt."""
    mock_ai_model.get_embedding_async.return_value = EXPECTED_VECTOR_OUTPUT

    result = await text_embedder_instance.process_item(TEST_DATA_INPUT)

    assert result == EXPECTED_VECTOR_OUTPUT
    mock_ai_model.get_embedding_async.assert_called_once_with(TEST_DATA_INPUT)


@pytest.mark.asyncio
async def test_process_item_success_after_one_retry(text_embedder_instance, mock_ai_model, mocker):
    """Test successful embedding after one retry due to ReadTimeout."""
    mock_log = mocker.patch('haintech.pipelines.ai.text_embedder.TextEmbedder._log')
    # Use mocker.AsyncMock for new_callable if available and preferred
    mock_sleep = mocker.patch('haintech.pipelines.ai.text_embedder.asyncio.sleep', new_callable=mocker.AsyncMock)

    timeout_exception = httpx.ReadTimeout("Test ReadTimeout", request=None)
    mock_ai_model.get_embedding_async.side_effect = [
        timeout_exception,
        EXPECTED_VECTOR_OUTPUT
    ]

    result = await text_embedder_instance.process_item(TEST_DATA_INPUT)

    assert result == EXPECTED_VECTOR_OUTPUT
    assert mock_ai_model.get_embedding_async.call_count == 2
    mock_ai_model.get_embedding_async.assert_any_call(TEST_DATA_INPUT)
    
    # Check sleep: called once, with wait_time * 2^0
    mock_sleep.assert_called_once_with(text_embedder_instance.wait_time_seconds * pow(2, 0))
    
    # Check logging: one warning
    expected_log_msg = (
        f"Attempt 1 of {text_embedder_instance.max_retries + 1} failed with timeout: {timeout_exception}. "
        f"Retrying in {text_embedder_instance.wait_time_seconds * pow(2, 0):.2f} seconds..."
    )
    mock_log.warning.assert_called_once_with(expected_log_msg)
    mock_log.error.assert_not_called()


@pytest.mark.asyncio
async def test_process_item_fail_after_all_retries(text_embedder_instance, mock_ai_model, mocker):
    """Test failure after all retries due to persistent ReadTimeout."""
    mock_log = mocker.patch('haintech.pipelines.ai.text_embedder.TextEmbedder._log')
    mock_sleep = mocker.patch('haintech.pipelines.ai.text_embedder.asyncio.sleep', new_callable=mocker.AsyncMock)

    timeout_exception = httpx.ReadTimeout("Test ReadTimeout", request=None)
    mock_ai_model.get_embedding_async.side_effect = timeout_exception

    with pytest.raises(httpx.ReadTimeout) as exc_info:
        await text_embedder_instance.process_item(TEST_DATA_INPUT)
    
    assert exc_info.value is timeout_exception # Ensure the original exception is raised

    assert mock_ai_model.get_embedding_async.call_count == text_embedder_instance.max_retries + 1 # 3 attempts
    
    # Check sleep calls
    assert mock_sleep.call_count == text_embedder_instance.max_retries # 2 sleep calls
    mock_sleep.assert_any_call(text_embedder_instance.wait_time_seconds * pow(2, 0)) # After 1st failure
    mock_sleep.assert_any_call(text_embedder_instance.wait_time_seconds * pow(2, 1)) # After 2nd failure
    
    # Check logging
    assert mock_log.warning.call_count == text_embedder_instance.max_retries # 2 warnings
    expected_log_msg_1 = (
        f"Attempt 1 of {text_embedder_instance.max_retries + 1} failed with timeout: {timeout_exception}. "
        f"Retrying in {text_embedder_instance.wait_time_seconds * pow(2, 0):.2f} seconds..."
    )
    expected_log_msg_2 = (
        f"Attempt 2 of {text_embedder_instance.max_retries + 1} failed with timeout: {timeout_exception}. "
        f"Retrying in {text_embedder_instance.wait_time_seconds * pow(2, 1):.2f} seconds..."
    )
    mock_log.warning.assert_any_call(expected_log_msg_1)
    mock_log.warning.assert_any_call(expected_log_msg_2)
    
    mock_log.error.assert_called_once_with(
        f"All {text_embedder_instance.max_retries + 1} attempts to get embedding failed with timeout."
    )


@pytest.mark.asyncio
async def test_process_item_other_exception_propagates(text_embedder_instance, mock_ai_model, mocker):
    """Test that non-ReadTimeout exceptions propagate immediately."""
    mock_log = mocker.patch('haintech.pipelines.ai.text_embedder.TextEmbedder._log')
    mock_sleep = mocker.patch('haintech.pipelines.ai.text_embedder.asyncio.sleep', new_callable=mocker.AsyncMock)

    other_exception = ValueError("Some other AI model error")
    mock_ai_model.get_embedding_async.side_effect = other_exception

    with pytest.raises(ValueError) as exc_info:
        await text_embedder_instance.process_item(TEST_DATA_INPUT)

    assert exc_info.value is other_exception
    mock_ai_model.get_embedding_async.assert_called_once_with(TEST_DATA_INPUT)
    mock_sleep.assert_not_called()
    mock_log.warning.assert_not_called()
    mock_log.error.assert_not_called()


@pytest.mark.asyncio
async def test_process_item_max_retries_zero_fail(text_embedder_instance, mock_ai_model, mocker):
    """Test failure on first attempt if max_retries is 0."""
    mock_log = mocker.patch('haintech.pipelines.ai.text_embedder.TextEmbedder._log')
    mock_sleep = mocker.patch('haintech.pipelines.ai.text_embedder.asyncio.sleep', new_callable=mocker.AsyncMock)

    # Override max_retries for this specific test
    text_embedder_instance.max_retries = 0 # Total 1 attempt
    
    timeout_exception = httpx.ReadTimeout("Test ReadTimeout", request=None)
    mock_ai_model.get_embedding_async.side_effect = timeout_exception

    with pytest.raises(httpx.ReadTimeout) as exc_info:
        await text_embedder_instance.process_item(TEST_DATA_INPUT)
    
    assert exc_info.value is timeout_exception
    mock_ai_model.get_embedding_async.assert_called_once_with(TEST_DATA_INPUT)
    
    mock_sleep.assert_not_called() # No retries, so no sleep
    
    mock_log.warning.assert_not_called()
    mock_log.error.assert_called_once_with(
        # Log message reflects max_retries = 0, so 0 + 1 = 1 attempt
        f"All 1 attempts to get embedding failed with timeout."
    )
