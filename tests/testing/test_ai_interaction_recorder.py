import json

from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.model import AIChatResponse, AIModelInteractionMessage
from haintech.testing.ai_interaction_recorder import AICall, AIInteractionRecorder


def test_record_interaction(ai_model_lite: BaseAIModel, ai_interaction_recorder: AIInteractionRecorder):
    # Given: AI model context and clean state
    if ai_interaction_recorder.test_data_file.exists():
        ai_interaction_recorder.test_data_file.unlink()

    try:
        # When: AI model is called
        with ai_interaction_recorder.mock(ai_model_lite):
            ai_model_lite.get_chat_response(
                message=AIModelInteractionMessage(role="user", content="Who was the first US president?")
            )

        # Then: The whole interaction is stored in file
        assert ai_interaction_recorder.test_data_file.exists()
    finally:
        # Clean up
        if ai_interaction_recorder.test_data_file.exists():
            ai_interaction_recorder.test_data_file.unlink()

async def test_record_interaction_async(ai_model_lite: BaseAIModel, ai_interaction_recorder: AIInteractionRecorder):
    # Given: AI model context and clean state
    if ai_interaction_recorder.test_data_file.exists():
        ai_interaction_recorder.test_data_file.unlink()

    try:
        # When: AI model is called
        with ai_interaction_recorder.mock(ai_model_lite):
            await ai_model_lite.get_chat_response_async(
                message=AIModelInteractionMessage(role="user", content="Who was the first US president?")
            )

        # Then: The whole interaction is stored in file
        assert ai_interaction_recorder.test_data_file.exists()
    finally:
        # Clean up
        if ai_interaction_recorder.test_data_file.exists():
            ai_interaction_recorder.test_data_file.unlink()

def test_replay_interaction(ai_model_lite: BaseAIModel, ai_interaction_recorder: AIInteractionRecorder):
    # Given: Mocked response in data file
    expected_response = AIChatResponse(
        content="George Washington",
        vendor_model_name="mocked-model",
        input_tokens=10,
        output_tokens=5,
    )
    mock_call = AICall(
        message=AIModelInteractionMessage(role="user", content="Who was the first US president?"),
        response=expected_response,
    )

    try:
        ai_interaction_recorder.test_data_file.parent.mkdir(parents=True, exist_ok=True)
        with ai_interaction_recorder.test_data_file.open("w", encoding="utf-8") as fd:
            json.dump([mock_call.model_dump(mode="json")], fd)

        # When: Executing call within mock context
        with ai_interaction_recorder.mock(ai_model_lite):
            response = ai_model_lite.get_chat_response(
                message=AIModelInteractionMessage(role="user", content="Who was the first US president?")
            )

        # Then: Recorded response is returned
        assert response == expected_response
    finally:
        # Clean up
        if ai_interaction_recorder.test_data_file.exists():
            ai_interaction_recorder.test_data_file.unlink()

async def test_replay_interaction_async(ai_model_lite: BaseAIModel, ai_interaction_recorder: AIInteractionRecorder):
    # Given: Mocked response in data file
    expected_response = AIChatResponse(
        content="George Washington",
        vendor_model_name="mocked-model",
        input_tokens=10,
        output_tokens=5,
    )
    mock_call = AICall(
        method="get_chat_response_async",
        message=AIModelInteractionMessage(role="user", content="Who was the first US president?"),
        response=expected_response,
    )

    try:
        ai_interaction_recorder.test_data_file.parent.mkdir(parents=True, exist_ok=True)
        with ai_interaction_recorder.test_data_file.open("w", encoding="utf-8") as fd:
            json.dump([mock_call.model_dump(mode="json")], fd)

        # When: Executing call within mock context
        with ai_interaction_recorder.mock(ai_model_lite):
            response = await ai_model_lite.get_chat_response_async(
                message=AIModelInteractionMessage(role="user", content="Who was the first US president?")
            )

        # Then: Recorded response is returned
        assert response == expected_response
    finally:
        # Clean up
        if ai_interaction_recorder.test_data_file.exists():
            ai_interaction_recorder.test_data_file.unlink()
