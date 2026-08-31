from haintech.ai.ai_model_factory import AIModelFactory
from haintech.ai.google_genai import GoogleAIModel


def test_get_ai_model_names():
    # Given: A factory with Google model
    factory = AIModelFactory()
    factory.add_ai_model_class(GoogleAIModel)
    # When: Get model names
    ret = factory.get_ai_model_names()
    # Then: Gemini model is returned
    assert len(ret) > 0
    assert any(r for r in ret if r.startswith("google/gemini"))

async def test_get_ai_model_names_async():
    # Given: A factory with Google model
    factory = AIModelFactory()
    factory.add_ai_model_class(GoogleAIModel)
    # When: Get model names
    ret = await factory.get_ai_model_names_async()
    # Then: Gemini model is returned
    assert len(ret) > 0
    assert any(r for r in ret if r.startswith("google/gemini"))


def test_create_ai_model():
    # Given: A factory with Google model
    factory = AIModelFactory([GoogleAIModel])
    # And: vendor_model_name
    name = factory.get_ai_model_names()[0]
    # When: Create AI model
    ai_model = factory.create_ai_model(name)
    # Then: The model is returned
    assert isinstance(ai_model, GoogleAIModel)
