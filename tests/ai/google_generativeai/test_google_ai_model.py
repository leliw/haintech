from haintech.ai.google_generativeai.google_ai_model import GoogleAIModel


def test_list_models():
    # Given: Google AI Model
    ai_model = GoogleAIModel()
    # When: Ask for model names
    ret = ai_model.get_model_names()
    # Then: At least one gemini model name is returned
    assert any("gemini" in n for n in ret)
