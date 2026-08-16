from haintech.ai.anthropic.anthropic_ai_model import AnthropicAIModel


def test_get_model_names():
    models = AnthropicAIModel.get_model_names()
    assert len(models) > 0
    assert any(m for m in models if m.startswith("claude"))
