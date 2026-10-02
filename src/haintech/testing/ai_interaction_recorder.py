import inspect
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from haintech.ai.base.base_ai_model import BaseAIModel
from haintech.ai.model import AIChatResponse, AIContext, AIModelInteractionMessage


class AICall(BaseModel):
    method: str = "get_chat_response"
    system_prompt: str | None = None
    history: list[AIModelInteractionMessage] | None = None
    context: AIContext | None = None
    message: AIModelInteractionMessage | None = None
    functions: list[str] | None = None
    response_format: str | dict[str, Any] = "text"
    response: AIChatResponse


class AIInteractionRecorder:
    def __init__(self, request: pytest.FixtureRequest, mocker: MockerFixture):
        self.mocker = mocker
        self.request = request
        pytestconfig = request.config
        self.project_root: Path = pytestconfig.rootpath
        self.tests_root: Path = self.project_root / "tests"
        self.tests_data: Path = self.tests_root / "data"

        full_test_id: str = self.request.node.nodeid
        parts = full_test_id.split("::")

        raw_name = "__".join(parts[1:])
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", raw_name)

        test_file_path = (self.project_root / parts[0]).resolve()
        try:
            rel_path = test_file_path.relative_to(self.tests_root.resolve()).with_suffix("")
        except ValueError:
            rel_path = Path(parts[0]).with_suffix("")

        self.test_data_file = self.tests_data / rel_path / f"{safe_name}.json"


    @contextmanager
    def mock(self, ai_model: BaseAIModel):
        methods = {
            "get_chat_response": {
                "function": ai_model.get_chat_response,
                "signature": inspect.signature(ai_model.get_chat_response),
                "async": False,
            },
            "get_chat_response_async": {
                "function": ai_model.get_chat_response_async,
                "signature": inspect.signature(ai_model.get_chat_response_async),
                "async": True,
            },
        }

        patches = []

        def patch_method(method_name: str, side_effect: Any) -> None:
            patched = self.mocker.patch.object(
                ai_model,
                method_name,
                side_effect=side_effect,
            )
            patches.append(patched)

        try:
            if not self.test_data_file.exists():
                recorded_calls = []

                def make_sync_wrapper(method_name: str, original: Any):
                    def wrapper(*args: Any, **kwargs: Any) -> AIChatResponse:
                        result = original(*args, **kwargs)
                        recorded_calls.append((method_name, args, kwargs, result))
                        return result

                    return wrapper

                def make_async_wrapper(method_name: str, original: Any):
                    async def wrapper(*args: Any, **kwargs: Any) -> AIChatResponse:
                        result = await original(*args, **kwargs)
                        recorded_calls.append((method_name, args, kwargs, result))
                        return result

                    return wrapper

                for method_name, method_data in methods.items():
                    original = method_data["function"]

                    if method_data["async"]:
                        wrapper = make_async_wrapper(method_name, original)
                    else:
                        wrapper = make_sync_wrapper(method_name, original)

                    patch_method(method_name, wrapper)

                yield self

                serialized_calls = []

                for method_name, args, kwargs, result in recorded_calls:
                    signature = methods[method_name]["signature"]
                    bound_args = signature.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    serialized_calls.append(
                        AICall(
                            method=method_name,
                            response=result,
                            **bound_args.arguments,
                        ).model_dump(
                            mode="json",
                            exclude_unset=True,
                            exclude_none=True,
                        )
                    )

                self.test_data_file.parent.mkdir(parents=True, exist_ok=True)

                with self.test_data_file.open("w", encoding="utf-8") as fd:
                    json.dump(
                        serialized_calls,
                        fd,
                        indent=2,
                        ensure_ascii=False,
                    )

            else:
                with self.test_data_file.open("r", encoding="utf-8") as fd:
                    data = json.load(fd)

                recorded_calls = [AICall.model_validate(item) for item in data]
                call_index = 0

                def validate_call(
                    method_name: str,
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                ) -> AIChatResponse:
                    nonlocal call_index

                    if call_index >= len(recorded_calls):
                        raise AssertionError(
                            "The number of AI interaction calls exceeds the number of recorded interactions."
                        )

                    expected_call = recorded_calls[call_index]

                    if expected_call.method != method_name:
                        raise AssertionError(
                            "The AI method does not match the recorded interaction:\n"
                            f"Expected method: {expected_call.method!r}\n"
                            f"Received method: {method_name!r}"
                        )

                    signature = methods[method_name]["signature"]
                    actual_bound_args = signature.bind(*args, **kwargs)
                    actual_bound_args.apply_defaults()

                    expected_args = expected_call.model_dump(
                        mode="json",
                        exclude={"method", "response"},
                        exclude_unset=True,
                        exclude_none=True,
                    )

                    actual_args = normalize(
                        {name: value for name, value in actual_bound_args.arguments.items() if value is not None}
                    )

                    if actual_args != expected_args:
                        raise AssertionError(
                            "The AI method arguments do not match "
                            "the recorded interaction:\n"
                            f"Expected: {expected_args!r}\n"
                            f"Received: {actual_args!r}"
                        )

                    call_index += 1
                    return expected_call.response

                def replay_sync(*args: Any, **kwargs: Any) -> AIChatResponse:
                    return validate_call(
                        "get_chat_response",
                        args,
                        kwargs,
                    )

                async def replay_async(*args: Any, **kwargs: Any) -> AIChatResponse:
                    return validate_call(
                        "get_chat_response_async",
                        args,
                        kwargs,
                    )

                patch_method("get_chat_response", replay_sync)
                patch_method("get_chat_response_async", replay_async)

                yield self

                if call_index != len(recorded_calls):
                    raise AssertionError(
                        "The number of AI interaction calls is lower than "
                        "the number of recorded interactions: "
                        f"{call_index} != {len(recorded_calls)}"
                    )
        finally:
            for patched in reversed(patches):
                self.mocker.stop(patched)


def normalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)

    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}

    if isinstance(value, list):
        return [normalize(item) for item in value]

    return value


@pytest.fixture
def ai_interaction_recorder(request: pytest.FixtureRequest, mocker: MockerFixture) -> AIInteractionRecorder:
    return AIInteractionRecorder(request, mocker)
