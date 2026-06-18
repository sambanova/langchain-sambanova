"""Standard LangChain interface tests."""

from typing import Any, Literal

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import (
    ChatModelIntegrationTests,
)

from langchain_sambanova import ChatSambaNova

rate_limiter = InMemoryRateLimiter(
    requests_per_second=2,
)


class TestSambaNovaStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatSambaNova

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "gemma-4-31B-it",
            "rate_limiter": rate_limiter,
        }

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(model, my_adder_tool)

    @property
    def supports_json_mode(self) -> bool:
        return True

    @pytest.mark.xfail(reason="Requires stream_options={'include_usage': True} to pass")
    def test_usage_metadata_streaming(self) -> Any:
        model = self.chat_model_class(
            **{
                **self.chat_model_params,
                "stream_options": {"include_usage": True},
            }
        )
        super().test_usage_metadata_streaming(model)

    @pytest.mark.xfail(reason="tool_choice param is not functional")
    def test_tool_choice(self, model: BaseChatModel) -> None:
        super().test_tool_choice(model)


@pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
def test_json_schema(
    schema_type: Literal["pydantic", "typeddict", "json_schema"],
) -> None:
    class JsonSchemaTests(ChatModelIntegrationTests):
        @property
        def chat_model_class(self) -> type[ChatSambaNova]:
            return ChatSambaNova

        @property
        def chat_model_params(self) -> dict:
            return {
                "model": "gemma-4-31B-it",
                "rate_limiter": rate_limiter,
            }

        @property
        def structured_output_kwargs(self) -> dict:
            return {"method": "json_schema"}

    test_instance = JsonSchemaTests()
    model = test_instance.chat_model_class(**test_instance.chat_model_params)
    JsonSchemaTests().test_structured_output(model, schema_type)
