"""Test ChatSambaNova chat model."""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_sambanova.chat_models import ChatSambaNovaCloud, ChatSambaStudio


class TestSambaNovaCloudBase(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatSambaNovaCloud]:
        return ChatSambaNovaCloud

    @property
    def chat_model_params(self) -> dict:
        return {"model": "Meta-Llama-3.3-70B-Instruct", "temperature": 0}

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_structured_output(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def returns_usage_metadata(self) -> bool:
        return True

    @pytest.mark.xfail(
        reason="omitted test given model can generate non parsable tool call"
    )
    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        pytest.skip("Test skipped")


class TestSambaStudioBase(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatSambaStudio]:
        return ChatSambaStudio

    @property
    def chat_model_params(self) -> dict:
        return {"model": "Meta-Llama-3.3-70B-SD-Llama-3.2-1B-TP16", "temperature": 0}

    @property
    def has_structured_output(self) -> bool:
        return False

    @property
    def has_tool_calling(self) -> bool:
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        return True
