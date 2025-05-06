"""Test ChatSambaNova chat model."""

from typing import Type

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


class TestSambaStudioBase(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatSambaStudio]:
        return ChatSambaStudio

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
