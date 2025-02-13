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
        return {"model": "Meta-Llama-3.3-70B-Instruct", "temperature": 0.7}

    @property
    def has_structured_output(self) -> bool:
        return False

    @property
    def has_tool_calling(self) -> bool:
        return False


class TestSambaStudioBase(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatSambaStudio]:
        return ChatSambaStudio

    @property
    def chat_model_params(self) -> dict:
        return {"model": "Meta-Llama-3.1-8B-Instruct", "temperature": 0}

    @property
    def has_structured_output(self) -> bool:
        return False

    @property
    def has_tool_calling(self) -> bool:
        return False
