"""Test chat model integration."""

from typing import Type

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_sambanova.chat_models import ChatSambaNovaCloud, ChatSambaStudio


class TestSambaNovaCloudBase(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatSambaNovaCloud]:
        return ChatSambaNovaCloud

    @property
    def chat_model_params(self) -> dict:
        return {"model": "Meta-Llama-3.3-70B-Instruct", "temperature": 0}


class TestSambaStudioBase(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatSambaStudio]:
        return ChatSambaStudio

    @property
    def chat_model_params(self) -> dict:
        return {"model": "Meta-Llama-3-70B-Instruct-4096", "temperature": 0}
