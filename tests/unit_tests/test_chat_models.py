"""Test chat model integration."""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests
from syrupy import SnapshotAssertion

from langchain_sambanova.chat_models import ChatSambaNovaCloud, ChatSambaStudio


class TestSambaNovaCloudBase(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatSambaNovaCloud]:
        return ChatSambaNovaCloud

    @property
    def chat_model_params(self) -> dict:
        return {"model": "Meta-Llama-3.3-70B-Instruct", "temperature": 0}

    @pytest.mark.xfail(
        reason="omitted test until mapping included in langchain_core loading module"
    )
    def test_serdes(self, model: BaseChatModel, snapshot: SnapshotAssertion) -> None:
        pytest.skip("Test skipped")


class TestSambaStudioBase(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatSambaStudio]:
        return ChatSambaStudio

    @property
    def chat_model_params(self) -> dict:
        return {"model": "Meta-Llama-3-70B-Instruct-4096", "temperature": 0}

    @pytest.mark.xfail(
        reason="omitted test until mapping included in langchain_core loading module"
    )
    def test_serdes(self, model: BaseChatModel, snapshot: SnapshotAssertion) -> None:
        pytest.skip("Test skipped")
