"""Standard LangChain interface tests."""
# run pytest tests/unit_tests/test_chat_models_standard.py --snapshot-update
# to update syrup snapshots

import os
from unittest import mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.load import dumpd, load
from langchain_tests.unit_tests.chat_models import (
    ChatModelUnitTests,
)
from syrupy import SnapshotAssertion

from langchain_sambanova import ChatSambaNova


class TestSambaNovaStandard(ChatModelUnitTests):
    """Run ChatSambaNova on LangChain standard tests."""

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatSambaNova

    @property
    def chat_model_params(self) -> dict:
        return {"model": "Llama-4-Maverick-17B-128E-Instruct"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "SAMBANOVA_API_BASE": "https://base.com",
                "SAMBANOVA_API_KEY": "api_key",
            },
            {},
            {
                "sambanova_api_base": "https://base.com",
                "sambanova_api_key": "api_key",
            },
        )

    @pytest.mark.xfail(
        reason="Overridden to add allowed_objects=[ChatSambaNova] parameter to load() "
        "due to stricter deserialization in langchain-core>=1.2"
    )
    def test_serdes(self, model: BaseChatModel, snapshot: SnapshotAssertion) -> None:
        """Test serialization and deserialization with allowed_objects."""
        if not self.chat_model_class.is_lc_serializable():
            pytest.skip("Model is not serializable.")
        else:
            env_params, _model_params, _expected_attrs = self.init_from_env_params
            with mock.patch.dict(os.environ, env_params):
                ser = dumpd(model)
                assert ser == snapshot(name="serialized")
                assert (
                    model.dict()
                    == load(
                        dumpd(model),
                        valid_namespaces=model.get_lc_namespace()[:1],
                        allowed_objects=[ChatSambaNova],
                    ).dict()
                )
