"""Standard LangChain interface tests."""
# run pytest tests/unit_tests/test_chat_models_standard.py --snapshot-update
# to update syrup snapshots

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests.chat_models import (
    ChatModelUnitTests,
)

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
