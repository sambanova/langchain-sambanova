"""Standard LangChain interface tests."""

from langchain_core.embeddings import Embeddings
from langchain_tests.unit_tests.embeddings import EmbeddingsUnitTests

from langchain_sambanova.embeddings import (
    SambaNovaEmbeddings,
)


class TestSambaNovaStandard(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> type[Embeddings]:
        return SambaNovaEmbeddings

    @property
    def embeddings_params(self) -> dict:
        return {"api_key": "test_api_key"}

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
