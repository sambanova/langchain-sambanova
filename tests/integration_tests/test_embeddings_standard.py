"""Standard LangChain interface tests."""

from langchain_core.embeddings import Embeddings
from langchain_tests.integration_tests.embeddings import EmbeddingsIntegrationTests

from langchain_sambanova import (
    SambaNovaEmbeddings,
)


class TestSambaNovaStandard(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[Embeddings]:
        return SambaNovaEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "E5-Mistral-7B-Instruct"}
