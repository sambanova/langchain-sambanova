"""Test SambaNova embeddings."""

from typing import Type

from langchain_sambanova.embeddings import SambaNovaEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[SambaNovaEmbeddings]:
        return SambaNovaEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
