"""Test embedding model integration."""

from typing import Type

from langchain_sambanova.embeddings import SambaNovaEmbeddings
from langchain_tests.unit_tests import EmbeddingsUnitTests


class TestParrotLinkEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[SambaNovaEmbeddings]:
        return SambaNovaEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
