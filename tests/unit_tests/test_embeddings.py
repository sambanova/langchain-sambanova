"""Test embedding model integration."""

from typing import Type

from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_sambanova.embeddings import SambaStudioEmbeddings


class TestSambaStudioEmbeddingsBase(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[SambaStudioEmbeddings]:
        return SambaStudioEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
