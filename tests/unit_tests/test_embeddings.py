"""Test embedding model integration."""

from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_sambanova.embeddings import (
    SambaNovaCloudEmbeddings,
    SambaStudioEmbeddings,
)


class TestSambaStudioEmbeddingsBase(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> type[SambaStudioEmbeddings]:
        return SambaStudioEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}


class TestSambaNovaCloudEmbeddingsBase(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> type[SambaNovaCloudEmbeddings]:
        return SambaNovaCloudEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
