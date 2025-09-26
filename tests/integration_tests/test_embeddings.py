"""Test SambaNova embeddings."""

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_sambanova.embeddings import (
    SambaNovaCloudEmbeddings,
    SambaStudioEmbeddings,
)


class TestSambaStudioEmbeddingsBase(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[SambaStudioEmbeddings]:
        return SambaStudioEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "e5-mistral-7b-instruct-8192"}


class TestSambaNovaCloudEmbeddingsBase(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[SambaNovaCloudEmbeddings]:
        return SambaNovaCloudEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "E5-Mistral-7B-Instruct"}
