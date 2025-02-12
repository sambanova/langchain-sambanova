"""Test SambaNova embeddings."""

from typing import Type

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_sambanova.embeddings import SambaStudioEmbeddings, SambaNovaCloudEmbeddings


class TestSambaStudioEmbeddingsBase(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[SambaStudioEmbeddings]:
        return SambaStudioEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model_kwargs": {"select_expert": "e5-mistral-7b-instruct-32768"}} # check

class TestSambaNovaCloudEmbeddingsBase(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[SambaNovaCloudEmbeddings]:
        return SambaNovaCloudEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "E5-Mistral-7B-Instruct"} # check