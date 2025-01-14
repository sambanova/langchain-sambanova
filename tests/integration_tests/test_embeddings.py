"""Test SambaNova embeddings."""

from typing import Type

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_sambanova.embeddings import SambaStudioEmbeddings


class TestSambaStudioEmbeddingsBase(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[SambaStudioEmbeddings]:
        return SambaStudioEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model_kwargs": {"select_expert": "e5-mistral-7b-instruct-32768"}}
