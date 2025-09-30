"""Standard LangChain interface tests"""

from langchain_core.embeddings import Embeddings
from langchain_tests.integration_tests.embeddings import EmbeddingsIntegrationTests

from langchain_sambanova import (
    SambaNovaCloudEmbeddings,
    SambaNovaEmbeddings,
    SambaStudioEmbeddings,
)


class TestSambaNovaStandard(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[Embeddings]:
        return SambaNovaEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "E5-Mistral-7B-Instruct"}


class TestSambaNovaCloudStandard(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[Embeddings]:
        return SambaNovaCloudEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "E5-Mistral-7B-Instruct"}


class TestSambaStudioStandard(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[Embeddings]:
        return SambaStudioEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "E5-Mistral-7B-Instruct"}
