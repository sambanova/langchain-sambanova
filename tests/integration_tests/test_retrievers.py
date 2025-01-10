from typing import Type

from langchain_sambanova.retrievers import SambaNovaRetriever
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)


class TestSambaNovaRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[SambaNovaRetriever]:
        """Get an empty vectorstore for unit tests."""
        return SambaNovaRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"k": 2}

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a dictionary representing the "args" of an example retriever call.
        """
        return "example query"
