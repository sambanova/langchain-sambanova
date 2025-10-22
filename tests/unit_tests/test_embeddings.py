"""Test embedding model integration."""

from langchain_sambanova.embeddings import (
    SambaNovaEmbeddings,
)


def test_sambanova_initialization() -> None:
    """Test sambanova embedding model initialization."""
    SambaNovaEmbeddings(model="E5-Mistral-7B-Instruct")
