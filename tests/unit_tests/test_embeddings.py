"""Test embedding model integration."""

from langchain_sambanova.embeddings import (
    SambaNovaCloudEmbeddings,
    SambaNovaEmbeddings,
    SambaStudioEmbeddings,
)


def test_sambanova_initialization() -> None:
    """Test sambanova embedding model initialization."""
    SambaNovaEmbeddings(model="E5-Mistral-7B-Instruct")


def test_sambacloud_initialization() -> None:
    """Test sambacloud embedding model initialization."""
    SambaNovaCloudEmbeddings(model="E5-Mistral-7B-Instruct")


def test_sambastudio_initialization() -> None:
    """Test sambastudio embedding model initialization."""
    SambaStudioEmbeddings(
        sambastudio_url="https://api.sambanova.ai/v1/embeddings",
        model="E5-Mistral-7B-Instruct",
    )
