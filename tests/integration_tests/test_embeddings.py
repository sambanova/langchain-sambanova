"""Test SambaNova embeddings."""

from langchain_sambanova import (
    SambaNovaEmbeddings,
)


def test_langchain_sambanova_embed_documents() -> None:
    """Test SambaNova embeddings."""
    documents = ["foo bar", "bar foo"]
    embedding = SambaNovaEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


def test_langchain_sambanova_embed_query() -> None:
    """Test SambaNova embeddings."""
    query = "foo bar"
    embedding = SambaNovaEmbeddings()
    output = embedding.embed_query(query)
    assert len(output) > 0


async def test_langchain_sambanova_aembed_documents() -> None:
    """Test SambaNova embeddings asynchronous."""
    documents = ["foo bar", "bar foo"]
    embedding = SambaNovaEmbeddings()
    output = await embedding.aembed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


async def test_langchain_sambanova_aembed_query() -> None:
    """Test SambaNova embeddings asynchronous."""
    query = "foo bar"
    embedding = SambaNovaEmbeddings()
    output = await embedding.aembed_query(query)
    assert len(output) > 0
