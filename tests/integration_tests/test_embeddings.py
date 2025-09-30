"""Test SambaNova embeddings."""

from langchain_sambanova import (
    SambaNovaCloudEmbeddings,
    SambaNovaEmbeddings,
    SambaStudioEmbeddings,
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


def test_langchain_sambacloud_embed_documents() -> None:
    """Test SambaNovaCloud embeddings."""
    documents = ["foo bar", "bar foo"]
    embedding = SambaNovaCloudEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


def test_langchain_sambacloud_embed_query() -> None:
    """Test SambaNovaCloud embeddings."""
    query = "foo bar"
    embedding = SambaNovaCloudEmbeddings()
    output = embedding.embed_query(query)
    assert len(output) > 0


async def test_langchain_sambacloud_aembed_documents() -> None:
    """Test SambaNovaCloud embeddings asynchronous."""
    documents = ["foo bar", "bar foo"]
    embedding = SambaNovaCloudEmbeddings()
    output = await embedding.aembed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


async def test_langchain_sambacloud_aembed_query() -> None:
    """Test SambaNovaCloud embeddings asynchronous."""
    query = "foo bar"
    embedding = SambaNovaCloudEmbeddings()
    output = await embedding.aembed_query(query)
    assert len(output) > 0


def test_langchain_sambastudio_embed_documents() -> None:
    """Test SambaStudio embeddings."""
    documents = ["foo bar", "bar foo"]
    embedding = SambaStudioEmbeddings(model="E5-Mistral-7B-Instruct")
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


def test_langchain_sambastudio_embed_query() -> None:
    """Test SambaStudio embeddings."""
    query = "foo bar"
    embedding = SambaStudioEmbeddings(model="E5-Mistral-7B-Instruct")
    output = embedding.embed_query(query)
    assert len(output) > 0


async def test_langchain_sambastudio_aembed_documents() -> None:
    """Test SambaStudio embeddings asynchronous."""
    documents = ["foo bar", "bar foo"]
    embedding = SambaStudioEmbeddings(model="E5-Mistral-7B-Instruct")
    output = await embedding.aembed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


async def test_langchain_sambastudio_aembed_query() -> None:
    """Test SambaStudio embeddings asynchronous."""
    query = "foo bar"
    embedding = SambaStudioEmbeddings(model="E5-Mistral-7B-Instruct")
    output = await embedding.aembed_query(query)
    assert len(output) > 0
