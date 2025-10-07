"""Test ChatSambaNova API wrapper.

You will need SAMBANOVA_API_KEY set in your environment to run these tests.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

import pytest
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_sambanova import ChatSambaNova

_MODEL = "Llama-4-Maverick-17B-128E-Instruct"


def test_invoke() -> None:
    """Test invoke tokens from ChatSambaNova."""
    llm = ChatSambaNova(model=_MODEL)

    result = llm.invoke("Tell me a joke", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatSambaNova."""
    llm = ChatSambaNova(model=_MODEL)

    result = await llm.ainvoke("Tell me a joke", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_stream() -> None:
    """Test streaming tokens from ChatSambaNova."""
    llm = ChatSambaNova(model=_MODEL)

    full: Optional[BaseMessageChunk] = None
    chunks_with_token_counts = 0
    chunks_with_response_metadata = 0
    for token in llm.stream("Tell me a joke", stream_options={"include_usage": True}):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        full = token if full is None else full + token
        if token.usage_metadata is not None:
            chunks_with_token_counts += 1
        if token.response_metadata:
            chunks_with_response_metadata += 1
    if chunks_with_token_counts != 1 or chunks_with_response_metadata != 1:
        msg = (
            "Expected exactly one chunk with token counts or response_metadata. "
            "AIMessageChunk aggregation adds / appends counts and metadata. Check that "
            "this is behaving properly."
        )
        raise AssertionError(msg)
    assert isinstance(full, AIMessageChunk)
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )
    assert isinstance(full.response_metadata["model_name"], str)
    assert full.response_metadata["model_name"]


async def test_astream() -> None:
    """Test streaming tokens from ChatSambaNova."""
    llm = ChatSambaNova(model=_MODEL)

    full: Optional[BaseMessageChunk] = None
    chunks_with_token_counts = 0
    chunks_with_response_metadata = 0
    async for token in llm.astream(
        "Tell me a joke", stream_options={"include_usage": True}
    ):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        full = token if full is None else full + token
        if token.usage_metadata is not None:
            chunks_with_token_counts += 1
        if token.response_metadata:
            chunks_with_response_metadata += 1
    if chunks_with_token_counts != 1 or chunks_with_response_metadata != 1:
        msg = (
            "Expected exactly one chunk with token counts or response_metadata. "
            "AIMessageChunk aggregation adds / appends counts and metadata. Check that "
            "this is behaving properly."
        )
        raise AssertionError(msg)
    assert isinstance(full, AIMessageChunk)
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )
    assert isinstance(full.response_metadata["model_name"], str)
    assert full.response_metadata["model_name"]


def test_batch() -> None:
    """Test batch tokens from ChatSambaNova."""
    llm = ChatSambaNova(model=_MODEL)

    result = llm.batch(["Tell me a joke", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch() -> None:
    """Test batch tokens from ChatSambaNova."""
    llm = ChatSambaNova(model=_MODEL)

    result = await llm.abatch(["Tell me a joke", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatSambaNova."""
    llm = ChatSambaNova(model=_MODEL)

    result = await llm.abatch(
        ["Tell me a joke", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def _get_joke_class(
    schema_type: Literal["pydantic", "typeddict", "json_schema"],
) -> Any:
    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

    def validate_joke(result: Any) -> bool:
        return isinstance(result, Joke)

    class JokeDict(TypedDict):
        """Joke to tell user."""

        setup: Annotated[str, ..., "question to set up a joke"]
        punchline: Annotated[str, ..., "answer to resolve the joke"]

    def validate_joke_dict(result: Any) -> bool:
        return all(key in ["setup", "punchline"] for key in result)

    if schema_type == "pydantic":
        return Joke, validate_joke

    if schema_type == "typeddict":
        return JokeDict, validate_joke_dict

    if schema_type == "json_schema":
        return Joke.model_json_schema(), validate_joke_dict
    msg = "Invalid schema type"
    raise ValueError(msg)


@pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
def test_structured_output_json_schema(schema_type: str) -> None:
    llm = ChatSambaNova(model=_MODEL)
    schema, validation_function = _get_joke_class(schema_type)  # type: ignore[arg-type]
    chat = llm.with_structured_output(schema, method="json_schema")

    # Test invoke
    result = chat.invoke("Tell me a joke about cats.")
    validation_function(result)

    # Test stream
    chunks = []
    for chunk in chat.stream("Tell me a joke about cats."):
        validation_function(chunk)
        chunks.append(chunk)
    assert chunk
