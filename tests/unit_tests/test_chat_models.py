"""Test SambaNova Chat API wrapper."""

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import langchain_core.load as lc_load
import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
)

from langchain_sambanova.chat_models import ChatSambaNova, _convert_dict_to_message

if "SAMBANOVA_API_KEY" not in os.environ:
    os.environ["SAMBANOVA_API_KEY"] = "dummy-key"


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatSambaNova()


def test_sambanova_model_param() -> None:
    llm = ChatSambaNova(model="foo")
    assert llm.model_name == "foo"
    llm = ChatSambaNova(model="foo")
    assert llm.model_name == "foo"


def test__convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_ai_with_reasoning() -> None:
    message = {"role": "assistant", "content": "foo", "reasoning": "bar"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(
        content="foo", additional_kwargs={"reasoning_content": "bar"}
    )
    assert result == expected_output


def test__convert_dict_to_message_tool_call() -> None:
    raw_tool_call = {
        "id": "call_bd254d722a8d471fac",
        "function": {
            "arguments": '{"kind":"time"}',
            "name": "get_time",
        },
        "type": "function",
    }
    message = {"role": "assistant", "content": None, "tool_calls": [raw_tool_call]}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": [raw_tool_call]},
        tool_calls=[
            ToolCall(
                name="get_time",
                args={"kind": "time"},
                id="call_bd254d722a8d471fac",
                type="tool_call",
            )
        ],
    )
    assert result == expected_output

    # Test malformed tool call
    raw_tool_calls = [
        {
            "id": "call_bd254d722a8d471fac",
            "function": {
                "arguments": "time",
                "name": "get_time",
            },
            "type": "function",
        },
        {
            "id": "call_abc123",
            "function": {
                "arguments": '{"kind": "time"}',
                "name": "get_time",
            },
            "type": "function",
        },
    ]
    message = {"role": "assistant", "content": None, "tool_calls": raw_tool_calls}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": raw_tool_calls},
        invalid_tool_calls=[
            InvalidToolCall(
                name="get_time",
                args="time",
                id="call_bd254d722a8d471fac",
                error="Function get_time arguments:\n\ntime\n\nare not valid JSON. Received JSONDecodeError Expecting value: line 1 column 1 (char 0)\nFor troubleshooting, visit: https://docs.langchain.com/oss/python/langchain/errors/OUTPUT_PARSING_FAILURE ",  # noqa: E501
                type="invalid_tool_call",
            ),
        ],
        tool_calls=[
            ToolCall(
                name="get_time",
                args={"kind": "time"},
                id="call_abc123",
                type="tool_call",
            ),
        ],
    )
    assert result == expected_output


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "run--770f482e-97fe-4390-9543-65649c2cdca6",
        "object": "chat.completion",
        "created": 1759447637.534756,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Bar",
                },
                "finish_reason": "stop",
            }
        ],
    }


def test_sambanova_invoke(mock_completion: dict) -> None:
    llm = ChatSambaNova(model="test-model")
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = llm.invoke("foo")
        assert res.content == "Bar"
        assert type(res) is AIMessage
    assert completed


async def test_sambanova_ainvoke(mock_completion: dict) -> None:
    llm = ChatSambaNova(model="test-model")
    mock_client = AsyncMock()
    completed = False

    async def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "async_client",
        mock_client,
    ):
        res = await llm.ainvoke("foo")
        assert res.content == "Bar"
        assert type(res) is AIMessage
    assert completed


def test_chat_sambanova_extra_kwargs() -> None:
    """Test extra kwargs to chat sambanova."""
    # Check that foo is saved in extra_kwargs.
    with pytest.warns(UserWarning) as record:
        llm = ChatSambaNova(model="test-model", foo=3, max_tokens=10)  # type: ignore[call-arg]
        assert llm.max_tokens == 10
        assert llm.model_kwargs == {"foo": 3}
    assert len(record) == 1
    assert type(record[0].message) is UserWarning
    assert "foo is not default parameter" in record[0].message.args[0]

    # Test that if extra_kwargs are provided, they are added to it.
    with pytest.warns(UserWarning) as record:
        llm = ChatSambaNova(model="test-model", foo=3, model_kwargs={"bar": 2})  # type: ignore[call-arg]
        assert llm.model_kwargs == {"foo": 3, "bar": 2}
    assert len(record) == 1
    assert type(record[0].message) is UserWarning
    assert "foo is not default parameter" in record[0].message.args[0]

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatSambaNova(model="test-model", foo=3, model_kwargs={"foo": 2})  # type: ignore[call-arg]

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        ChatSambaNova(model="test-model", model_kwargs={"temperature": 0.2})

    # Test that "model" cannot be specified in kwargs
    with pytest.raises(ValueError):
        ChatSambaNova(model="test-model", model_kwargs={"model": "test-model"})


def test_chat_sambanova_secret() -> None:
    """Test that secret is not printed."""
    secret = "secretKey"  # noqa: S105
    not_secret = "safe"  # noqa: S105
    llm = ChatSambaNova(
        model="test-model",
        api_key=secret,  # type: ignore[arg-type]
        model_kwargs={"not_secret": not_secret},
    )
    stringified = str(llm)
    assert not_secret in stringified
    assert secret not in stringified


@pytest.mark.filterwarnings("ignore:The function `loads` is in beta")
def test_sambanova_serialization() -> None:
    """Test that ChatSambanova can be successfully serialized and deserialized."""
    api_key1 = "secret_key"
    api_key2 = "secret_key_2"
    llm = ChatSambaNova(model="test-model", api_key=api_key1, temperature=0.7)  # type: ignore[call-arg, arg-type]
    dump = lc_load.dumps(llm)
    llm2 = lc_load.loads(
        dump,
        valid_namespaces=["langchain_sambanova"],
        secrets_map={"SAMBANOVA_API_KEY": api_key2},
        allowed_objects=[ChatSambaNova],
    )

    assert type(llm2) is ChatSambaNova

    # Ensure api key wasn't dumped and instead was read from secret map.
    assert llm.sambanova_api_key is not None
    assert llm.sambanova_api_key.get_secret_value() not in dump
    assert llm2.sambanova_api_key is not None
    assert llm2.sambanova_api_key.get_secret_value() == api_key2

    # Ensure a non-secret field was preserved
    assert llm.temperature == llm2.temperature

    # Ensure a None was preserved
    assert llm.sambanova_api_base == llm2.sambanova_api_base
