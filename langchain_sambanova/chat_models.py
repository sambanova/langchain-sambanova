"""SambaNova chat models."""

from __future__ import annotations

import contextlib
import json
import logging
import urllib.parse
import warnings
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)

import requests
from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    convert_to_secret_str,
    from_env,
    get_from_dict_or_env,
    get_pydantic_field_names,
    secret_from_env,
)
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import (
    TypeBaseModel,
    is_basemodel_subclass,
)
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from requests import Response
from sambanova import AsyncSambaNova, SambaNova
from typing_extensions import Self

logger = logging.getLogger(__name__)


class ChatSambaNova(BaseChatModel):
    r"""SambaNova chat model.

    Setup:

        Install ``langchain_sambanova`` and set environment variable
        ``SAMBANOVA_API_KEY`` with your SambaNova cloud or SambaStack API Key.
        Get one at https://cloud.sambanova.ai/

        .. code-block:: bash

            pip install -U langchain_sambanova
            export SAMBANOVA_API_KEY="your-api-key"

    Key init args — completion params:

        model: str
            Name of SambaNova model to use, e.g. ``Llama-4-Maverick-17B-128E-Instruct``.
        max_tokens: int
            max tokens to generate
        temperature: float
            model temperature
        top_p: float
            model top p
        stream_options: dict
            stream options, include usage to get generation metrics
        model_kwargs: Dict[str, Any]
            Holds any model parameters valid for create call not
            explicitly specified.

    Key init args — client params:

        api_key: Optional[SecretStr] = None
            SambaNova API key.
        base_url: Optional[str] = "https://api.sambanova.ai/v1/"
            SambaNova URL, set it when using a SambaStack environment.
        max_retries: Optional[int] = 2
            Maximum number of retries to make when generating.
        request_timeout: Optional[float] = 60.0
            Timeout for requests to SambaCloud or SambaStack completion API

    Instantiate:

        .. code-block:: python

            from langchain_sambanova import ChatSambaNova

            llm = ChatSambaNovaCloud(
                model="Llama-4-Maverick-17B-128E-Instruct",
                max_tokens=4096,
                temperature=0.7,
                # api_base = "SambaNova endpoint URL",
                # api_key = "SambaNova API key",
                # other params...
            )

    Invoke:

        .. code-block:: python

            messages = [
                ("system", "your are an AI assistant."),
                ("human", "tell me a joke."),
            ]
            response = llm.invoke(messages)
            print(response)

        .. code-block:: python

            AIMessage(
                content="Why couldn't the bicycle stand up by itself?\n\n(wait for it...)\n\nBecause it was two-tired!",
                response_metadata={
                    "token_usage": {
                        "completion_tokens": 16,
                        "completion_tokens_after_first_per_sec": 478.5578129872896,
                        "completion_tokens_after_first_per_sec_first_ten": 499.20304689359676,
                        "completion_tokens_after_first_per_sec_graph": 499.20304689359676,
                        "completion_tokens_per_sec": 177.65276251942407,
                        "end_time": 1759254758.8066719,
                        "is_last_response": true,
                        "prompt_tokens": 42,
                        "stop_reason": "stop",
                        "time_to_first_token": 0.05871915817260742,
                        "total_latency": 0.09006333351135254,
                        "total_tokens": 58,
                        "total_tokens_per_sec": 643.9912641329122,
                    },
                    "model_name": "Llama-4-Maverick-17B-128E-Instruct",
                    "system_fingerprint": "fastcoe",
                    "finish_reason": "stop",
                    "logprobs": None,
                },
                id="9fab00ba-fb1f-4fdb-af57-55f6a1dc158b",
                usage_metadata={"input_tokens": 42, "output_tokens": 16, "total_tokens": 58},
            )

    Stream:

        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.text(), end="")

        .. code-block:: python

            AIMessageChunk(content="Why", id="a735db9b-530b-4591-aa2d-7c07216011c2")
            AIMessageChunk(
                content=" couldn't the bicycle stand up by itself?\n\n(wait for it...",
                id="a735db9b-530b-4591-aa2d-7c07216011c2"
            )
            AIMessageChunk(
                content="\n\nBecause it was two-tired!",
                id="a735db9b-530b-4591-aa2d-7c07216011c2"
            )
            AIMessageChunk(
                content="",
                response_metadata={
                    'finish_reason': 'stop', 'model_name': 'Llama-4-Maverick-17B-128E-Instruct', "system_fingerprint":"fastcoe"
                }
                id="a735db9b-530b-4591-aa2d-7c07216011c2",
            )

    Async:

        .. code-block:: python

            response = llm.ainvoke(messages)
            await response

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            AIMessage(
                content="Why couldn't the bicycle stand up by itself?\n\n(wait for it...)\n\nBecause it was two-tired!",
                response_metadata={
                    "token_usage": {
                        "completion_tokens": 16,
                        "completion_tokens_after_first_per_sec": 478.5578129872896,
                        "completion_tokens_after_first_per_sec_first_ten": 499.20304689359676,
                        "completion_tokens_after_first_per_sec_graph": 499.20304689359676,
                        "completion_tokens_per_sec": 177.65276251942407,
                        "end_time": 1759254758.8066719,
                        "is_last_response": true,
                        "prompt_tokens": 42,
                        "stop_reason": "stop",
                        "time_to_first_token": 0.05871915817260742,
                        "total_latency": 0.09006333351135254,
                        "total_tokens": 58,
                        "total_tokens_per_sec": 643.9912641329122,
                    },
                    "model_name": "Llama-4-Maverick-17B-128E-Instruct",
                    "system_fingerprint": "fastcoe",
                    "finish_reason": "stop",
                    "logprobs": None,
                },
                id="9fab00ba-fb1f-4fdb-af57-55f6a1dc158b",
                usage_metadata={"input_tokens": 42, "output_tokens": 16, "total_tokens": 58},
            )

    Tool calling:

        .. code-block:: python

            from pydantic import BaseModel, Field


            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. Los Angeles, CA")


            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke("Should I bring my umbrella today in LA?")
            ai_msg.tool_calls

        .. code-block:: python

            [{"name": "GetWeather", "args": {"location": "Los Angeles, CA"}, "id": "call_adf61180ea2b4d228a"}]

    Structured output:

        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field


            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")


            structured_model = llm.with_structured_output(Joke)
            structured_model.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(setup="Why did the cat join a band?", punchline="Because it wanted to be the purr-cussionist!")

        See `ChatSambanovaCloud.with_structured_output()` for more.

    JSON mode:

        .. code-block:: python

            json_llm = llm.bind(response_format={"type": "json_object"})
            ai_msg = json_llm.invoke(
                "Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]"
            )
            ai_msg.content

        .. code-block:: python

            '{"random_ints": [14, 73, 28, 42, 67, 85, 31, 19, 46, 51]}'

    Image input:

        .. code-block:: python
            import base64
            import httpx
            from langchain_core.messages import HumanMessage

            image_url = "https://sambanova.ai/hubfs/sambanova-banner.jpg"
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ]
            )
            ai_msg = llm.invoke([message])
            ai_msg.content

        .. code-block:: python
            "The image presents a logo and branding for "sambanova" against a vibrant purple background."

    Token usage:

        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {"input_tokens": 16, "output_tokens": 12, "total_tokens": 28}

        When streaming, set the ``stream_options`` arg with ``"{include_usage": True}`` :

        .. code-block:: python

            stream = llm.stream(messages, stream_options={"include_usage": True})
            full = next(stream)
            for chunk in stream:
                full += chunk
            full.usage_metadata

        .. code-block:: python

            {"input_tokens": 16, "output_tokens": 12, "total_tokens": 28}

    Response metadata

        .. code-block:: python

            response = llm.invoke(messages)
            print(response.response_metadata)

        .. code-block:: python

            {
                "token_usage": {
                    "completion_tokens": 16,
                    "completion_tokens_after_first_per_sec": 478.5578129872896,
                    "completion_tokens_after_first_per_sec_first_ten": 499.20304689359676,
                    "completion_tokens_after_first_per_sec_graph": 499.20304689359676,
                    "completion_tokens_per_sec": 177.65276251942407,
                    "end_time": 1759254758.8066719,
                    "is_last_response": true,
                    "prompt_tokens": 42,
                    "stop_reason": "stop",
                    "time_to_first_token": 0.05871915817260742,
                    "total_latency": 0.09006333351135254,
                    "total_tokens": 58,
                    "total_tokens_per_sec": 643.9912641329122,
                },
                "model_name": "Llama-4-Maverick-17B-128E-Instruct",
                "system_fingerprint": "fastcoe",
                "finish_reason": "stop",
                "logprobs": None,
            }

    """  # noqa: E501

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:

    sambanova_api_base: Optional[str] = Field(
        alias="base_url", default_factory=from_env("SAMBANOVA_API_BASE", default=None)
    )
    """Base URL path for API requests, leave blank if not a proxy
        or SambaStack deployment."""
    sambanova_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("SAMBANOVA_API_KEY", default=None),
    )
    """SambaNova api key Automatically inferred from env var
        ``SAMBANOVA_API_KEY`` if not provided."""

    model_name: str = Field(default="Llama-4-Maverick-17B-128E-Instruct", alias="model")
    """The name of the model"""

    max_tokens: int = Field(default=1024)
    """max tokens to generate"""

    temperature: float = Field(default=0.7)
    """model temperature"""

    top_p: Optional[float] = Field(default=None)
    """model top p"""

    top_k: Optional[int] = Field(default=None)
    """model top k"""

    stop: Optional[Union[list[str], str]] = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""

    streaming: bool = Field(default=False)
    """Whether to use streaming handler when using non streaming methods"""

    stream_options: Optional[dict[str, Any]] = Field(default=None)
    """stream options, include usage to get generation metrics"""

    reasoning_effort: Optional[str] = Field(default=None)
    """The level of effort the model will put into reasoning.
        only available in reasoning models like ``gpt-oss-120b``
    """

    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.

    max_retries: int = Field(default=2, alias="retries")
    """Maximum number of retries to make when generating."""

    request_timeout: Optional[Union[float, tuple[float, float], Any]] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to OpenAI completion API. Can be float, ``httpx.Timeout`` or
        None."""

    http_client: Union[Any, None] = None
    """Optional ``httpx.Client``. Only used for sync invocations. Must specify
        ``http_async_client`` as well if you'd like a custom client for async
        invocations.
    """
    http_async_client: Union[Any, None] = None
    """Optional ``httpx.AsyncClient``. Only used for async invocations. Must specify
        ``http_client`` as well if you'd like a custom client for sync invocations."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                msg = f"Found {field_name} supplied twice."
                raise ValueError(msg)
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended.""",
                    stacklevel=2,
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            msg = (
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )
            raise ValueError(msg)

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        client_params: dict[str, Any] = {
            "api_key": (
                self.sambanova_api_key.get_secret_value()
                if self.sambanova_api_key
                else None
            ),
            "base_url": self.sambanova_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        if not (self.client or None):
            sync_specific: dict[str, Any] = (
                {"http_client": self.http_client} if self.http_client else {}
            )
            self.client = SambaNova(**client_params, **sync_specific).chat.completions
        if not (self.async_client or None):
            async_specific: dict[str, Any] = (
                {"http_client": self.http_async_client}
                if self.http_async_client
                else {}
            )
            self.async_client = AsyncSambaNova(
                **client_params, **async_specific
            ).chat.completions
        return self

    #
    # Serializable class method overrides
    #
    @property
    def lc_secrets(self) -> dict[str, str]:
        """Mapping of secret environment variables."""
        return {"sambanova_api_key": "SAMBANOVA_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    #
    # BaseChatModel method overrides
    #
    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "sambanova-chat"

    def _get_ls_params(
        self, stop: Optional[list[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="sambanova",
            ls_model_name=params.get("model", self.model_name),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop if isinstance(ls_stop, list) else [ls_stop]
        return ls_params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **kwargs,
        }
        response = self.client.create(messages=message_dicts, **params)
        return self._create_chat_result(response, params)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **kwargs,
        }
        response = await self.async_client.create(messages=message_dicts, **params)
        return self._create_chat_result(response, params)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)

        params = {**params, **kwargs, "stream": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        for chunk in self.client.create(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk["choices"]) == 0:
                chunk["choices"] = [{"delta": {}}]
            choice = chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                generation_info["model_name"] = self.model_name
                if system_fingerprint := chunk.get("system_fingerprint"):
                    generation_info["system_fingerprint"] = system_fingerprint
                reasoning_effort = (
                    params.get("reasoning_effort") or self.reasoning_effort
                )
                if reasoning_effort:
                    generation_info["reasoning_effort"] = reasoning_effort
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )

            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk, logprobs=logprobs
                )
            yield generation_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)

        params = {**params, **kwargs, "stream": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        async for chunk in await self.async_client.create(
            messages=message_dicts, **params
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk["choices"]) == 0:
                chunk["choices"] = [{"delta": {}}]
            choice = chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                generation_info["model_name"] = self.model_name
                if system_fingerprint := chunk.get("system_fingerprint"):
                    generation_info["system_fingerprint"] = system_fingerprint
                reasoning_effort = (
                    params.get("reasoning_effort") or self.reasoning_effort
                )
                if reasoning_effort:
                    generation_info["reasoning_effort"] = reasoning_effort
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )

            if run_manager:
                await run_manager.on_llm_new_token(
                    token=generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            yield generation_chunk

    #
    # Internal methods
    #
    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling SambaNova API."""
        exclude_if_none: dict = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop": self.stop or None,
            "stream_options": self.stream_options,
            "reasoning_effort": self.reasoning_effort,
        }
        return {
            **{k: v for k, v in exclude_if_none.items() if v is not None},
            **self.model_kwargs,
        }

    def _combine_llm_outputs(self, llm_outputs: list[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage:
                        overall_token_usage[k] += v
                    else:
                        overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        return combined

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: Optional[list[str]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(
        self, response: dict | BaseModel, params: dict
    ) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.model_dump()
        token_usage = response.get("usage", {})
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)
                total_tokens = token_usage.get("total_tokens", 0)
                message.usage_metadata = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            generation_info = {"finish_reason": res.get("finish_reason")}
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        reasoning_effort = params.get("reasoning_effort") or self.reasoning_effort
        if reasoning_effort:
            llm_output["reasoning_effort"] = reasoning_effort
        return ChatResult(generations=generations, llm_output=llm_output)

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]  # noqa: PYI051
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function,
                "auto" to automatically determine which function to call
                with the option to not call any function, "any" to enforce that some
                function is called, or a dict of the form:
                ``{"type": "function", "function": {"name": <<tool_name>>}}``.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.

        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if tool_choice == "any":
                tool_choice = "required"
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "none", "required")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    msg = (
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                    raise ValueError(msg)
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[dict[str, Any], type[BaseModel]]] = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        r"""Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

                - an OpenAI function/tool schema,
                - a JSON Schema,
                - a TypedDict class,
                - or a Pydantic class.

                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.

            method: The method for steering model generation, one of:

                - ``'function_calling'``:
                    Uses SambaNova tool-calling `API <https://docs.sambanova.ai/docs/en/features/function-calling>`__
                - ``'json_schema'``:
                    Uses SambaNova's `Structured Output API <https://docs.sambanova.ai/docs/en/features/function-calling#json-schema>`__.
                    Supported for a subset of models, See `docs <https://docs.sambanova.ai/docs/en/features/function-calling#supported-models>`__
                    for details.
                - ``'json_mode'``:
                    Uses SambaNova's `JSON mode <https://docs.sambanova.ai/docs/en/features/function-calling#json-mode>`__.
                    Note that if using JSON mode then you must include instructions for
                    formatting the output into the desired schema into the model call

            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys ``'raw'``, ``'parsed'``, and ``'parsing_error'``.

            kwargs:
                Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            If ``include_raw`` is False and ``schema`` is a Pydantic class, Runnable outputs
            an instance of ``schema`` (i.e., a Pydantic object).

            Otherwise, if ``include_raw`` is False then Runnable outputs a dict.

            If ``include_raw`` is True, then Runnable outputs a dict with keys:

            - ``'raw'``: BaseMessage
            - ``'parsed'``: None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
            - ``'parsing_error'``: Optional[BaseException]

        Example: schema=Pydantic class, method="function_calling", include_raw=False:

            .. code-block:: python

                from typing import Optional

                from langchain_sambanova import ChatSambaNova
                from pydantic import BaseModel, Field


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    # If we provide default values and/or descriptions for fields, these will be passed
                    # to the model. This is an important part of improving a model's ability to
                    # correctly return structured outputs.
                    justification: Optional[str] = Field(
                        default=None, description="A justification for the answer."
                    )


                llm = ChatSambaNova(model="Llama-4-Maverick-17B-128E-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, method="function_calling")

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same.'
                # )

        Example: schema=Pydantic class, method="function_calling", include_raw=True:
            .. code-block:: python

                from langchain_sambanova import ChatSambaNova
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatSambaNova(model="Llama-4-Maverick-17B-128E-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="function_calling",
                    include_raw=True,
                )

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': '63cacdd8-5cb0-4646-aa10-6c0c4b3a4eea', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]},response_metadata={'finish_reason': 'tool_calls', 'usage': {'acceptance_rate': 5, 'completion_tokens': 53, 'completion_tokens_after_first_per_sec': 343.7964936837758, 'completion_tokens_after_first_per_sec_first_ten': 439.1205661878638, 'completion_tokens_per_sec': 162.8511306784833, 'end_time': 1731527851.0698032, 'is_last_response': True, 'prompt_tokens': 213, 'start_time': 1731527850.7137961, 'time_to_first_token': 0.20475482940673828, 'total_latency': 0.32545061111450196, 'total_tokens': 266, 'total_tokens_per_sec': 817.3283162354066}, 'model_name': 'Meta-Llama-3.3-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731527850}, id='95667eaf-447f-4b53-bb6e-b6e1094ded88'),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same.'),
                #     'parsing_error': None
                # }

        Example: schema=TypedDict class, method="function_calling", include_raw=False:
            .. code-block:: python

                # IMPORTANT: If you are using Python <=3.8, you need to import Annotated
                # from typing_extensions, not from typing.
                from typing_extensions import Annotated, TypedDict

                from langchain_sambanova import ChatSambaNova


                class AnswerWithJustification(TypedDict):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: Annotated[Optional[str], None, "A justification for the answer."]


                llm = ChatSambaNova(model="Llama-4-Maverick-17B-128E-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="function_calling",
                )

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same.'
                # }

        Example: schema=OpenAI function schema, method="function_calling", include_raw=False:
            .. code-block:: python

                from langchain_sambanova import ChatSambaNova

                oai_schema = {
                    'name': 'AnswerWithJustification',
                    'description': 'An answer to the user question along with justification for the answer.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'answer': {'type': 'string'},
                            'justification': {'description': 'A justification for the answer.', 'type': 'string'}
                        },
                       'required': ['answer']
                   }
               }

                llm = ChatSambaNova(model="Llama-4-Maverick-17B-128E-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(
                    oai_schema,
                    method="function_calling"
                    )

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same.'
                # }

        Example: schema=Pydantic class, method="json_schema", include_raw=False:
            .. code-block:: python

                from typing import Optional

                from langchain_sambanova import ChatSambaNova
                from pydantic import BaseModel, Field


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    # If we provide default values and/or descriptions for fields, these will be passed
                    # to the model. This is an important part of improving a model's ability to
                    # correctly return structured outputs.
                    justification: Optional[str] = Field(
                        default=None, description="A justification for the answer."
                    )


                llm = ChatSambaNova(model="Llama-4-Maverick-17B-128E-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_schema",
                )

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same.'
                # )

        Example: schema=Pydantic class, method="json_mode", include_raw=True:
            .. code-block::

                from langchain_sambanova import ChatSambaNova
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatSambaNova(model="Llama-4-Maverick-17B-128E-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_mode",
                    include_raw=True
                )

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n    "answer": "They are both the same weight.",\n    "justification": "A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same." \n}' additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'usage': {'acceptance_rate': 5.3125, 'completion_tokens': 79, 'completion_tokens_after_first_per_sec': 292.65701089829776, 'completion_tokens_after_first_per_sec_first_ten': 346.43324678555325, 'completion_tokens_per_sec': 200.012158915008, 'end_time': 1731528071.1708555, 'is_last_response': True, 'prompt_tokens': 70, 'start_time': 1731528070.737394, 'time_to_first_token': 0.16693782806396484, 'total_latency': 0.3949759876026827, 'total_tokens': 149, 'total_tokens_per_sec': 377.2381225105847}, 'model_name': 'Meta-Llama-3.3-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731528070}, id='83208297-3eb9-4021-a856-ca78a15758df'),
                #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same.'),
                #     'parsing_error': None
                # }

        """  # noqa: E501
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling"},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_schema":
            # Use structured outputs (json_schema) for models that support it
            # Convert schema to JSON Schema format for structured outputs
            if schema is None:
                msg = (
                    "schema must be specified when method is 'json_schema'. "
                    "Received None."
                )
                raise ValueError(msg)
            json_schema = convert_to_json_schema(schema)
            schema_name = json_schema.get("title", "")
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema},
            }
            ls_format_info = {
                "kwargs": {"method": "json_schema"},
                "schema": json_schema,
            }
            llm = self.bind(
                response_format=response_format,
                ls_structured_output_format=ls_format_info,
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )

        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": "json_mode"},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            msg = (
                f"Unrecognized method argument. Expected one of 'function_calling', "
                f"'json_schema' or 'json_mode'. Received: '{method}'"
            )
            raise ValueError(msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser


@deprecated(
    since="0.2.0",
    alternative="langchain_sambanova.ChatSambaNova",
    removal="1.0.0",
)
class ChatSambaNovaCloud(BaseChatModel):
    """SambaNova Cloud chat model.

    Setup:
        To use, you should have the environment variables:
        `SAMBANOVA_URL` set with your SambaNova Cloud URL.
        `SAMBANOVA_API_KEY` set with your SambaNova Cloud API Key.
        http://cloud.sambanova.ai/

    Example:
        .. code-block:: python
            ChatSambaNovaCloud(
                sambanova_url = SambaNova cloud endpoint URL,
                sambanova_api_key = set with your SambaNova cloud API key,
                model = model name,
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                stream_options = include usage to get generation metrics
                model_kwargs: Optional = Extra Key word arguments to pass to the model.
            )

    Key init args — completion params:
        model: str
            The name of the model to use, e.g., Meta-Llama-3-70B-Instruct.
        streaming: bool
            Whether to use streaming handler when using non streaming methods
        max_tokens: int
            max tokens to generate
        temperature: float
            model temperature
        top_p: float
            model top p
        stream_options: dict
            stream options, include usage to get generation metrics
        model_kwargs: dict
            Extra Key word arguments to pass to the model.

    Key init args — client params:
        sambanova_url: str
            SambaNova Cloud Url
        sambanova_api_key: str
            SambaNova Cloud api key

    Instantiate:
        .. code-block:: python

            from langchain_sambanova import ChatSambaNovaCloud

            chat = ChatSambaNovaCloud(
                sambanova_url = SambaNova cloud endpoint URL,
                sambanova_api_key = set with your SambaNova cloud API key,
                model = model name,
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                stream_options = include usage to get generation metrics,
                model_kwargs: Optional = Extra Key word arguments to pass to the model.
            )

    Invoke:
        .. code-block:: python

            messages = [
                SystemMessage(content="your are an AI assistant."),
                HumanMessage(content="tell me a joke."),
            ]
            response = chat.invoke(messages)

    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk.content, end="", flush=True)

    Async:
        .. code-block:: python

            response = chat.ainvoke(messages)
            await response

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field


            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. Los Angeles, CA")


            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke("Should I bring my umbrella today in LA?")
            ai_msg.tool_calls

        .. code-block:: none

            [
                {
                    'name': 'GetWeather',
                    'args': {'location': 'Los Angeles, CA'},
                    'id': 'call_adf61180ea2b4d228a'
                }
            ]

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field


            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")


            structured_model = llm.with_structured_output(Joke)
            structured_model.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(setup="Why did the cat join a band?", punchline="Because it wanted to be the purr-cussionist!")

        See `ChatSambanovaCloud.with_structured_output()` for more.

    Token usage:
        .. code-block:: python

            response = chat.invoke(messages)
            print(response.usage_metadata["input_tokens"])
            print(response.usage_metadata["completion_tokens"])
            print(response.usage_metadata["total_tokens"])

    Response metadata
        .. code-block:: python

            response = chat.invoke(messages)
            print(response.response_metadata)

    """  # noqa: E501

    sambanova_url: str = Field(default="")
    """SambaNova Cloud Url"""

    sambanova_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("SAMBANOVA_API_KEY", default=None),
    )
    """SambaNova Cloud api key"""

    model: str = Field(default="Meta-Llama-3.1-8B-Instruct")
    """The name of the model"""

    streaming: bool = Field(default=False)
    """Whether to use streaming handler when using non streaming methods"""

    max_tokens: int = Field(default=1024)
    """max tokens to generate"""

    temperature: float = Field(default=0.7)
    """model temperature"""

    top_p: Optional[float] = Field(default=None)
    """model top p"""

    stream_options: dict[str, Any] = Field(default={"include_usage": True})
    """stream options, include usage to get generation metrics"""

    additional_headers: dict[str, Any] = Field(default={})
    """Additional headers to sent in request"""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain_sambanova", "chat_models"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"sambanova_api_key": "sambanova_api_key"}

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model": self.model,
            "streaming": self.streaming,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream_options": self.stream_options,
            "model_kwargs": self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "sambanovacloud-chatmodel"

    def __init__(self, **kwargs: Any) -> None:
        warnings.warn(
            "ChatSambaNovaCloud is deprecated and will be removed in a future version. "
            "Use ChatSambaNova instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        """init and validate environment variables"""
        url = get_from_dict_or_env(
            kwargs,
            "sambanova_url",
            "SAMBANOVA_URL",
            default="https://api.sambanova.ai/v1/",
        )
        url = url.replace("embeddings", "")
        url = url.replace("chat/completions", "")
        kwargs["sambanova_url"] = urllib.parse.urljoin(url + "/", "chat/completions")
        kwargs["sambanova_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(kwargs, "sambanova_api_key", "SAMBANOVA_API_KEY")
        )
        super().__init__(**kwargs)

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type[Any], Callable[..., Any], BaseTool]],
        *,
        tool_choice: Optional[Union[dict[str, Any], bool, str]] = None,
        parallel_tool_calls: Optional[bool] = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        tool_choice: does not currently support "any", choice like
        should be one of ["auto", "none", "required"]
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "required"):
                    tool_choice = "auto"
            elif isinstance(tool_choice, bool):
                if tool_choice:
                    tool_choice = "required"
            elif isinstance(tool_choice, dict):
                msg = "tool_choice must be one of ['auto', 'none', 'required']"
                raise ValueError(msg)
            else:
                msg = (
                    "Unrecognized tool_choice type. Expected str, bool"
                    f"Received: {tool_choice}"
                )
                raise ValueError(msg)
        else:
            tool_choice = "auto"
        kwargs["tool_choice"] = tool_choice
        kwargs["parallel_tool_calls"] = parallel_tool_calls
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[dict[str, Any], type[BaseModel]]] = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        strict: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[dict[str, Any], BaseModel]]:
        r"""Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema:
                The output schema. Can be passed in as:
                    - an OpenAI function/tool schema,
                    - a JSON Schema,
                    - a TypedDict class,
                    - or a Pydantic.BaseModel class.
                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.

            method:
                The method for steering model generation, either "function_calling"
                "json_mode" or "json_schema".
                If "function_calling" then the schema will be converted
                to an OpenAI function and the returned model will make use of the
                function-calling API. If "json_mode" or "json_schema" then OpenAI's
                JSON mode will be used.
                Note that if using "json_mode" then you must include instructions
                for formatting the output into the desired schema into the model call.

            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

            strict:
                - True:
                    Model output is guaranteed to exactly match the schema.
                    The input schema will also be validated
                - False:
                    Input schema will not be validated and model output will not be
                    validated.

            kwargs: Additional keyword args aren't supported.

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            If `include_raw` is False and `schema` is a Pydantic class, Runnable outputs
            an instance of `schema` (i.e., a Pydantic object).

            Otherwise, if `include_raw` is False then Runnable outputs a dict.

            If `include_raw` is True, then Runnable outputs a dict with keys:
                - `"raw"`: BaseMessage
                - `"parsed"`: None if there was a parsing error, otherwise the type depends on the `schema` as described above.
                - `"parsing_error"`: Optional[BaseException]

        Example: schema=Pydantic class, method="function_calling", include_raw=False:
            .. code-block:: python

                from typing import Optional

                from langchain_sambanova import ChatSambaNovaCloud
                from pydantic import BaseModel, Field


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str = Field(description="A justification for the answer.")


                llm = ChatSambaNovaCloud(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same.'
                # )

        Example: schema=Pydantic class, method="function_calling", include_raw=True:
            .. code-block:: python

                from langchain_sambanova import ChatSambaNovaCloud
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatSambaNovaCloud(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"answer": "They weigh the same.", "justification": "A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount."}', 'name': 'AnswerWithJustification'}, 'id': 'call_17a431fc6a4240e1bd', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'usage': {'acceptance_rate': 5, 'completion_tokens': 53, 'completion_tokens_after_first_per_sec': 343.7964936837758, 'completion_tokens_after_first_per_sec_first_ten': 439.1205661878638, 'completion_tokens_per_sec': 162.8511306784833, 'end_time': 1731527851.0698032, 'is_last_response': True, 'prompt_tokens': 213, 'start_time': 1731527850.7137961, 'time_to_first_token': 0.20475482940673828, 'total_latency': 0.32545061111450196, 'total_tokens': 266, 'total_tokens_per_sec': 817.3283162354066}, 'model_name': 'Meta-Llama-3.3-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731527850}, id='95667eaf-447f-4b53-bb6e-b6e1094ded88', tool_calls=[{'name': 'AnswerWithJustification', 'args': {'answer': 'They weigh the same.', 'justification': 'A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'}, 'id': 'call_17a431fc6a4240e1bd', 'type': 'tool_call'}]),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'),
                #     'parsing_error': None
                # }

        Example: schema=TypedDict class, method="function_calling", include_raw=False:
            .. code-block:: python

                # IMPORTANT: If you are using Python <=3.8, you need to import Annotated
                # from typing_extensions, not from typing.
                from typing_extensions import Annotated, TypedDict

                from langchain_sambanova import ChatSambaNovaCloud


                class AnswerWithJustification(TypedDict):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: Annotated[Optional[str], None, "A justification for the answer."]


                llm = ChatSambaNovaCloud(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'
                # }

        Example: schema=OpenAI function schema, method="function_calling", include_raw=False:
            .. code-block:: python

                from langchain_sambanova import ChatSambaNovaCloud

                oai_schema = {
                    "name": "AnswerWithJustification",
                    "description": "An answer to the user question along with justification for the answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "justification": {"description": "A justification for the answer.", "type": "string"},
                        },
                        "required": ["answer"],
                    },
                }

                llm = ChatSambaNovaCloud(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(oai_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'
                # }

        Example: schema=Pydantic class, method="json_mode", include_raw=True:
            .. code-block::

                from langchain_sambanova import ChatSambaNovaCloud
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatSambaNovaCloud(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_mode",
                    include_raw=True
                )

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n  "answer": "They are the same weight",\n  "justification": "A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities."\n}', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'usage': {'acceptance_rate': 5.3125, 'completion_tokens': 79, 'completion_tokens_after_first_per_sec': 292.65701089829776, 'completion_tokens_after_first_per_sec_first_ten': 346.43324678555325, 'completion_tokens_per_sec': 200.012158915008, 'end_time': 1731528071.1708555, 'is_last_response': True, 'prompt_tokens': 70, 'start_time': 1731528070.737394, 'time_to_first_token': 0.16693782806396484, 'total_latency': 0.3949759876026827, 'total_tokens': 149, 'total_tokens_per_sec': 377.2381225105847}, 'model_name': 'Meta-Llama-3.3-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731528070}, id='83208297-3eb9-4021-a856-ca78a15758df'),
                #     'parsed': AnswerWithJustification(answer='They are the same weight', justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities.'),
                #     'parsing_error': None
                # }

        Example: schema=None, method="json_mode", include_raw=True:
            .. code-block::

                from langchain_sambanova import ChatSambaNovaCloud

                llm = ChatSambaNovaCloud(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n  "answer": "They are the same weight",\n  "justification": "A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities."\n}', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'usage': {'acceptance_rate': 4.722222222222222, 'completion_tokens': 79, 'completion_tokens_after_first_per_sec': 357.1315485254867, 'completion_tokens_after_first_per_sec_first_ten': 416.83279609305305, 'completion_tokens_per_sec': 240.92819585198137, 'end_time': 1731528164.8474727, 'is_last_response': True, 'prompt_tokens': 70, 'start_time': 1731528164.4906917, 'time_to_first_token': 0.13837409019470215, 'total_latency': 0.3278985247892492, 'total_tokens': 149, 'total_tokens_per_sec': 454.4088757208256}, 'model_name': 'Meta-Llama-3.3-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731528164}, id='15261eaf-8a25-42ef-8ed5-f63d8bf5b1b0'),
                #     'parsed': {
                #         'answer': 'They are the same weight',
                #         'justification': 'A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities.'},
                #     },
                #     'parsing_error': None
                # }

        Example: schema=None, method="json_schema", include_raw=True:
            .. code-block::

                from langchain_sambanova import ChatSambaNovaCloud

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatSambaNovaCloud(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, method="json_schema", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n  "answer": "They are the same weight",\n  "justification": "A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities."\n}', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'usage': {'acceptance_rate': 5.3125, 'completion_tokens': 79, 'completion_tokens_after_first_per_sec': 292.65701089829776, 'completion_tokens_after_first_per_sec_first_ten': 346.43324678555325, 'completion_tokens_per_sec': 200.012158915008, 'end_time': 1731528071.1708555, 'is_last_response': True, 'prompt_tokens': 70, 'start_time': 1731528070.737394, 'time_to_first_token': 0.16693782806396484, 'total_latency': 0.3949759876026827, 'total_tokens': 149, 'total_tokens_per_sec': 377.2381225105847}, 'model_name': 'Meta-Llama-3.3-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731528070}, id='83208297-3eb9-4021-a856-ca78a15758df'),
                #     'parsed': AnswerWithJustification(answer='They are the same weight', justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities.'),
                #     'parsing_error': None
                # }

        """  # noqa: E501
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                msg = (
                    "`schema` must be specified when method is `function_calling`."
                    " Received None."
                )
                raise ValueError(msg)
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                structured_output_format={
                    "kwargs": {"method": "function_calling"},
                    "schema": schema,
                },
            )
            if is_pydantic_schema:
                output_parser: OutputParserLike[Any] = PydanticToolsParser(
                    tools=[schema],  # type: ignore  # noqa: PGH003
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                structured_output_format={
                    "kwargs": {"method": "json_mode"},
                    "schema": schema,
                },
            )
            if is_pydantic_schema:
                schema = cast(type[BaseModel], schema)
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                output_parser = JsonOutputParser()

        elif method == "json_schema":
            if schema is None:
                msg = (
                    "`schema` must be specified when method is not `json_mode`."
                    " Received None."
                )
                raise ValueError(msg)
            response_format = _convert_to_openai_response_format(schema, strict=strict)
            llm = self.bind(
                response_format=response_format,
                structured_output_format={
                    "kwargs": {"method": "json_schema", "strict": strict},
                    "schema": convert_to_openai_tool(schema),
                },
            )
            if is_pydantic_schema:
                schema = cast(type[BaseModel], schema)
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                output_parser = JsonOutputParser()
        else:
            msg = (
                f"Unrecognized method argument. Expected one of `function_calling` or "
                f"`json_mode`. Received: `{method}`"
            )
            raise ValueError(msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback

        return llm | output_parser

    def _handle_request(
        self,
        messages_dicts: list[dict[str, Any]],
        stop: Optional[list[str]] = None,
        *,
        streaming: Optional[bool] = False,
        **kwargs: Any,
    ) -> Response:
        """Performs a post request to the LLM API.

        Args:
            messages_dicts: List of role / content dicts to use as input.
            stop: list of stop tokens
            streaming: whether to do a streaming call
            **kwargs: Additional parameters passed to the underlying API client.

        Returns:
            An iterator of response dicts.
        """
        if streaming:
            data = {
                "messages": messages_dicts,
                "max_tokens": self.max_tokens,
                "stop": stop,
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": True,
                "stream_options": self.stream_options,
                **kwargs,
                **self.model_kwargs,
            }
        else:
            data = {
                "messages": messages_dicts,
                "max_tokens": self.max_tokens,
                "stop": stop,
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                **kwargs,
                **self.model_kwargs,
            }
        http_session = requests.Session()
        assert self.sambanova_api_key is not None  # noqa: S101
        response = http_session.post(
            self.sambanova_url,
            headers={
                "Authorization": f"Bearer {self.sambanova_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                **self.additional_headers,
            },
            json=data,
            stream=streaming,
        )
        if response.status_code != 200:
            msg = (
                f"Sambanova /complete call failed with status code "
                f"{response.status_code}.",
                f"{response.text}.",
            )
            raise RuntimeError(msg)
        return response

    def _process_response(self, response: Response) -> AIMessage:
        """Process a non streaming response from the api.

        Args:
            response: A request Response object

        Returns:
            generation: an AIMessage with model generation
        """
        try:
            response_dict = response.json()
            if response_dict.get("error"):
                msg = (
                    f"Sambanova /complete call failed with status code "
                    f"{response.status_code}."
                    f"{response_dict}."
                )
                raise RuntimeError(msg)
        except Exception as e:
            msg = (
                f"Sambanova /complete call failed couldn't get JSON response {e}"
                f"response: {response.text}"
            )
            raise RuntimeError(msg) from e
        content = response_dict["choices"][0]["message"].get("content", "")
        if content is None:
            content = ""
        additional_kwargs: dict[str, Any] = {}
        tool_calls = []
        invalid_tool_calls = []
        raw_tool_calls = response_dict["choices"][0]["message"].get("tool_calls")
        if raw_tool_calls:
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                if isinstance(raw_tool_call["function"]["arguments"], dict):
                    raw_tool_call["function"]["arguments"] = json.dumps(
                        raw_tool_call["function"].get("arguments", {})
                    )
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        if response_dict.get("usage") is not None:
            usage_metadata = {
                "input_tokens": response_dict.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": response_dict.get("usage", {}).get(
                    "completion_tokens", 0
                ),
                "total_tokens": response_dict.get("usage", {}).get("total_tokens", 0),
            }
        else:
            usage_metadata = None
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            response_metadata={
                "finish_reason": response_dict["choices"][0]["finish_reason"],
                "usage": response_dict.get("usage"),
                "model_name": response_dict["model"],
                "system_fingerprint": response_dict["system_fingerprint"],
                "created": response_dict["created"],
            },
            usage_metadata=usage_metadata,
            id=response_dict["id"],
        )

    def _process_stream_response(
        self, response: Response
    ) -> Iterator[BaseMessageChunk]:
        """Process a streaming response from the api.

        Args:
            response: An iterable request Response object

        Yields:
            generation: an AIMessageChunk with model partial generation
        """
        try:
            import sseclient
        except ImportError as e:
            msg = (
                "could not import sseclient library"
                "Please install it with `pip install sseclient-py`."
            )
            raise ImportError(msg) from e

        client = sseclient.SSEClient(response)  # type: ignore  # noqa: PGH003

        for event in client.events():
            if event.event == "error_event":
                msg = (
                    f"Sambanova /complete call failed with status code "
                    f"{response.status_code}."
                    f"{event.data}."
                )
                raise RuntimeError(msg)

            try:
                # check if the response is a final event
                # in that case event data response is '[DONE]'
                if event.data != "[DONE]":
                    if isinstance(event.data, str):
                        data = json.loads(event.data)
                    else:
                        msg = (
                            f"Sambanova /complete call failed with status code "
                            f"{response.status_code}."
                            f"{event.data}."
                        )
                        raise RuntimeError(msg)
                    if data.get("error"):
                        msg = (
                            f"Sambanova /complete call failed with status code "
                            f"{response.status_code}."
                            f"{event.data}."
                        )
                        raise RuntimeError(msg)
                    metadata = {}
                    usage_metadata: Optional[UsageMetadata] = None
                    tool_calls = []
                    invalid_tool_calls = []
                    additional_kwargs = {}
                    if (
                        len(data["choices"]) > 0
                        and data["choices"][0].get("delta", {}) != {}
                    ):
                        finish_reason = data["choices"][0].get("finish_reason")
                        content = data["choices"][0]["delta"].get("content", "")
                        if content is None:
                            content = ""
                        response_id = data["id"]
                        raw_tool_calls = data["choices"][0]["delta"].get("tool_calls")
                        if raw_tool_calls:
                            additional_kwargs["tool_calls"] = raw_tool_calls
                            for raw_tool_call in raw_tool_calls:
                                if isinstance(
                                    raw_tool_call["function"]["arguments"], dict
                                ):
                                    raw_tool_call["function"]["arguments"] = json.dumps(
                                        raw_tool_call["function"].get("arguments", {})
                                    )
                                try:
                                    tool_calls.append(
                                        parse_tool_call(raw_tool_call, return_id=True)
                                    )
                                except Exception as e:
                                    invalid_tool_calls.append(
                                        make_invalid_tool_call(raw_tool_call, str(e))
                                    )
                    else:
                        content = ""
                        response_id = data["id"]
                        metadata = {
                            "finish_reason": finish_reason
                            or data["choices"][0].get("finish_reason"),
                            "usage": data.get("usage"),
                            "model_name": data.get("model"),
                            "system_fingerprint": data.get("system_fingerprint"),
                            "created": data.get("created"),
                        }
                    if data.get("usage") is not None:
                        usage_metadata = {
                            "input_tokens": data.get("usage", {}).get(
                                "prompt_tokens", 0
                            ),
                            "output_tokens": data.get("usage", {}).get(
                                "completion_tokens", 0
                            ),
                            "total_tokens": data.get("usage", {}).get(
                                "total_tokens", 0
                            ),
                        }
                    else:
                        usage_metadata = None
                    chunk = AIMessageChunk(
                        content=content,
                        id=response_id,
                        tool_calls=tool_calls,  # type: ignore  # noqa: PGH003
                        invalid_tool_calls=invalid_tool_calls,
                        additional_kwargs=additional_kwargs,
                        response_metadata=metadata,
                        usage_metadata=usage_metadata,
                    )
                    yield chunk

            except Exception as e:
                msg = (
                    f"Error getting content chunk raw streamed response: {e}"
                    f"data: {event.data}"
                )
                raise RuntimeError(msg) from e

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call SambaNovaCloud models.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
            kwargs: Additional args to pass in generation.

        Returns:
            result: ChatResult with model generation
        """
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            if stream_iter:
                return generate_from_stream(stream_iter)
        messages_dicts = _create_message_dicts(messages)
        response = self._handle_request(messages_dicts, stop, streaming=False, **kwargs)
        message = self._process_response(response)
        generation = ChatGeneration(
            message=message,
            generation_info={
                "finish_reason": message.response_metadata["finish_reason"]
            },
        )
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the SambaNovaCloud chat model.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
            kwargs: Additional args to pass in generation.

        Yields:
            chunk: ChatGenerationChunk with model partial generation
        """
        messages_dicts = _create_message_dicts(messages)
        response = self._handle_request(messages_dicts, stop, streaming=True, **kwargs)
        for ai_message_chunk in self._process_stream_response(response):
            chunk = ChatGenerationChunk(message=ai_message_chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


@deprecated(
    since="0.2.0",
    alternative="langchain_sambanova.ChatSambaNova",
    removal="1.0.0",
)
class ChatSambaStudio(BaseChatModel):
    """SambaStudio chat model.

    Setup:
        To use, you should have the environment variables:
        `SAMBASTUDIO_URL` set with your SambaStudio deployed endpoint URL.
        `SAMBASTUDIO_API_KEY` set with your SambaStudio deployed endpoint Key.
        https://docs.sambanova.ai/sambastudio/latest/index.html
        Example:

        .. code-block:: python

            ChatSambaStudio(
                sambastudio_url = set with your SambaStudio deployed endpoint URL,
                sambastudio_api_key = set with your SambaStudio deployed endpoint Key.
                model = model or expert name (set for Bundle endpoints),
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                do_sample = whether to do sample
                process_prompt = whether to process prompt
                    (set for Bundle generic v1 and v2 endpoints)
                stream_options = include usage to get generation metrics
                special_tokens = start, start_role, end_role, end special tokens
                    (set for Bundle generic v1 and v2 endpoints when process prompt
                     set to false or for StandAlone v1 and v2 endpoints)
                model_kwargs: Optional = Extra Key word arguments to pass to the model.
            )

    Key init args — completion params:
        model: str
            The name of the model to use, e.g., Meta-Llama-3-70B-Instruct-4096
            (set for Bundle endpoints).
        streaming: bool
            Whether to use streaming
        max_tokens: int handler when using non streaming methods
            max tokens to generate
        temperature: float
            model temperature
        top_p: float
            model top p
        do_sample: bool
            whether to do sample
        process_prompt:
            whether to process prompt (set for Bundle generic v1 and v2 endpoints)
        stream_options: dict
            stream options, include usage to get generation metrics
        special_tokens: dict
            start, start_role, end_role and end special tokens
            (set for Bundle generic v1 and v2 endpoints when process prompt set to false
             or for StandAlone v1 and v2 endpoints) default to llama3 special tokens
        model_kwargs: dict
            Extra Key word arguments to pass to the model.

    Key init args — client params:
        sambastudio_url: str
            SambaStudio endpoint Url
        sambastudio_api_key: str
            SambaStudio endpoint api key

    Instantiate:
        .. code-block:: python

            from langchain_sambanova import ChatSambaStudio

            chat = ChatSambaStudio=(
                sambastudio_url = set with your SambaStudio deployed endpoint URL,
                sambastudio_api_key = set with your SambaStudio deployed endpoint Key.
                model = model or expert name (set for Bundle endpoints),
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                do_sample = whether to do sample
                process_prompt = whether to process prompt
                    (set for Bundle generic v1 and v2 endpoints)
                stream_options = include usage to get generation metrics
                special_tokens = start, start_role, end_role, and special tokens
                    (set for Bundle generic v1 and v2 endpoints when process prompt
                     set to false or for StandAlone v1 and v2 endpoints)
                model_kwargs: Optional = Extra Key word arguments to pass to the model.
            )

    Invoke:
        .. code-block:: python

            messages = [
                SystemMessage(content="your are an AI assistant."),
                HumanMessage(content="tell me a joke."),
            ]
            response = chat.invoke(messages)

    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk.content, end="", flush=True)

    Async:
        .. code-block:: python

            response = chat.ainvoke(messages)
            await response

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field


            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. Los Angeles, CA")


            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke("Should I bring my umbrella today in LA?")
            ai_msg.tool_calls

        .. code-block:: python

            [{"name": "GetWeather", "args": {"location": "Los Angeles, CA"}, "id": "call_adf61180ea2b4d228a"}]

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field


            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")


            structured_model = llm.with_structured_output(Joke)
            structured_model.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(setup="Why did the cat join a band?", punchline="Because it wanted to be the purr-cussionist!")

        See `ChatSambaStudio.with_structured_output()` for more.

    Token usage:
        .. code-block:: python

            response = chat.invoke(messages)
            print(response.usage_metadata["input_tokens"])
            print(response.usage_metadata["completion_tokens"])
            print(response.usage_metadata["total_tokens"])

    Response metadata
        .. code-block:: python

            response = chat.invoke(messages)
            print(response.response_metadata)
    """  # noqa: E501

    sambastudio_url: str = Field(default="")
    """SambaStudio Url"""

    sambastudio_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("SAMBASTUDIO_API_KEY", default=None),
    )
    """SambaStudio api key"""

    non_streaming_url: str = Field(default="", exclude=True)
    """SambaStudio non streaming Url"""

    streaming_url: str = Field(default="", exclude=True)
    """SambaStudio streaming Url"""

    model: Optional[str] = Field(default=None)
    """The name of the model or expert to use (for Bundle endpoints)"""

    streaming: bool = Field(default=False)
    """Whether to use streaming handler when using non streaming methods"""

    max_tokens: int = Field(default=1024)
    """max tokens to generate"""

    temperature: Optional[float] = Field(default=0.7)
    """model temperature"""

    top_p: Optional[float] = Field(default=None)
    """model top p"""

    do_sample: Optional[bool] = Field(default=None)
    """whether to do sampling"""

    process_prompt: Optional[bool] = Field(default=True)
    """whether process prompt (for Bundle generic v1 and v2 endpoints)"""

    stream_options: dict[str, Any] = Field(default={"include_usage": True})
    """stream options, include usage to get generation metrics"""

    special_tokens: dict[str, Any] = Field(
        default={
            "start": "<|begin_of_text|>",
            "start_role": "<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>",
            "end_role": "<|eot_id|>",
            "end": "<|start_header_id|>assistant<|end_header_id|>\n",
        }
    )
    """start, start_role, end_role and end special tokens
    (set for Bundle generic v1 and v2 endpoints when process prompt set to false
     or for StandAlone v1 and v2 endpoints)
    default to llama3 special tokens"""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""

    additional_headers: dict[str, Any] = Field(default={})
    """Additional headers to send in request"""

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain_sambanova", "chat_models"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {
            "sambastudio_url": "sambastudio_url",
            "sambastudio_api_key": "sambastudio_api_key",
        }

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model": self.model,
            "streaming": self.streaming,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "process_prompt": self.process_prompt,
            "stream_options": self.stream_options,
            "special_tokens": self.special_tokens,
            "model_kwargs": self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "sambastudio-chatmodel"

    def __init__(self, **kwargs: Any) -> None:
        warnings.warn(
            "ChatSambaStudio is deprecated and will be removed in a future version. "
            "Use ChatSambaNova instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        """init and validate environment variables"""
        url = get_from_dict_or_env(kwargs, "sambastudio_url", "SAMBASTUDIO_URL")
        if "api/v2/predict/generic" not in url and "api/predict/generic" not in url:
            # if not generic sent is considered openAI compatible API
            url = url.replace("embeddings", "")
            url = url.replace("chat/completions", "")
            kwargs["sambastudio_url"] = urllib.parse.urljoin(
                url + "/", "chat/completions"
            )
        else:
            # in case is generic v1 or v2 API (full endpoint is in the env)
            kwargs["sambastudio_url"] = url
        kwargs["non_streaming_url"], kwargs["streaming_url"] = (
            self._get_sambastudio_urls(kwargs["sambastudio_url"])
        )
        kwargs["sambastudio_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(kwargs, "sambastudio_api_key", "SAMBASTUDIO_API_KEY")
        )

        super().__init__(**kwargs)

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type[Any], Callable[..., Any], BaseTool]],
        *,
        tool_choice: Optional[Union[dict[str, Any], bool, str]] = None,
        parallel_tool_calls: Optional[bool] = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        tool_choice: does not currently support "any", choice like
        should be one of ["auto", "none", "required"]
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "required"):
                    tool_choice = "auto"
            elif isinstance(tool_choice, bool):
                if tool_choice:
                    tool_choice = "required"
            elif isinstance(tool_choice, dict):
                msg = "tool_choice must be one of ['auto', 'none', 'required']"
                raise ValueError(msg)
            else:
                msg = (
                    f"Unrecognized tool_choice type. Expected str, bool"
                    f"Received: {tool_choice}"
                )
                raise ValueError(msg)
        else:
            tool_choice = "auto"
        kwargs["tool_choice"] = tool_choice
        kwargs["parallel_tool_calls"] = parallel_tool_calls
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[dict[str, Any], type[BaseModel]]] = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        strict: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[dict[str, Any], BaseModel]]:
        r"""Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema:
                The output schema. Can be passed in as:
                    - an OpenAI function/tool schema,
                    - a JSON Schema,
                    - a TypedDict class,
                    - or a Pydantic class.
                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.

            method:
                The method for steering model generation, either "function_calling"
                "json_mode" or "json_schema".
                If "function_calling" then the schema will be converted
                to an OpenAI function and the returned model will make use of the
                function-calling API. If "json_mode" or "json_schema" then OpenAI's
                JSON mode will be used.
                Note that if using "json_mode" then you must include instructions
                for formatting the output into the desired schema into the model call.

            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

            strict:
                - True:
                    Model output is guaranteed to exactly match the schema.
                    The input schema will also be validated
                - False:
                    Input schema will not be validated and model output will not be
                    validated.

            kwargs: Additional keyword args aren't supported.

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            If `include_raw` is False and `schema` is a Pydantic class, Runnable outputs
            an instance of `schema` (i.e., a Pydantic object).

            Otherwise, if `include_raw` is False then Runnable outputs a dict.

            If `include_raw` is True, then Runnable outputs a dict with keys:
                - `"raw"`: BaseMessage
                - `"parsed"`: None if there was a parsing error, otherwise the type depends on the `schema` as described above.
                - `"parsing_error"`: Optional[BaseException]

        Example: schema=Pydantic class, method="function_calling", include_raw=False:
            .. code-block:: python

                from typing import Optional

                from langchain_sambanova import ChatSambaStudio
                from pydantic import BaseModel, Field


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str = Field(description="A justification for the answer.")


                llm = ChatSambaStudio(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same.'
                # )

        Example: schema=Pydantic class, method="function_calling", include_raw=True:
            .. code-block:: python

                from langchain_sambanova import ChatSambaStudio
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatSambaStudio(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"answer": "They weigh the same.", "justification": "A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount."}', 'name': 'AnswerWithJustification'}, 'id': 'call_17a431fc6a4240e1bd', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'usage': {'acceptance_rate': 5, 'completion_tokens': 53, 'completion_tokens_after_first_per_sec': 343.7964936837758, 'completion_tokens_after_first_per_sec_first_ten': 439.1205661878638, 'completion_tokens_per_sec': 162.8511306784833, 'end_time': 1731527851.0698032, 'is_last_response': True, 'prompt_tokens': 213, 'start_time': 1731527850.7137961, 'time_to_first_token': 0.20475482940673828, 'total_latency': 0.32545061111450196, 'total_tokens': 266, 'total_tokens_per_sec': 817.3283162354066}, 'model_name': 'Meta-Llama-3.3-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731527850}, id='95667eaf-447f-4b53-bb6e-b6e1094ded88', tool_calls=[{'name': 'AnswerWithJustification', 'args': {'answer': 'They weigh the same.', 'justification': 'A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'}, 'id': 'call_17a431fc6a4240e1bd', 'type': 'tool_call'}]),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'),
                #     'parsing_error': None
                # }

        Example: schema=TypedDict class, method="function_calling", include_raw=False:
            .. code-block:: python

                # IMPORTANT: If you are using Python <=3.8, you need to import Annotated
                # from typing_extensions, not from typing.
                from typing_extensions import Annotated, TypedDict

                from langchain_sambanova import ChatSambaStudio


                class AnswerWithJustification(TypedDict):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: Annotated[Optional[str], None, "A justification for the answer."]


                llm = ChatSambaStudio(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'
                # }

        Example: schema=OpenAI function schema, method="function_calling", include_raw=False:
            .. code-block:: python

                from langchain_sambanova import ChatSambaStudio

                oai_schema = {
                    "name": "AnswerWithJustification",
                    "description": "An answer to the user question along with justification for the answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "justification": {"description": "A justification for the answer.", "type": "string"},
                        },
                        "required": ["answer"],
                    },
                }

                llm = ChatSambaStudio(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(oai_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'
                # }

        Example: schema=Pydantic class, method="json_mode", include_raw=True:
            .. code-block::

                from langchain_sambanova import ChatSambaStudio
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatSambaStudio(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_mode",
                    include_raw=True
                )

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n  "answer": "They are the same weight",\n  "justification": "A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities."\n}', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'usage': {'acceptance_rate': 5.3125, 'completion_tokens': 79, 'completion_tokens_after_first_per_sec': 292.65701089829776, 'completion_tokens_after_first_per_sec_first_ten': 346.43324678555325, 'completion_tokens_per_sec': 200.012158915008, 'end_time': 1731528071.1708555, 'is_last_response': True, 'prompt_tokens': 70, 'start_time': 1731528070.737394, 'time_to_first_token': 0.16693782806396484, 'total_latency': 0.3949759876026827, 'total_tokens': 149, 'total_tokens_per_sec': 377.2381225105847}, 'model_name': 'Meta-Llama-3.3-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731528070}, id='83208297-3eb9-4021-a856-ca78a15758df'),
                #     'parsed': AnswerWithJustification(answer='They are the same weight', justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities.'),
                #     'parsing_error': None
                # }

        Example: schema=None, method="json_mode", include_raw=True:
            .. code-block::

                from langchain_sambanova import ChatSambaStudio

                llm = ChatSambaStudio(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n  "answer": "They are the same weight",\n  "justification": "A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities."\n}', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'usage': {'acceptance_rate': 4.722222222222222, 'completion_tokens': 79, 'completion_tokens_after_first_per_sec': 357.1315485254867, 'completion_tokens_after_first_per_sec_first_ten': 416.83279609305305, 'completion_tokens_per_sec': 240.92819585198137, 'end_time': 1731528164.8474727, 'is_last_response': True, 'prompt_tokens': 70, 'start_time': 1731528164.4906917, 'time_to_first_token': 0.13837409019470215, 'total_latency': 0.3278985247892492, 'total_tokens': 149, 'total_tokens_per_sec': 454.4088757208256}, 'model_name': 'Meta-Llama-3.3-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731528164}, id='15261eaf-8a25-42ef-8ed5-f63d8bf5b1b0'),
                #     'parsed': {
                #         'answer': 'They are the same weight',
                #         'justification': 'A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities.'},
                #     },
                #     'parsing_error': None
                # }

        Example: schema=None, method="json_schema", include_raw=True:
            .. code-block::

                from langchain_sambanova import ChatSambaStudio

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatSambaStudio(model="Meta-Llama-3.3-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, method="json_schema", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n  "answer": "They are the same weight",\n  "justification": "A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities."\n}', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'usage': {'acceptance_rate': 5.3125, 'completion_tokens': 79, 'completion_tokens_after_first_per_sec': 292.65701089829776, 'completion_tokens_after_first_per_sec_first_ten': 346.43324678555325, 'completion_tokens_per_sec': 200.012158915008, 'end_time': 1731528071.1708555, 'is_last_response': True, 'prompt_tokens': 70, 'start_time': 1731528070.737394, 'time_to_first_token': 0.16693782806396484, 'total_latency': 0.3949759876026827, 'total_tokens': 149, 'total_tokens_per_sec': 377.2381225105847}, 'model_name': 'Meta-Llama-3.3-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731528070}, id='83208297-3eb9-4021-a856-ca78a15758df'),
                #     'parsed': AnswerWithJustification(answer='They are the same weight', justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities.'),
                #     'parsing_error': None
                # }

        """  # noqa: E501
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
                raise ValueError(msg)
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                structured_output_format={
                    "kwargs": {"method": "function_calling"},
                    "schema": schema,
                },
            )
            if is_pydantic_schema:
                output_parser: OutputParserLike[Any] = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                structured_output_format={
                    "kwargs": {"method": "json_mode"},
                    "schema": schema,
                },
            )
            if is_pydantic_schema:
                schema = cast(type[BaseModel], schema)
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                output_parser = JsonOutputParser()

        elif method == "json_schema":
            if schema is None:
                msg = (
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
                raise ValueError(msg)
            response_format = _convert_to_openai_response_format(schema, strict=strict)
            llm = self.bind(
                response_format=response_format,
                structured_output_format={
                    "kwargs": {"method": "json_schema", "strict": strict},
                    "schema": convert_to_openai_tool(schema),
                },
            )
            if is_pydantic_schema:
                schema = cast(type[BaseModel], schema)
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                output_parser = JsonOutputParser()
        else:
            msg = (
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
            )
            raise ValueError(msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback

        return llm | output_parser

    def _get_role(self, message: BaseMessage) -> str:
        """Get the role of LangChain BaseMessage.

        Args:
            message: LangChain BaseMessage

        Returns:
            str: Role of the LangChain BaseMessage
        """
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = "tool"
        elif isinstance(message, ChatMessage):
            role = message.role
        else:
            msg = f"Got unknown type {message}"
            raise TypeError(msg)
        return role

    def _messages_to_string(self, messages: list[BaseMessage], **kwargs: Any) -> str:
        """Convert a list of BaseMessages to a string.

        - dumped json string with Role / content dict structure
            when process_prompt is true,
        - string with special tokens if process_prompt is false
        for generic V1 and V2 endpoints

        Args:
            messages: list of BaseMessages
            kwargs: Additional args

        Returns:
            str: string to send as model input depending on process_prompt param
        """
        if self.process_prompt:
            messages_dict: dict[str, Any] = {
                "conversation_id": "sambastudio-conversation-id",
                "messages": [],
                **kwargs,
            }
            for message in messages:
                if isinstance(message, AIMessage):
                    message_dict = {
                        "message_id": message.id,
                        "role": self._get_role(message),
                        "content": message.content,
                    }
                    if "tool_calls" in message.additional_kwargs:
                        message_dict["tool_calls"] = message.additional_kwargs[
                            "tool_calls"
                        ]
                        if message_dict["content"] == "":
                            message_dict["content"] = None

                elif isinstance(message, ToolMessage):
                    message_dict = {
                        "message_id": message.id,
                        "role": self._get_role(message),
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                    }

                else:
                    message_dict = {
                        "message_id": message.id,
                        "role": self._get_role(message),
                        "content": message.content,
                    }

                messages_dict["messages"].append(message_dict)

            messages_string = json.dumps(messages_dict)

        else:
            if "tools" in kwargs:
                msg = (
                    "tool calling not supported in API Generic V2 without"
                    " process_prompt, switch to OpenAI compatible API"
                    " or Generic V2 API with process_prompt=True"
                )
                raise NotImplementedError(msg)
            messages_string = self.special_tokens["start"]
            for message in messages:
                messages_string += self.special_tokens["start_role"].format(
                    role=self._get_role(message)
                )
                messages_string += f" {message.content} "
                messages_string += self.special_tokens["end_role"]
            messages_string += self.special_tokens["end"]

        return messages_string

    def _get_sambastudio_urls(self, url: str) -> tuple[str, str]:
        """Get streaming and non streaming URLs from the given URL.

        Args:
            url: string with sambastudio base or streaming endpoint url

        Returns:
            non_streaming_url: string with url to do non streaming calls
            streaming_url: string with url to do streaming calls
        """
        if "chat/completions" in url:
            non_streaming_url = url
            streaming_url = url
        else:
            if "stream" in url:
                non_streaming_url = url.replace("stream/", "")
                streaming_url = url
            else:
                non_streaming_url = url
                if "generic" in url:
                    streaming_url = "generic/stream".join(url.split("generic"))
                else:
                    msg = "Unsupported URL"
                    raise ValueError(msg)
        return non_streaming_url, streaming_url

    def _handle_request(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        *,
        streaming: Optional[bool] = False,
        **kwargs: Any,
    ) -> Response:
        """Performs a post request to the LLM API.

        Args:
            messages: List of role / content dicts to use as input.
            stop: list of stop tokens
            streaming: whether to do a streaming call
            kwargs: Additional parameters passed to the underlying API client.

        Returns:
            A request Response object
        """
        assert self.sambastudio_api_key is not None  # noqa: S101

        # create request payload for openai compatible API
        if "chat/completions" in self.sambastudio_url:
            messages_dicts = _create_message_dicts(messages)
            data = {
                "messages": messages_dicts,
                "max_tokens": self.max_tokens,
                "stop": stop,
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": streaming,
                "stream_options": self.stream_options,
                **kwargs,
                **self.model_kwargs,
            }
            data = {key: value for key, value in data.items() if value is not None}
            headers = {
                "Authorization": f"Bearer "
                f"{self.sambastudio_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                **self.additional_headers,
            }

        # create request payload for generic v2 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            items = [
                {"id": "item0", "value": self._messages_to_string(messages, **kwargs)}
            ]
            params: dict[str, Any] = {
                "select_expert": self.model,
                "process_prompt": self.process_prompt,
                "max_tokens_to_generate": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": self.do_sample,
                **kwargs,
                **self.model_kwargs,
            }
            params = {key: value for key, value in params.items() if value is not None}
            data = {"items": items, "params": params}
            headers = {
                "key": self.sambastudio_api_key.get_secret_value(),
                **self.additional_headers,
            }

        # create request payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            if "tools" in kwargs:
                msg = (
                    "tool calling not supported in API Generic V1, "
                    "switch to OpenAI compatible API or Generic V2 API"
                )
                raise NotImplementedError(msg)
            params = {
                "select_expert": self.model,
                "process_prompt": self.process_prompt,
                "max_tokens_to_generate": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": self.do_sample,
                **kwargs,
                **self.model_kwargs,
            }
            params = {
                key: {"type": type(value).__name__, "value": str(value)}
                for key, value in params.items()
                if value is not None
            }
            if streaming:
                data = {
                    "instance": self._messages_to_string(messages),
                    "params": params,
                }
            else:
                data = {
                    "instances": [self._messages_to_string(messages)],
                    "params": params,
                }
            headers = {
                "key": self.sambastudio_api_key.get_secret_value(),
                **self.additional_headers,
            }

        else:
            msg = (
                f"Unsupported URL{self.sambastudio_url} "
                "only openai, generic v1 and generic v2 APIs are supported"
            )
            raise ValueError(msg)

        http_session = requests.Session()
        if streaming:
            response = http_session.post(
                self.streaming_url, headers=headers, json=data, stream=True
            )
        else:
            response = http_session.post(
                self.non_streaming_url, headers=headers, json=data, stream=False
            )
        if response.status_code != 200:
            msg = (
                f"Sambanova /complete call failed with status code "
                f"{response.status_code}."
                f"{response.text}."
            )
            raise RuntimeError(msg)
        return response

    def _process_response(self, response: Response) -> AIMessage:
        """Process a non streaming response from the api.

        Args:
            response: A request Response object

        Returns:
            generation: an AIMessage with model generation
        """
        # Extract json payload form response
        try:
            response_dict = response.json()
        except Exception as e:
            msg = (
                f"Sambanova /complete call failed couldn't get JSON response {e}"
                f"response: {response.text}"
            )
            raise RuntimeError(msg) from e

        additional_kwargs: dict[str, Any] = {}
        tool_calls = []
        invalid_tool_calls = []

        # process response payload for openai compatible API
        if "chat/completions" in self.sambastudio_url:
            content = response_dict["choices"][0]["message"].get("content", "")
            if content is None:
                content = ""
            response_id = response_dict["id"]
            response_metadata = {
                "finish_reason": response_dict["choices"][0]["finish_reason"],
                "usage": response_dict.get("usage"),
                "model_name": response_dict["model"],
                "system_fingerprint": response_dict["system_fingerprint"],
                "created": response_dict["created"],
            }
            if response_dict.get("usage") is not None:
                usage_metadata = {
                    "input_tokens": response_dict.get("usage", {}).get(
                        "prompt_tokens", 0
                    ),
                    "output_tokens": response_dict.get("usage", {}).get(
                        "completion_tokens", 0
                    ),
                    "total_tokens": response_dict.get("usage", {}).get(
                        "total_tokens", 0
                    ),
                }
            else:
                usage_metadata = None
            raw_tool_calls = response_dict["choices"][0]["message"].get("tool_calls")
            if raw_tool_calls:
                additional_kwargs["tool_calls"] = raw_tool_calls
                for raw_tool_call in raw_tool_calls:
                    if isinstance(raw_tool_call["function"]["arguments"], dict):
                        raw_tool_call["function"]["arguments"] = json.dumps(
                            raw_tool_call["function"].get("arguments", {})
                        )
                    try:
                        tool_calls.append(
                            parse_tool_call(raw_tool_call, return_id=True)
                        )
                    except Exception as e:
                        invalid_tool_calls.append(
                            make_invalid_tool_call(raw_tool_call, str(e))
                        )

        # process response payload for generic v2 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            content = response_dict["items"][0]["value"]["completion"]
            response_id = response_dict["items"][0]["id"]
            response_metadata = response_dict["items"][0]
            usage_metadata = {
                "input_tokens": response_dict["items"][0]["value"].get(
                    "prompt_tokens_count", 0
                ),
                "output_tokens": response_dict["items"][0]["value"].get(
                    "completion_tokens_count", 0
                ),
                "total_tokens": response_dict["items"][0]["value"].get(
                    "total_tokens_count", 0
                ),
            }
            raw_tool_calls = response_dict["items"][0]["value"].get("tool_calls")
            if raw_tool_calls:
                additional_kwargs["tool_calls"] = raw_tool_calls
                for raw_tool_call in raw_tool_calls:
                    if isinstance(raw_tool_call["function"]["arguments"], dict):
                        raw_tool_call["function"]["arguments"] = json.dumps(
                            raw_tool_call["function"].get("arguments", {})
                        )
                    try:
                        tool_calls.append(
                            parse_tool_call(raw_tool_call, return_id=True)
                        )
                    except Exception as e:
                        invalid_tool_calls.append(
                            make_invalid_tool_call(raw_tool_call, str(e))
                        )

        # process response payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            content = response_dict["predictions"][0]["completion"]
            response_id = None
            response_metadata = response_dict
            usage_metadata = {
                "input_tokens": response_dict["predictions"][0].get(
                    "prompt_tokens_count", 0
                ),
                "output_tokens": response_dict["predictions"][0].get(
                    "completion_tokens_count", 0
                ),
                "total_tokens": response_dict["predictions"][0].get(
                    "total_tokens_count", 0
                ),
            }

        else:
            msg = (
                f"Unsupported URL{self.sambastudio_url} "
                "only openai, generic v1 and generic v2 APIs are supported"
            )
            raise ValueError(msg)

        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            response_metadata=response_metadata,
            usage_metadata=usage_metadata,
            id=response_id,
        )

    def _process_stream_response(
        self, response: Response
    ) -> Iterator[BaseMessageChunk]:
        """Process a streaming response from the api.

        Args:
            response: An iterable request Response object

        Yields:
            generation: an AIMessageChunk with model partial generation
        """
        try:
            import sseclient
        except ImportError as e:
            msg = (
                "could not import sseclient library"
                "Please install it with `pip install sseclient-py`."
            )
            raise ImportError(msg) from e

        # process response payload for openai compatible API
        if "chat/completions" in self.sambastudio_url:
            client = sseclient.SSEClient(response)  # type: ignore  # noqa: PGH003

            for event in client.events():
                if event.event == "error_event":
                    msg = (
                        f"Sambanova /complete call failed with status code "
                        f"{response.status_code}."
                        f"{event.data}."
                    )
                    raise RuntimeError(msg)
                try:
                    # check if the response is a final event
                    # in that case event data response is '[DONE]'
                    if event.data != "[DONE]":
                        if isinstance(event.data, str):
                            data = json.loads(event.data)
                        else:
                            msg = (
                                f"Sambanova /complete call failed with status code "
                                f"{response.status_code}."
                                f"{event.data}."
                            )
                            raise RuntimeError(msg)
                        if data.get("error"):
                            msg = (
                                f"Sambanova /complete call failed with status code "
                                f"{response.status_code}."
                                f"{event.data}."
                            )
                            raise RuntimeError(msg)
                        metadata = {}
                        usage_metadata: Optional[UsageMetadata] = None
                        tool_calls = []
                        invalid_tool_calls = []
                        additional_kwargs = {}
                        if (
                            len(data["choices"]) > 0
                            and data["choices"][0].get("delta", {}) != {}
                        ):
                            finish_reason = data["choices"][0].get("finish_reason")
                            content = data["choices"][0]["delta"].get("content", "")
                            if content is None:
                                content = ""
                            response_id = data["id"]
                            raw_tool_calls = data["choices"][0]["delta"].get(
                                "tool_calls"
                            )
                            if raw_tool_calls:
                                additional_kwargs["tool_calls"] = raw_tool_calls
                                for raw_tool_call in raw_tool_calls:
                                    if isinstance(
                                        raw_tool_call["function"]["arguments"], dict
                                    ):
                                        raw_tool_call["function"]["arguments"] = (
                                            json.dumps(
                                                raw_tool_call["function"].get(
                                                    "arguments", {}
                                                )
                                            )
                                        )
                                    try:
                                        tool_calls.append(
                                            parse_tool_call(
                                                raw_tool_call, return_id=True
                                            )
                                        )
                                    except Exception as e:
                                        invalid_tool_calls.append(
                                            make_invalid_tool_call(
                                                raw_tool_call, str(e)
                                            )
                                        )
                        else:
                            content = ""
                            response_id = data["id"]
                            metadata = {
                                "finish_reason": finish_reason
                                or data["choices"][0].get("finish_reason"),
                                "usage": data.get("usage"),
                                "model_name": data.get("model"),
                                "system_fingerprint": data.get("system_fingerprint"),
                                "created": data.get("created"),
                            }
                        if data.get("usage") is not None:
                            usage_metadata = {
                                "input_tokens": data.get("usage", {}).get(
                                    "prompt_tokens", 0
                                ),
                                "output_tokens": data.get("usage", {}).get(
                                    "completion_tokens", 0
                                ),
                                "total_tokens": data.get("usage", {}).get(
                                    "total_tokens", 0
                                ),
                            }
                        else:
                            usage_metadata = None
                        chunk = AIMessageChunk(
                            content=content,
                            id=response_id,
                            tool_calls=tool_calls,  # type: ignore  # noqa: PGH003
                            invalid_tool_calls=invalid_tool_calls,
                            additional_kwargs=additional_kwargs,
                            response_metadata=metadata,
                            usage_metadata=usage_metadata,
                        )
                        yield chunk

                except Exception as e:
                    msg = (
                        f"Error getting content chunk raw streamed response: {e}"
                        f"data: {event.data}"
                    )
                    raise RuntimeError(msg) from e

        # process response payload for generic v2 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            for line in response.iter_lines():
                try:
                    metadata = {}
                    usage_metadata = None
                    tool_calls = []
                    invalid_tool_calls = []
                    additional_kwargs = {}
                    data = json.loads(line)
                    content = data["result"]["items"][0]["value"]["stream_token"]
                    response_id = data["result"]["items"][0]["id"]
                    raw_tool_calls = data["result"]["items"][0]["value"].get(
                        "tool_calls"
                    )
                    if raw_tool_calls:
                        additional_kwargs["tool_calls"] = raw_tool_calls
                        for raw_tool_call in raw_tool_calls:
                            if isinstance(raw_tool_call["function"]["arguments"], dict):
                                raw_tool_call["function"]["arguments"] = json.dumps(
                                    raw_tool_call["function"].get("arguments", {})
                                )
                            try:
                                tool_calls.append(
                                    parse_tool_call(raw_tool_call, return_id=True)
                                )
                            except Exception as e:
                                invalid_tool_calls.append(
                                    make_invalid_tool_call(raw_tool_call, str(e))
                                )
                    if data["result"]["items"][0]["value"]["is_last_response"]:
                        metadata = {
                            "finish_reason": data["result"]["items"][0]["value"].get(
                                "stop_reason"
                            ),
                            "prompt": data["result"]["items"][0]["value"].get("prompt"),
                            "usage": {
                                "prompt_tokens_count": data["result"]["items"][0][
                                    "value"
                                ].get("prompt_tokens_count"),
                                "completion_tokens_count": data["result"]["items"][0][
                                    "value"
                                ].get("completion_tokens_count"),
                                "total_tokens_count": data["result"]["items"][0][
                                    "value"
                                ].get("total_tokens_count"),
                                "start_time": data["result"]["items"][0]["value"].get(
                                    "start_time"
                                ),
                                "end_time": data["result"]["items"][0]["value"].get(
                                    "end_time"
                                ),
                                "model_execution_time": data["result"]["items"][0][
                                    "value"
                                ].get("model_execution_time"),
                                "time_to_first_token": data["result"]["items"][0][
                                    "value"
                                ].get("time_to_first_token"),
                                "throughput_after_first_token": data["result"]["items"][
                                    0
                                ]["value"].get("throughput_after_first_token"),
                                "batch_size_used": data["result"]["items"][0][
                                    "value"
                                ].get("batch_size_used"),
                            },
                        }
                        usage_metadata = {
                            "input_tokens": data["result"]["items"][0]["value"].get(
                                "prompt_tokens_count", 0
                            ),
                            "output_tokens": data["result"]["items"][0]["value"].get(
                                "completion_tokens_count", 0
                            ),
                            "total_tokens": data["result"]["items"][0]["value"].get(
                                "total_tokens_count", 0
                            ),
                        }
                    yield AIMessageChunk(
                        content=content,
                        id=response_id,
                        tool_calls=tool_calls,  # type: ignore  # noqa: PGH003
                        invalid_tool_calls=invalid_tool_calls,
                        response_metadata=metadata,
                        additional_kwargs=additional_kwargs,
                        usage_metadata=usage_metadata,
                    )

                except Exception as e:
                    msg = (
                        f"Error getting content chunk raw streamed response: {e}"
                        f"line: {line}"
                    )
                    raise RuntimeError(msg) from e

        # process response payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            for line in response.iter_lines():
                try:
                    data = json.loads(line)
                    content = data["result"]["responses"][0]["stream_token"]
                    response_id = None
                    metadata = {}
                    usage_metadata = None
                    if data["result"]["responses"][0]["is_last_response"]:
                        metadata = {
                            "finish_reason": data["result"]["responses"][0].get(
                                "stop_reason"
                            ),
                            "prompt": data["result"]["responses"][0].get("prompt"),
                            "usage": {
                                "prompt_tokens_count": data["result"]["responses"][
                                    0
                                ].get("prompt_tokens_count"),
                                "completion_tokens_count": data["result"]["responses"][
                                    0
                                ].get("completion_tokens_count"),
                                "total_tokens_count": data["result"]["responses"][
                                    0
                                ].get("total_tokens_count"),
                                "start_time": data["result"]["responses"][0].get(
                                    "start_time"
                                ),
                                "end_time": data["result"]["responses"][0].get(
                                    "end_time"
                                ),
                                "model_execution_time": data["result"]["responses"][
                                    0
                                ].get("model_execution_time"),
                                "time_to_first_token": data["result"]["responses"][
                                    0
                                ].get("time_to_first_token"),
                                "throughput_after_first_token": data["result"][
                                    "responses"
                                ][0].get("throughput_after_first_token"),
                                "batch_size_used": data["result"]["responses"][0].get(
                                    "batch_size_used"
                                ),
                            },
                        }
                        usage_metadata = {
                            "input_tokens": data["result"]["responses"][0].get(
                                "prompt_tokens_count", 0
                            ),
                            "output_tokens": data["result"]["responses"][0].get(
                                "completion_tokens_count", 0
                            ),
                            "total_tokens": data["result"]["responses"][0].get(
                                "total_tokens_count", 0
                            ),
                        }
                    yield AIMessageChunk(
                        content=content,
                        id=response_id,
                        response_metadata=metadata,
                        usage_metadata=usage_metadata,
                        additional_kwargs={},
                    )

                except Exception as e:
                    msg = (
                        f"Error getting content chunk raw streamed response: {e}"
                        f"line: {line}"
                    )
                    raise RuntimeError(msg) from e

        else:
            msg = (
                f"Unsupported URL{self.sambastudio_url} "
                "only openai, generic v1 and generic v2 APIs are supported"
            )
            raise ValueError(msg)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call SambaStudio models.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
            kwargs: Extra arguments to pass in generation

        Returns:
            result: ChatResult with model generation
        """
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            if stream_iter:
                return generate_from_stream(stream_iter)
        response = self._handle_request(messages, stop, streaming=False, **kwargs)
        message = self._process_response(response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the SambaStudio model.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
            kwargs: Extra arguments to pass in generation

        Yields:
            chunk: ChatGenerationChunk with model partial generation
        """
        response = self._handle_request(messages, stop, streaming=True, **kwargs)
        for ai_message_chunk in self._process_stream_response(response):
            chunk = ChatGenerationChunk(message=ai_message_chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


#
# Type conversion helpers
#
def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: LangChain BaseMessage message.

    Returns:
        messages_dict:  role / content dict

    """
    message_dict: dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_sambanova_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_sambanova_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
        # If tool calls only, content is None not empty string
        if "tool_calls" in message_dict and message_dict["content"] == "":
            message_dict["content"] = None
        # if reasoning in ai message additional kwargs add it to message_dict
        if "reasoning_content" in message.additional_kwargs:
            message_dict["reasoning"] = message.additional_kwargs["reasoning_content"]
        elif hasattr(message, "reasoning"):
            message_dict["reasoning"] = message.reasoning
        elif hasattr(message, "reasoning_content"):
            message_dict["reasoning"] = message.reasoning_content
        else:
            pass
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        msg = f"Got unknown type {message}"
        raise TypeError(msg)
    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.

    """
    id_ = _dict.get("id")
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    if role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: dict = {}
        if reasoning := _dict.get("reasoning"):
            additional_kwargs["reasoning_content"] = reasoning
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:  # pylint: disable=broad-except
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        return AIMessage(
            content=content,
            id=id_,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id"),
            additional_kwargs=additional_kwargs,
        )
    return ChatMessage(content=_dict.get("content", ""), role=role)  # type: ignore[arg-type]


def _create_message_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert a list of BaseMessages to a list of dictionaries with Role / content.

    deprecation removal in 1.0.0

    Args:
        messages: list of BaseMessages

    Returns:
        messages_dicts:  list of role / content dicts
    """
    return [_convert_message_to_dict(m) for m in messages]


def _convert_chunk_to_message_chunk(
    chunk: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    choice = chunk["choices"][0]
    _dict = choice["delta"]
    role = cast("str", _dict.get("role"))
    content = cast("str", _dict.get("content") or "")
    additional_kwargs: dict = {}
    tool_call_chunks: list[ToolCallChunk] = []
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        for rtc in raw_tool_calls:
            with contextlib.suppress(KeyError):
                tool_call_chunks.append(
                    tool_call_chunk(
                        name=rtc["function"].get("name"),
                        args=rtc["function"].get("arguments"),
                        id=rtc.get("id"),
                        index=rtc.get("index"),
                    )
                )
    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        if reasoning := _dict.get("reasoning"):
            additional_kwargs["reasoning_content"] = reasoning
        if usage := chunk.get("usage"):
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            usage_metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
            }
        else:
            usage_metadata = None
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            usage_metadata=usage_metadata,  # type: ignore[arg-type]
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    return default_class(content=content)  # type: ignore[call-arg]


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _convert_to_openai_response_format(
    schema: Union[dict[str, Any], type], *, strict: Optional[bool] = None
) -> Union[dict, TypeBaseModel]:  # type: ignore  # noqa: PGH003
    """Convert to openai response format.

    deprecation removal in 1.0.0
    Args:
        Schema Union[dict[str, Any], type]

    Returns:
        Union[dict, TypeBaseModel]
    """
    if isinstance(schema, BaseModel):
        schema = schema.model_dump()
    elif isinstance(schema, type) and issubclass(schema, BaseModel):
        schema = schema.model_json_schema()
    if (
        isinstance(schema, dict)
        and "json_schema" in schema
        and schema.get("type") == "json_schema"
    ):
        response_format = schema
    elif isinstance(schema, dict) and "name" in schema and "schema" in schema:
        response_format = {"type": "json_schema", "json_schema": schema}
    else:
        if strict is None:
            if isinstance(schema, dict) and isinstance(schema.get("strict"), bool):
                strict = schema["strict"]
            else:
                strict = False
        function = convert_to_openai_function(schema, strict=strict)
        function["schema"] = function.pop("parameters")
        response_format = {"type": "json_schema", "json_schema": function}

    if (
        strict is not None
        and strict is not response_format["json_schema"].get("strict")
        and isinstance(schema, dict)
    ):
        msg = (
            f"Output schema already has 'strict' value set to "
            f"{schema['json_schema']['strict']} but 'strict' also passed in to "
            f"with_structured_output as {strict}. Please make sure that "
            f"'strict' is only specified in one place."
        )
        raise ValueError(msg)
    return response_format


def _lc_tool_call_to_sambanova_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
        },
    }


def _lc_invalid_tool_call_to_sambanova_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }
