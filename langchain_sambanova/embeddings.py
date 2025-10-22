"""SambaNova embedding models."""

from __future__ import annotations

import logging
import warnings
from collections.abc import AsyncGenerator, Generator, Mapping
from typing import Any, Optional, Union

from langchain_core._api import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.utils import (
    from_env,
    get_pydantic_field_names,
    secret_from_env,
)
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from sambanova import AsyncSambaNova, SambaNova
from typing_extensions import Self

logger = logging.getLogger(__name__)


class SambaNovaEmbeddings(BaseModel, Embeddings):
    """SambaNova embedding models.

    Setup:
        Install ``langchain_sambanova`` and set environment variable
        ``SAMBANOVA_API_KEY`` with your SambaNova cloud or SambaStack API Key.
        Get one at https://cloud.sambanova.ai/

        .. code-block:: bash

            pip install -U langchain_sambanova
            export SAMBANOVA_API_KEY="your-api-key"

    Key init args — embedding params:
        model: str
            Name of model to use.

    Key init args — client params:
        api_key: Optional[SecretStr] = None
            SambaNova API key.
        base_url: Optional[str] = "https://api.sambanova.ai/v1/"
            SambaNova URL, set it when using a SambaStack environment.
        max_retries: Optional[int] = 2
            Maximum number of retries to make when generating.
        request_timeout: Optional[float] = 60.0
            Timeout for requests to SambaCloud or SambaStack completion API

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_sambanova import SambaNovaEmbeddings

            embeddings = SambaNovaEmbeddings(model="E5-Mistral-7B-Instruct")

    Embed single text:
        .. code-block:: python

            input_text = "The quick brown fox jumps over the lazy dog"
            vector = embeddings.embed_query("input_text")
            print(vector[:3])

        .. code-block:: python

            [-0.03124895128374129, -0.005862340914273618, 0.004587129384612357]

    Embed multiple texts:
        .. code-block:: python

            vectors = embeddings.embed_documents(["the brow fox", "the quick fox"])
            # Showing only the first 3 coordinates
            print(len(vectors))
            print(vectors[0][:3])

        .. code-block:: python

            2
            [-0.018742563917482904, -0.009134578224193842, 0.002856491732190473]

    Async:
        .. code-block:: python

            input_text = "The quick brown fox jumps over the lazy dog"
            vector = await embeddings.aembed_query(input_text)
            print(vector[:3])

            # multiple:
            # await embed.aembed_documents(input_texts)

        .. code-block:: python

            [-0.03124895128374129, -0.005862340914273618, 0.004587129384612357]

    """

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

    model: Optional[str] = Field(default="E5-Mistral-7B-Instruct")
    """The name of the model"""

    max_characters: int = Field(default=16384)
    """"max characters, longer will be trimmed"""

    batch_size: int = Field(default=32)
    """Batch size for the embedding models"""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

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

    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, protected_namespaces=()
    )

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
            self.client = SambaNova(**client_params, **sync_specific).embeddings
        if not (self.async_client or None):
            async_specific: dict[str, Any] = (
                {"http_client": self.http_async_client}
                if self.http_async_client
                else {}
            )
            self.async_client = AsyncSambaNova(
                **client_params, **async_specific
            ).embeddings
        return self

    @property
    def _invocation_params(self) -> dict[str, Any]:
        params: dict = {"model": self.model, **self.model_kwargs}
        return params

    def _iterate_over_batches(self, texts: list[str], batch_size: int) -> Generator:
        """Generator for creating batches in the embed documents method.

        Args:
            texts (List[str]): list of strings to embed
            batch_size (int, optional): batch size to be used for the embedding model.
            Will depend on the RDU endpoint used.

        Yields:
            List[str]: list (batch) of strings of size batch size
        """
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    async def _aiterate_over_batches(
        self, texts: list[str], batch_size: int
    ) -> AsyncGenerator:
        """Asynchronous generator for creating batches in the embed documents method.

        Args:
            texts (list[str]): List of strings to embed.
            batch_size (int): Batch size for the embedding model.

        Yields:
            list[str]: Batch of strings of size up to batch_size.
        """
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    def _trim_documents(self, texts: list[str], max_size: int) -> list[str]:
        """Trim each text to a maximum number of characters.

        Args:
            texts (list[str]): List of text documents.
            max_size (int): Maximum number of characters per text document.

        Returns:
            list[str]: List of trimmed text documents.
        """
        return [text[:max_size] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Returns a list of embeddings for the given sentences.

        Args:
            text: (`str`) sentence to encode

        Returns:
            `List[float]: Embeddings for the given sentence
        """
        params = self._invocation_params

        text = self._trim_documents([text], self.max_characters)[0]

        response = self.client.create(input=text, **params)

        if not isinstance(response, dict):
            response = response.model_dump()
        return response["data"][0]["embedding"]

    def embed_documents(
        self, texts: list[str], batch_size: Optional[int] = None
    ) -> list[list[float]]:
        """Returns a list of embeddings for the given sentences.

        Args:
            texts (`List[str]`): List of texts to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings
            for the given sentences
        """
        embeddings = []
        params = self._invocation_params
        texts = self._trim_documents(texts, self.max_characters)

        if batch_size is None:
            batch_size = self.batch_size
        for batch in self._iterate_over_batches(texts, batch_size):
            response = self.client.create(input=batch, **params)

            if not isinstance(response, dict):
                response = response.model_dump()
                embeddings.extend([i["embedding"] for i in response["data"]])
        return embeddings

    async def aembed_query(self, text: str) -> list[float]:
        """Returns asynchronously a list of embeddings for the given sentences.

        Args:
            text: (`str`) sentence to encode

        Returns:
            `List[float]: Embeddings for the given sentence
        """
        params = self._invocation_params
        params["model"] = params["model"]

        text = self._trim_documents([text], self.max_characters)[0]

        response = await self.async_client.create(input=text, **params)

        if not isinstance(response, dict):
            response = response.model_dump()
        return response["data"][0]["embedding"]

    async def aembed_documents(
        self, texts: list[str], batch_size: Optional[int] = None
    ) -> list[list[float]]:
        """Returns asynchronously a list of embeddings for the given sentences.

        Args:
            texts (`List[str]`): List of texts to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings
            for the given sentences
        """
        embeddings = []
        params = self._invocation_params
        params["model"] = params["model"]

        if batch_size is None:
            batch_size = self.batch_size

        async for batch in self._aiterate_over_batches(texts, batch_size):
            response = await self.async_client.create(input=batch, **params)

            if not isinstance(response, dict):
                response = response.model_dump()

            embeddings.extend([i["embedding"] for i in response["data"]])

        return embeddings


@deprecated(
    since="0.2.0",
    alternative="langchain_sambanova.SambaNovaEmbeddings",
    removal="1.0.0",
)
class SambaNovaCloudEmbeddings:
    """SambaNovaCloud embedding models deprecated since 0.2.0 and removed from 1.0.0.

    Alternative langchain_sambanova.SambaNovaEmbeddings
    """


@deprecated(
    since="0.2.0",
    alternative="langchain_sambanova.SambaNovaEmbeddings",
    removal="1.0.0",
)
class SambaStudioEmbeddings:
    """SambaStudio embedding models deprecated since 0.2.0 and removed from 1.0.0.

    Alternative langchain_sambanova.SambaNovaEmbeddings
    """
