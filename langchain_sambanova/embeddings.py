"""SambaNova embedding models."""

from typing import Any, Dict, Generator, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import BaseModel, Field, SecretStr


class SambaStudioEmbeddings(BaseModel, Embeddings):
    """SambaNova embedding models.

    Setup:
        To use, you should have the environment variables:
        `SAMBASTUDIO_URL` set with your SambaStudio deployed endpoint URL.
        `SAMBASTUDIO_API_KEY` set with your SambaStudio deployed endpoint Key.
        https://docs.sambanova.ai/sambastudio/latest/index.html

        Example:

        .. code-block:: python

            from langchain_sambanova import SambaStudioEmbeddings

            embeddings = SambaStudioEmbeddings(
                sambastudio_url=base_url,
                sambastudio_api_key=api_key,
                batch_size=32
                )
            (or)

            embeddings = SambaStudioEmbeddings(batch_size=32)

            (or)

            # bundle example
            embeddings = SambaStudioEmbeddings(
                batch_size=1,
                model:'e5-mistral-7b-instruct'
            )
    """

    sambastudio_url: str = Field(default="")
    """SambaStudio Url"""

    sambastudio_api_key: SecretStr = Field(default=SecretStr(""))
    """SambaStudio api key"""

    model: Optional[str] = Field(default=None)
    """The name of the model or expert to use (for Bundle endpoints)"""

    batch_size: int = Field(default=32)
    """Batch size for the embedding models"""

    model_kwargs: Optional[Dict[str, Any]] = None
    """Key word arguments to pass to the model."""

    additional_headers: Dict[str, Any] = Field(default={})
    """Additional headers to send in request"""

    class Config:
        populate_by_name = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "sambastudio_url": "sambastudio_url",
            "sambastudio_api_key": "sambastudio_api_key",
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor models.
        """
        return {
            "model": self.model,
            "batch_size": self.batch_size,
            "model_kwargs": self.model_kwargs,
        }

    def __init__(self, **kwargs: Any) -> None:
        """init and validate environment variables"""
        kwargs["sambastudio_url"] = get_from_dict_or_env(
            kwargs, "sambastudio_url", "SAMBASTUDIO_URL"
        )

        kwargs["sambastudio_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(kwargs, "sambastudio_api_key", "SAMBASTUDIO_API_KEY")
        )

        super().__init__(**kwargs)

    def _iterate_over_batches(self, texts: List[str], batch_size: int) -> Generator:
        """Generator for creating batches in the embed documents method
        Args:
            texts (List[str]): list of strings to embed
            batch_size (int, optional): batch size to be used for the embedding model.
            Will depend on the RDU endpoint used.
        Yields:
            List[str]: list (batch) of strings of size batch size
        """
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    def embed_documents(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Returns a list of embeddings for the given sentences.
        Args:
            texts (`List[str]`): List of texts to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings
            for the given sentences
        """
        if batch_size is None:
            batch_size = self.batch_size
        http_session = requests.Session()
        params: Dict[str, Any] = {}
        embeddings = []

        if "api/v2/predict/generic" in self.sambastudio_url:
            for batch in self._iterate_over_batches(texts, batch_size):
                items = [
                    {"id": f"item{i}", "value": item} for i, item in enumerate(batch)
                ]
                params = {"select_expert": self.model}
                if self.model_kwargs is not None:
                    params = {**params, **self.model_kwargs}
                params = {
                    key: value for key, value in params.items() if value is not None
                }
                data = {"items": items, "params": params}
                response = http_session.post(
                    self.sambastudio_url,
                    headers={
                        "key": self.sambastudio_api_key.get_secret_value(),
                        **self.additional_headers,
                    },
                    json=data,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code "
                        f"{response.status_code}.\n Details: {response.text}"
                    )
                try:
                    embedding = [item["value"] for item in response.json()["items"]]
                    embeddings.extend(embedding)
                except KeyError:
                    raise KeyError(
                        "'items' not found in endpoint response",
                        response.json(),
                    )

        elif "api/predict/generic" in self.sambastudio_url:
            for batch in self._iterate_over_batches(texts, batch_size):
                params = {"select_expert": self.model}
                if self.model_kwargs is not None:
                    params = {**params, **self.model_kwargs}
                params = {
                    key: {"type": type(value).__name__, "value": str(value)}
                    for key, value in params.items()
                    if value is not None
                }
                data = {"instances": batch, "params": params}
                response = http_session.post(
                    self.sambastudio_url,
                    headers={
                        "key": self.sambastudio_api_key.get_secret_value(),
                        **self.additional_headers,
                    },
                    json=data,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code "
                        f"{response.status_code}.\n Details: {response.text}"
                    )
                try:
                    embedding = response.json()["predictions"]
                    embeddings.extend(embedding)
                except KeyError:
                    raise KeyError(
                        "'predictions' not found in endpoint response",
                        response.json(),
                    )

        else:
            raise ValueError(
                f"Unsupported URL {self.sambastudio_url}"
                "only generic v1 and generic v2 APIs are supported"
            )

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings
            for the given sentences
        """
        http_session = requests.Session()
        params: Dict[str, Any] = {}

        if "api/v2/predict/generic" in self.sambastudio_url:
            params = {"select_expert": self.model}
            if self.model_kwargs is not None:
                params = {**params, **self.model_kwargs}
            data = {"items": [{"id": "item0", "value": text}], "params": params}
            response = http_session.post(
                self.sambastudio_url,
                headers={
                    "key": self.sambastudio_api_key.get_secret_value(),
                    **self.additional_headers,
                },
                json=data,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code "
                    f"{response.status_code}.\n Details: {response.text}"
                )
            try:
                embedding = response.json()["items"][0]["value"]
            except KeyError:
                raise KeyError(
                    "'items' not found in endpoint response",
                    response.json(),
                )

        elif "api/predict/generic" in self.sambastudio_url:
            params = {"select_expert": self.model}
            if self.model_kwargs is not None:
                params = {**params, **self.model_kwargs}
            params = {
                key: {"type": type(value).__name__, "value": str(value)}
                for key, value in params.items()
                if value is not None
            }
            data = {"instances": [text], "params": params}
            response = http_session.post(
                self.sambastudio_url,
                headers={
                    "key": self.sambastudio_api_key.get_secret_value(),
                    **self.additional_headers,
                },
                json=data,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code "
                    f"{response.status_code}.\n Details: {response.text}"
                )
            try:
                embedding = response.json()["predictions"][0]
            except KeyError:
                raise KeyError(
                    "'predictions' not found in endpoint response",
                    response.json(),
                )

        else:
            raise ValueError(
                f"Unsupported URL {self.sambastudio_url}"
                "only generic v1 and generic v2 APIs are supported"
            )

        return embedding
