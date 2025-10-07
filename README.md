<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./img/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="./img/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

# langchain-sambanova

This package contains the LangChain integration with SambaNova

## Installation

```bash
pip install -U langchain-sambanova
```

And you should configure credentials by setting the following environment variables:

If you are a SambaCloud user:

```bash
export SAMBANOVA_API_KEY="your-sambacloud-api-key-here"
```
> You can obtain a free SambaCloud API key [here](https://cloud.sambanova.ai/)

If you are a SambaStack user:

```bash
export SAMBANOVA_API_BASE="your-sambastack-api-base-url-here"
export SAMBANOVA_API_KEY="your-sambastack-api-key-here"
```

or if you are SambaStudio User

```bash
export SAMBASTUDIO_URL="your-sambastudio-endpoint-url-here"
export SAMBASTUDIO_API_KEY="your-sambastudio-api-key-here"
```

## Chat Models

### SambaNova

`ChatSambaNova` class exposes chat models from SambaNova unified interface for SambaCloud and SambaStack.

```python
from langchain_sambanova import ChatSambaNova

llm = ChatSambaNova(
    model = "Llama-4-Maverick-17B-128E-Instruct",
    temperature = 0.7
)
llm.invoke("Tell me a joke about artificial intelligence.")
```

### SambaNova Cloud

⚠️ **Deprecated**: `ChatSambaNovaCloud` is deprecated in favor of [`ChatSambaNova`](#sambanova).
Please migrate to [`ChatSambaNova`](#sambanova) which unifies support for SambaCloud, SambaStack, and SambaStudio.

`ChatSambaNovaCloud` class exposes chat models from SambaNovaCloud.

```python
from langchain_sambanova import ChatSambaNovaCloud

llm = ChatSambaNovaCloud(
    model = "Meta-Llama-3.3-70B-Instruct",
    temperature = 0.7
)
llm.invoke("Tell me a joke about artificial intelligence.")
```

### SambaStudio

⚠️ **Deprecated**: `ChatSambaStudio` is deprecated in favor of [`ChatSambaNova`](#sambanova).
Please migrate to [`ChatSambaNova`](#sambanova) which unifies support for SambaCloud, SambaStack, and SambaStudio.

`ChatSambaStudio` class exposes chat models from SambaStudio Platform.

```python
from langchain_sambanova import ChatSambaStudio

llm = ChatSambaStudio(
    model = "Meta-Llama-3.3-70B-Instruct",
    temperature = 0.7
)
llm.invoke("Tell me a joke about artificial intelligence.")
```

## Embeddings

### SambaNova

`SambaNovaEmbeddings` class exposes embeddings from SambaNova unified interface for SambaCloud and SambaStack.

```python
from langchain_sambanova import SambaNovaEmbeddings

embeddings = SambaNovaEmbeddings(
    model="E5-Mistral-7B-Instruct"
)
embeddings.embed_query("What is the meaning of life?")
```

### SambaNova Cloud Embeddings

⚠️ **Deprecated**: `SambaNovaCloudEmbeddings` is deprecated in favor of [`SambaNovaEmbeddings`](#sambanova-1).
Please migrate to [`SambaNovaEmbeddings`](#sambanova-1) which unifies support for SambaCloud, SambaStack, and SambaStudio.

`SambaNovaCloudEmbeddings` class exposes embeddings from SambaNovaCloud.

```python
from langchain_sambanova import SambaNovaCloudEmbeddings

embeddings = SambaNovaCloudEmbeddings(
    model = "E5-Mistral-7B-Instruct"
)
embeddings.embed_query("What is the meaning of life?")
```

### SambaStudio Embeddings

⚠️ **Deprecated**: `SambaNovaCloudEmbeddings` is deprecated in favor of [`SambaNovaEmbeddings`](#sambanova-1).
Please migrate to [`SambaNovaEmbeddings`](#sambanova-1) which unifies support for SambaCloud, SambaStack, and SambaStudio.

`SambaStudioEmbeddings` class exposes embeddings from SambaStudio platform.

```python
from langchain_sambanova import SambaStudioEmbeddings

embeddings = SambaStudioEmbeddings(
    model = "e5-mistral-7b-instruct"
)
embeddings.embed_query("What is the meaning of life?")
```
