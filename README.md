# langchain-sambanova

This package contains the LangChain integration with SambaNova

## Installation

```bash
pip install -U langchain-sambanova
```

And you should configure credentials by setting the following environment variables:

If you are a SambaNovaCloud user:

```bash
export SAMBANOVA_API_KEY="your-sambanova-cloud-api-key-here"
```

or if you are SambaStudio User

```bash
export SAMBASTUDIO_API_KEY="your-sambastudio-api-key-here"
```
> You can obtain a free SambaNovaCloud API key [here](https://cloud.sambanova.ai/)

## Chat Models

### SambaNova Cloud

`ChatSambaNovaCloud` class exposes chat models from SambaNovaCloud.

```python
from langchain_sambanova import ChatSambaNovaCloud

llm = ChatSambaNovaCloud(
    model = "Meta-Llama-3.3-70B-Instruct"
    temperature = 0.7
)
llm.invoke("Tell me a joke about artificial intelligence.")
```

### SambaStudio

`ChatSambaStudio` class exposes chat models from SambaStudio Platform.

```python
from langchain_sambanova import ChatSambaStudio

llm = ChatSambaStudio(
    model = "Meta-Llama-3.3-70B-Instruct"
    temperature = 0.7
)
llm.invoke("Tell me a joke about artificial intelligence.")
```

## Embeddings

### SambaStudio Embeddings

`SambaStudioEmbeddings` class exposes embeddings from SambaStudio platform.

```python
from langchain_sambanova import SambaStudioEmbeddings

embeddings = SambaStudioEmbeddings(
    model = "e5-mistral-7b-instruct"
)
embeddings.embed_query("What is the meaning of life?")
```
