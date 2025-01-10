# langchain-sambanova

This package contains the LangChain integration with SambaNova

## Installation

```bash
pip install -U langchain-sambanova
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatSambaNova` class exposes chat models from SambaNova.

```python
from langchain_sambanova import ChatSambaNova

llm = ChatSambaNova()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`SambaNovaEmbeddings` class exposes embeddings from SambaNova.

```python
from langchain_sambanova import SambaNovaEmbeddings

embeddings = SambaNovaEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`SambaNovaLLM` class exposes LLMs from SambaNova.

```python
from langchain_sambanova import SambaNovaLLM

llm = SambaNovaLLM()
llm.invoke("The meaning of life is")
```
