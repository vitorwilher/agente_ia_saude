"""RAG primitives: embeddings, LLM, vector store, retriever.

Split from the previous monolithic script so the vector store can be built
once (via src/index.py) and loaded cheaply on every app startup.
"""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from globals import configs

_base_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_base_dir, ".."))

load_dotenv(os.path.join(_project_root, configs["server"]["env_file_path"]))


def project_path(relative: str) -> str:
    return os.path.join(_project_root, relative)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=configs["rag"]["embedding_model"]["name"],
        google_api_key=os.environ.get(configs["chat"]["llm_model"]["api_env_var_name"]),
        task_type=configs["rag"]["embedding_model"]["task_type"],
        transport="rest",
    )


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=configs["chat"]["llm_model"]["name"],
        google_api_key=os.environ.get(configs["chat"]["llm_model"]["api_env_var_name"]),
        temperature=configs["chat"]["llm_model"]["temperature"],
        max_output_tokens=configs["chat"]["llm_model"]["max_tokens"],
        top_k=configs["chat"]["llm_model"]["top_k"],
        top_p=configs["chat"]["llm_model"]["top_p"],
        transport="rest",
    )


def load_vector_store() -> Chroma:
    """Load the persisted Chroma store. Raises if it hasn't been built yet."""
    persist_dir = project_path(configs["rag"]["persist_dir"])
    if not os.path.isdir(persist_dir) or not os.listdir(persist_dir):
        raise RuntimeError(
            f"Vector store not found at {persist_dir!r}. "
            "Run `python src/index.py` first to index the PDFs."
        )
    return Chroma(
        collection_name=configs["rag"]["collection_name"],
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
    )


def get_retriever():
    return load_vector_store().as_retriever(
        search_type=configs["rag"]["retriever"]["search_type"],
        search_kwargs=configs["rag"]["retriever"]["search_kwargs"],
    )
