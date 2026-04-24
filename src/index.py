"""One-shot PDF indexing.

Loads every PDF under the configured database path, splits into chunks,
and writes them to a persistent Chroma collection. Run this once before
starting the app, and again whenever the PDFs change.

Usage:
    python src/index.py            # refuses if an index already exists
    python src/index.py --force    # wipes and rebuilds
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from globals import configs
from rag import get_embeddings, project_path


def build_index(force: bool) -> None:
    persist_dir = project_path(configs["rag"]["persist_dir"])
    database_dir = project_path(configs["rag"]["database_path"])

    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        if not force:
            print(
                f"Index already exists at {persist_dir!r}. "
                "Use --force to rebuild."
            )
            return
        print(f"Wiping existing index at {persist_dir!r}...")
        shutil.rmtree(persist_dir)

    print(f"Loading PDFs from {database_dir!r}...")
    loader = DirectoryLoader(path=database_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise SystemExit(f"No PDFs found in {database_dir!r}.")
    print(f"Loaded {len(documents)} pages across the PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=configs["rag"]["chunk_size"],
        chunk_overlap=configs["rag"]["chunk_overlap"],
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks. Embedding and writing to Chroma...")

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=configs["rag"]["collection_name"],
        persist_directory=persist_dir,
    )
    print(f"Done. Index persisted at {persist_dir!r}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Wipe and rebuild the index.")
    args = parser.parse_args()
    build_index(force=args.force)
