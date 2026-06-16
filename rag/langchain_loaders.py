from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.settings import settings


def load_pdf_pages(pdf_path: str) -> List[Document]:
    """
    Returns one Document per page.
    Metadata includes 'page' (0-indexed in LangChain).
    """
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_documents(
    docs: List[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Splits documents while preserving per-page metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    if extra_metadata:
        for d in chunks:
            d.metadata.update(extra_metadata)

    return chunks
