"""
Document Loader — supports PDF, TXT, DOC/DOCX, Excel (xlsx/xls), CSV,
and a broad fallback for other common document formats.
"""

import os
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    CSVLoader,
    Docx2txtLoader,
)

try:
    from langchain_community.document_loaders import UnstructuredFileLoader
except Exception:  # pragma: no cover - optional dependency
    UnstructuredFileLoader = None

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".doc",
    ".docx",
    ".xlsx",
    ".xls",
    ".csv",
    ".md",
    ".rtf",
    ".html",
    ".htm",
    ".json",
    ".xml",
    ".pptx",
    ".ppt",
    ".odt",
}


def load_document(file_path: str) -> List[Document]:
    """
    Loads a document from the given file path.
    Supports: PDF, TXT, DOC/DOCX, Excel (xlsx/xls), CSV.

    Args:
        file_path: Absolute path to the file.

    Returns:
        List of LangChain Document objects.
    """
    ext = os.path.splitext(file_path)[1].lower()

    print(f"Loading document: {file_path} (type: {ext})")

    try:
        if ext == ".pdf":
            # PyMuPDFLoader — same as notebook PDF loading
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()

        elif ext == ".txt":
            # TextLoader — same as notebook text loading
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()

        elif ext == ".docx":
            # Docx2txtLoader for Word documents
            loader = Docx2txtLoader(file_path)
            documents = loader.load()

        elif ext == ".doc":
            # Fallback to unstructured for legacy Word documents
            loader = UnstructuredFileLoader(file_path)
            documents = loader.load()

        elif ext in (".xlsx", ".xls"):
            # Convert Excel to text chunks using pandas for reliability
            documents = _load_excel(file_path)

        elif ext == ".csv":
            # CSVLoader — LangChain community loader
            loader = CSVLoader(file_path, encoding="utf-8")
            documents = loader.load()

        elif ext in {".md", ".rtf", ".html", ".htm", ".json", ".xml", ".pptx", ".ppt", ".odt"}:
            # Broad fallback for other common document formats
            if UnstructuredFileLoader is None:
                raise ImportError(
                    "UnstructuredFileLoader is not available. Install the 'unstructured' dependency to load this file type."
                )
            loader = UnstructuredFileLoader(file_path)
            documents = loader.load()

        else:
            # Last-resort fallback for any other readable document format
            if UnstructuredFileLoader is None:
                raise ImportError(
                    "UnstructuredFileLoader is not available. Install the 'unstructured' dependency to load this file type."
                )
            loader = UnstructuredFileLoader(file_path)
            documents = loader.load()

        print(f"Loaded {len(documents)} document chunk(s) from {os.path.basename(file_path)}")
        return documents

    except UnicodeDecodeError:
        # Retry TXT with latin-1 encoding
        if ext == ".txt":
            loader = TextLoader(file_path, encoding="latin-1")
            return loader.load()
        raise
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise


def _load_excel(file_path: str) -> List[Document]:
    """
    Loads Excel file using pandas and converts each sheet to Document chunks.
    """
    documents = []
    filename = os.path.basename(file_path)

    try:
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            # Convert dataframe to text
            text = f"Sheet: {sheet_name}\n\n"
            text += df.to_string(index=False)
            doc = Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "sheet": sheet_name,
                    "rows": len(df),
                    "columns": len(df.columns),
                },
            )
            documents.append(doc)
    except Exception as e:
        print(f"Error reading Excel file {file_path}: {e}")
        raise

    return documents
