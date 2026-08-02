import pickle
from typing import Any, List, Optional, Tuple

from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_core.stores import BaseStore


class PickleFileStore(BaseStore):
    """Wraps LocalFileStore to serialize Document objects via pickle for local storage."""

    def __init__(self, path: str) -> None:
        self._store = LocalFileStore(path)

    def mget(self, keys: List[str]) -> List[Optional[Document]]:
        raw_values = self._store.mget(keys)
        return [pickle.loads(v) if v is not None else None for v in raw_values]

    def mset(self, key_value_pairs: List[Tuple[str, Document]]) -> None:
        self._store.mset([(k, pickle.dumps(v)) for k, v in key_value_pairs])

    def mdelete(self, keys: List[str]) -> None:
        self._store.mdelete(keys)

    def yield_keys(self, prefix: Optional[str] = None) -> Any:
        yield from self._store.yield_keys(prefix=prefix)

from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_splitters() -> Tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]:
    """Returns the standardized Parent and Child text splitters used across the system.

    Child chunk size is set to 700 characters (~175 tokens average) to stay safely
    within the all-MiniLM-L6-v2 embedding model's hard 256-token context window.
    Dense mathematical LaTeX text can tokenize at ~0.35 tokens/char, so 700 chars
    produces at most ~245 tokens — within the model's limit.

    Parent chunks (4000 chars / ~1000 tokens) are never embedded — they are only
    fed to the LLM for answer generation, which has a much larger context window.
    """
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    return parent_splitter, child_splitter
