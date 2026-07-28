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
