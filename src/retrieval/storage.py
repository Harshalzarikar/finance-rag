import pickle
from collections.abc import Iterator, Sequence

from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_core.stores import BaseStore


class PickleFileStore(BaseStore[str, Document]):
    """Wraps LocalFileStore to serialize Document objects via pickle for local storage."""

    def __init__(self, path: str) -> None:
        self._store = LocalFileStore(path)

    def mget(self, keys: Sequence[str]) -> list[Document | None]:
        raw_values = self._store.mget(keys)
        return [pickle.loads(value) if value is not None else None for value in raw_values]

    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        self._store.mset([(key, pickle.dumps(value)) for key, value in key_value_pairs])

    def mdelete(self, keys: Sequence[str]) -> None:
        self._store.mdelete(keys)

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        yield from self._store.yield_keys(prefix=prefix)
