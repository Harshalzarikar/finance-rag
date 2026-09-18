"""Tests for the ingestion manifest.

The manifest is what makes ingestion resumable, so its invalidation rules matter:
skipping a file is only correct when the vectors it produced are in the collection
this run targets.
"""

from __future__ import annotations

import json

from scripts.ingest import load_manifest, manifest_target, save_manifest, source_key
from src.config.settings import get_settings


def _write(path, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def test_source_key_is_relative_and_posix(tmp_path, monkeypatch):
    monkeypatch.setenv("RAW_PDFS_DIR", str(tmp_path / "corpus"))
    get_settings.cache_clear()

    key = source_key(str(tmp_path / "corpus" / "2024" / "paper one.pdf"))

    assert key == "2024/paper one.pdf"
    assert "\\" not in key


def test_source_key_contains_no_absolute_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("RAW_PDFS_DIR", str(tmp_path / "corpus"))
    get_settings.cache_clear()

    key = source_key(str(tmp_path / "corpus" / "nested" / "a.pdf"))

    assert not key.startswith(str(tmp_path))
    assert ":" not in key


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path):
    path = str(tmp_path / "manifest.json")

    save_manifest(path, {"a.pdf": {"sha256": "abc", "chunks": 3}})

    assert load_manifest(path) == {"a.pdf": {"sha256": "abc", "chunks": 3}}


def test_manifest_records_the_target_it_was_built_for(tmp_path):
    path = tmp_path / "manifest.json"

    save_manifest(str(path), {"a.pdf": {"sha256": "abc"}})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["target"] == manifest_target()
    assert payload["target"]["collection"]
    assert payload["target"]["embedding_model"]


def test_save_manifest_leaves_no_temporary_file(tmp_path):
    save_manifest(str(tmp_path / "manifest.json"), {"a.pdf": {}})

    assert not (tmp_path / "manifest.json.tmp").exists()


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


def test_manifest_without_a_target_is_ignored(tmp_path):
    """A pre-target manifest cannot be verified, so it must not suppress work."""
    path = tmp_path / "manifest.json"
    _write(path, {"a.pdf": {"sha256": "abc"}})

    assert load_manifest(str(path)) == {}


def test_manifest_is_ignored_when_the_collection_changes(tmp_path):
    path = tmp_path / "manifest.json"
    stale_target = {**manifest_target(), "collection": "some_other_collection"}
    _write(path, {"version": 2, "target": stale_target, "files": {"a.pdf": {"sha256": "abc"}}})

    assert load_manifest(str(path)) == {}


def test_manifest_is_ignored_when_the_embedding_model_changes(tmp_path):
    path = tmp_path / "manifest.json"
    stale_target = {**manifest_target(), "embedding_model": "BAAI/bge-large-en-v1.5", "embedding_dim": 1024}
    _write(path, {"version": 2, "target": stale_target, "files": {"a.pdf": {"sha256": "abc"}}})

    assert load_manifest(str(path)) == {}


def test_manifest_survives_a_target_that_matches(tmp_path):
    path = tmp_path / "manifest.json"
    _write(path, {"version": 2, "target": manifest_target(), "files": {"a.pdf": {"sha256": "abc"}}})

    assert load_manifest(str(path)) == {"a.pdf": {"sha256": "abc"}}


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


def test_load_manifest_returns_empty_for_corrupt_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")

    assert load_manifest(str(path)) == {}


def test_load_manifest_returns_empty_for_unexpected_shape(tmp_path):
    path = tmp_path / "manifest.json"
    _write(path, ["a", "b"])

    assert load_manifest(str(path)) == {}


def test_load_manifest_returns_empty_when_absent(tmp_path):
    assert load_manifest(str(tmp_path / "nothing.json")) == {}
