"""API contract tests.

The real pipeline is replaced with a stub, so nothing here loads a model or
contacts Qdrant, Redis, Groq, or Cohere.
"""

from __future__ import annotations

import json

from src.api.ratelimit import RateLimitResult
from src.config.settings import get_settings


def _sse_events(payload: str) -> list[dict]:
    return [json.loads(line[5:]) for line in payload.splitlines() if line.startswith("data:")]


# ---------------------------------------------------------------------------
# Service descriptor and liveness
# ---------------------------------------------------------------------------


def test_root_describes_the_service(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "Policy Intelligence RAG API"


def test_liveness_does_not_require_authentication(client):
    assert client.get("/health/live").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_chat_without_a_key_is_rejected(client):
    response = client.post("/chat", json={"query": "hello"})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] in {"ApiKey", "Bearer"}


def test_chat_with_a_wrong_key_is_rejected(client):
    response = client.post("/chat", json={"query": "hello"}, headers={"X-API-Key": "not-the-key"})

    assert response.status_code == 401


def test_authentication_can_be_disabled_by_configuration(client, monkeypatch):
    monkeypatch.setenv("API_KEYS", "")
    monkeypatch.setenv("DATABASE_URL", "")
    get_settings.cache_clear()

    assert client.post("/chat", json={"query": "hello"}).status_code == 200


def test_chat_requires_tenant_api_key_when_postgres_is_enabled(client, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://raguser:ragpassword@localhost:5432/ragdb")
    monkeypatch.setenv("API_KEYS", "")
    get_settings.cache_clear()

    response = client.post("/chat", json={"query": "hello"})

    assert response.status_code == 401
    assert "Sign in" in response.json()["detail"] or "X-API-Key" in response.json()["detail"]


def test_login_rejects_invalid_credentials(client, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://raguser:ragpassword@localhost:5432/ragdb")
    get_settings.cache_clear()
    monkeypatch.setattr("src.db.user_store.authenticate_user", lambda *_a, **_k: None)

    response = client.post("/auth/login", json={"email": "user@acme.com", "password": "wrongpass1"})

    assert response.status_code == 401


def test_signup_disabled_when_configured_off(client, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://raguser:ragpassword@localhost:5432/ragdb")
    monkeypatch.setenv("PUBLIC_SIGNUP_ENABLED", "false")
    get_settings.cache_clear()

    response = client.post(
        "/auth/signup",
        json={
            "email": "new@acme.com",
            "password": "Secret123!",
            "tenant_id": "acme-insurance",
            "full_name": "New User",
        },
    )

    assert response.status_code == 403


def test_tenants_me_requires_bearer_session(client, monkeypatch, auth_headers):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://raguser:ragpassword@localhost:5432/ragdb")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "src.db.tenant_store.get_tenant_by_api_key",
        lambda _url, key: (
            type("T", (), {"id": "acme", "name": "Acme", "plan": "pro", "is_active": True})()
            if key == auth_headers["X-API-Key"]
            else None
        ),
    )

    response = client.get("/tenants/me", headers=auth_headers)

    assert response.status_code == 401


def test_metrics_is_not_public(client):
    assert client.get("/metrics").status_code == 401


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_chat_returns_the_answer_with_citations(client, auth_headers):
    response = client.post("/chat", json={"query": "What is a volatility smile?"}, headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["sources"][0]["source"] == "paper.pdf"
    assert body["sources"][0]["page"] == 3
    assert body["cached"] is False
    assert body["request_id"]


def test_blank_query_is_rejected(client, auth_headers):
    assert client.post("/chat", json={"query": "   "}, headers=auth_headers).status_code == 422


def test_top_k_outside_the_allowed_range_is_rejected(client, auth_headers):
    for value in (0, 21, -1):
        response = client.post("/chat", json={"query": "hello", "top_k_rerank": value}, headers=auth_headers)
        assert response.status_code == 422, value


def test_overlong_query_is_rejected(client, auth_headers, monkeypatch):
    monkeypatch.setenv("MAX_QUERY_CHARS", "10")
    get_settings.cache_clear()

    response = client.post("/chat", json={"query": "x" * 11}, headers=auth_headers)

    assert response.status_code == 422


def test_top_k_is_forwarded_to_the_pipeline(client, auth_headers, fake_pipeline):
    client.post("/chat", json={"query": "hello", "top_k_rerank": 7}, headers=auth_headers)

    assert fake_pipeline.calls == [("hello", 7)]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_headers_are_returned(client, auth_headers):
    response = client.post("/chat", json={"query": "hello"}, headers=auth_headers)

    assert response.headers["X-RateLimit-Limit"] == "30"
    assert int(response.headers["X-RateLimit-Remaining"]) >= 0


def test_rate_limit_exceeded_returns_429(client, auth_headers, monkeypatch):
    from src.api import deps

    class _Denied:
        def check(self, identity: str) -> RateLimitResult:
            del identity
            return RateLimitResult(allowed=False, remaining=0, reset_seconds=42)

    monkeypatch.setattr(deps, "get_rate_limiter", _Denied)

    response = client.post("/chat", json={"query": "hello"}, headers=auth_headers)

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "42"


def test_rate_limit_is_checked_after_authentication(client, monkeypatch):
    """An unauthenticated caller must be rejected before spending rate-limit budget."""
    from src.api import deps

    class _Exploding:
        def check(self, identity: str) -> RateLimitResult:
            raise AssertionError("rate limiter must not run for unauthenticated requests")

    monkeypatch.setattr(deps, "get_rate_limiter", _Exploding)

    assert client.post("/chat", json={"query": "hello"}).status_code == 401


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


def test_request_id_is_echoed_back(client, auth_headers):
    response = client.post("/chat", json={"query": "hello"}, headers={**auth_headers, "X-Request-ID": "trace-me"})

    assert response.headers["X-Request-ID"] == "trace-me"


def test_request_id_is_generated_when_absent(client, auth_headers):
    response = client.post("/chat", json={"query": "hello"}, headers=auth_headers)

    assert len(response.headers["X-Request-ID"]) == 16


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_emits_sources_then_tokens_then_done(client, auth_headers):
    with client.stream("POST", "/chat/stream", json={"query": "explain garch"}, headers=auth_headers) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _sse_events("".join(response.iter_text()))

    assert events[0]["type"] == "sources"
    assert events[0]["sources"][0]["source"] == "paper.pdf"
    assert "".join(event["value"] for event in events if event["type"] == "token") == "A grounded answer."
    assert events[-1]["type"] == "done"


def test_stream_requires_authentication(client):
    assert client.post("/chat/stream", json={"query": "hello"}).status_code == 401


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_is_exposed_with_a_valid_key(client, auth_headers):
    # Metrics only appear once something has been observed.
    client.post("/chat", json={"query": "hello"}, headers=auth_headers)

    response = client.get("/metrics", headers=auth_headers)

    assert response.status_code == 200
    assert "http_requests" in response.text
