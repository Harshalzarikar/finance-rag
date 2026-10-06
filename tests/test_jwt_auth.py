"""JWT helper tests."""

from __future__ import annotations

from src.api.jwt_auth import create_access_token, decode_access_token
from src.config.settings import get_settings


def test_jwt_round_trip(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "unit-test-secret")
    get_settings.cache_clear()
    settings = get_settings()

    token, expires_in = create_access_token(
        settings,
        user_id="user-1",
        tenant_id="acme",
        email="user@acme.com",
    )
    assert expires_in > 0

    payload = decode_access_token(settings, token)
    assert payload["sub"] == "user-1"
    assert payload["tenant_id"] == "acme"
    assert payload["email"] == "user@acme.com"
