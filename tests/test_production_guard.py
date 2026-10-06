"""Production configuration guard tests."""

from __future__ import annotations

import pytest

from src.config.production_guard import collect_production_errors
from src.config.settings import Settings, get_settings


def _prod_base(**overrides):
    base = dict(
        app_env="production",
        site_domain="app.example.com",
        caddy_email="ops@example.com",
        groq_api_key="gsk_test",
        cohere_api_key="cohere_test",
        database_url="postgresql+psycopg://raguser:goodlongrandomsecretpassword@postgres:5432/ragdb",
        jwt_secret="x" * 40,
        admin_api_key="a" * 30,
        public_signup_enabled=False,
        cors_origins=["https://app.example.com"],
    )
    base.update(overrides)
    return base


def test_development_allows_weak_defaults():
    settings = Settings(app_env="development", database_url="postgresql+psycopg://u:ragpassword@h/db")
    assert collect_production_errors(settings) == []


def test_production_rejects_weak_postgres_password():
    settings = Settings.model_construct(**_prod_base(database_url="postgresql+psycopg://raguser:ragpassword@postgres:5432/ragdb"))
    errors = collect_production_errors(settings)
    assert any("weak Postgres password" in item for item in errors)


def test_production_requires_jwt_and_admin(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="Production configuration invalid"):
        Settings(
            **_prod_base(jwt_secret="", admin_api_key=""),
        )


def test_production_accepts_localhost_cors_when_site_domain_set():
    settings = Settings.model_construct(
        **_prod_base(cors_origins=["http://localhost:5173"]),
    )
    assert "https://app.example.com" in settings.effective_cors_origins
    assert collect_production_errors(settings) == []
