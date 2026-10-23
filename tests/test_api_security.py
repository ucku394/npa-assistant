import os

import api


def test_cors_origins_are_closed_by_default(monkeypatch):
    monkeypatch.delenv("WEB_CORS_ORIGINS", raising=False)
    assert api._cors_origins() == []


def test_cors_origins_accept_explicit_allowlist(monkeypatch):
    monkeypatch.setenv("WEB_CORS_ORIGINS", "https://example.by, https://app.example.by")
    assert api._cors_origins() == ["https://example.by", "https://app.example.by"]
