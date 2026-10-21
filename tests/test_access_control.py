"""Tests for web API and Telegram bot access control."""
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")

from fastapi.testclient import TestClient
from telegram import Chat, Message, Update, User
from telegram.ext import ApplicationHandlerStop

import api
import bot
from config import _parse_id_set
from core.access import SlidingWindowRateLimiter

ACT_N1_PAYLOAD = {
    "organization_name": "ОАО Тест",
    "accident_location": "Склад",
    "victim_fio": "Иванов И.И.",
    "victim_profession": "Кладовщик",
    "incident_description": "Падение со стремянки",
}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_rate_limiter_blocks_after_limit_and_recovers():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(2, window_seconds=60, clock=clock)
    assert limiter.hit("a") == 0
    assert limiter.hit("a") == 0
    assert limiter.hit("a") == pytest.approx(60)
    assert limiter.hit("b") == 0
    clock.now = 60
    assert limiter.hit("a") == 0


def test_rate_limiter_disabled_with_zero_limit():
    limiter = SlidingWindowRateLimiter(0)
    assert all(limiter.hit("a") == 0 for _ in range(100))


def test_parse_id_set():
    assert _parse_id_set("") == frozenset()
    assert _parse_id_set(" 1, 2;3 ,") == frozenset({1, 2, 3})


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "WEB_API_KEY", "")
    monkeypatch.setattr(api, "_rate_limiter", SlidingWindowRateLimiter(100))
    return TestClient(api.app)


def test_health_is_public_and_has_security_headers(client, monkeypatch):
    monkeypatch.setattr(api, "WEB_API_KEY", "secret")
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_web_ui_has_csp(client):
    response = client.get("/")
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]


def test_api_key_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(api, "WEB_API_KEY", "secret")
    assert client.post("/api/documents/act-n1", json=ACT_N1_PAYLOAD).status_code == 401
    assert client.post(
        "/api/documents/act-n1", json=ACT_N1_PAYLOAD, headers={"X-API-Key": "wrong"}
    ).status_code == 401
    assert client.post(
        "/api/documents/act-n1", json=ACT_N1_PAYLOAD, headers={"X-API-Key": "secret"}
    ).status_code == 200
    assert client.post(
        "/api/documents/act-n1", json=ACT_N1_PAYLOAD, headers={"Authorization": "Bearer secret"}
    ).status_code == 200


def test_api_open_when_key_not_configured(client):
    assert client.post("/api/documents/act-n1", json=ACT_N1_PAYLOAD).status_code == 200


def test_api_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(api, "_rate_limiter", SlidingWindowRateLimiter(2))
    for _ in range(2):
        assert client.post("/api/documents/act-n1", json=ACT_N1_PAYLOAD).status_code == 200
    response = client.post("/api/documents/act-n1", json=ACT_N1_PAYLOAD)
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0


def test_oversized_request_rejected_before_processing(client):
    response = client.post(
        "/api/inspect",
        content=b"x",
        headers={"Content-Length": str(api.MAX_REQUEST_BYTES + 1), "Content-Type": "image/jpeg"},
    )
    assert response.status_code == 413


def test_cors_is_same_origin_by_default(monkeypatch):
    monkeypatch.delenv("WEB_CORS_ORIGINS", raising=False)
    assert api._cors_origins() == []
    monkeypatch.setenv("WEB_CORS_ORIGINS", "https://a.example, https://b.example")
    assert api._cors_origins() == ["https://a.example", "https://b.example"]


def _update(user_id: int, text: str) -> Update:
    chat = Chat(id=user_id, type="private")
    message = Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=chat,
        from_user=User(id=user_id, first_name="Test", is_bot=False),
        text=text,
    )
    return Update(update_id=1, message=message)


@pytest.fixture
def gate(monkeypatch):
    reply = AsyncMock()
    monkeypatch.setattr(bot, "_reply_gate", reply)
    monkeypatch.setattr(bot, "TELEGRAM_ALLOWED_USER_IDS", frozenset())
    monkeypatch.setattr(bot, "_user_rate_limiter", SlidingWindowRateLimiter(100))
    return reply


@pytest.mark.anyio
async def test_bot_gate_allows_everyone_without_allowlist(gate):
    await bot.access_gate(_update(1, "вопрос"), None)
    gate.assert_not_awaited()


@pytest.mark.anyio
async def test_bot_gate_denies_user_outside_allowlist(gate, monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_ALLOWED_USER_IDS", frozenset({42}))
    await bot.access_gate(_update(42, "вопрос"), None)
    with pytest.raises(ApplicationHandlerStop):
        await bot.access_gate(_update(7, "вопрос"), None)
    gate.assert_awaited_once()


@pytest.mark.anyio
async def test_bot_gate_rate_limits_but_not_start(gate, monkeypatch):
    monkeypatch.setattr(bot, "_user_rate_limiter", SlidingWindowRateLimiter(1))
    await bot.access_gate(_update(5, "первый"), None)
    with pytest.raises(ApplicationHandlerStop):
        await bot.access_gate(_update(5, "второй"), None)
    await bot.access_gate(_update(5, "/start"), None)
    assert gate.await_count == 1
