import sys
from types import ModuleType

import pytest
from vidgen_core import observability
from vidgen_core.observability import ObservabilityStatus, init_observability


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIDGEN_SENTRY_DSN", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)


def test_noop_when_unset() -> None:
    assert init_observability("svc") == ObservabilityStatus(sentry=False, tracing=False)


def test_missing_packages_do_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIDGEN_SENTRY_DSN", "https://k@example.invalid/1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)  # forces ImportError
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    assert init_observability("svc") == ObservabilityStatus(sentry=False, tracing=False)


def test_sentry_initialised_with_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    fake = ModuleType("sentry_sdk")
    fake.init = lambda **kw: calls.update(kw)  # type: ignore[attr-defined]
    fake.set_tag = lambda k, v: calls.update({f"tag_{k}": v})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
    monkeypatch.setenv("VIDGEN_SENTRY_DSN", "https://k@example.invalid/1")
    status = init_observability("api")
    assert status.sentry and not status.tracing
    assert calls["dsn"] == "https://k@example.invalid/1"
    assert calls["tag_service"] == "api"


def test_tracing_not_attempted_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: str) -> bool:
        raise AssertionError("must not be called")

    monkeypatch.setattr(observability, "_init_tracing", boom)
    assert init_observability("svc").tracing is False
