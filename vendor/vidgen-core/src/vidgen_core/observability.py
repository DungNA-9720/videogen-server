"""Optional Sentry and OpenTelemetry setup.

Both integrations are opt-in and imported lazily, so no extra dependency is needed unless the
matching environment variable is set. Install them with the `observability` extra.

- Sentry: enabled when VIDGEN_SENTRY_DSN is set (needs `sentry-sdk`).
- OpenTelemetry: a tracer provider is installed when `opentelemetry-sdk` is available and
  OTEL_EXPORTER_OTLP_ENDPOINT is set (also needs `opentelemetry-exporter-otlp`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from vidgen_core.logging import get_logger

SENTRY_DSN_ENV = "VIDGEN_SENTRY_DSN"
SENTRY_ENV_ENV = "VIDGEN_ENVIRONMENT"
OTLP_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"

log = get_logger("observability")


@dataclass(frozen=True)
class ObservabilityStatus:
    sentry: bool = False
    tracing: bool = False


def _init_sentry(service_name: str, dsn: str) -> bool:
    try:
        import sentry_sdk
    except ImportError:
        log.warning("sentry_unavailable", hint="install vidgen-core[observability]")
        return False
    sentry_sdk.init(
        dsn=dsn,
        environment=os.environ.get(SENTRY_ENV_ENV) or None,
        server_name=service_name,
    )
    sentry_sdk.set_tag("service", service_name)
    return True


def _init_tracing(service_name: str, endpoint: str) -> bool:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning("otel_unavailable", hint="install vidgen-core[observability]")
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # reads endpoint from env
    trace.set_tracer_provider(provider)
    return True


def init_observability(service_name: str) -> ObservabilityStatus:
    """Configure Sentry and tracing from the environment. No-op when nothing is configured."""
    dsn = os.environ.get(SENTRY_DSN_ENV, "").strip()
    endpoint = os.environ.get(OTLP_ENDPOINT_ENV, "").strip()
    return ObservabilityStatus(
        sentry=_init_sentry(service_name, dsn) if dsn else False,
        tracing=_init_tracing(service_name, endpoint) if endpoint else False,
    )
