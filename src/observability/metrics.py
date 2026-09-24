"""Prometheus instrumentation."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from fastapi import FastAPI
from prometheus_client import REGISTRY, Counter
from prometheus_client.metrics_core import GaugeMetricFamily
from prometheus_fastapi_instrumentator import Instrumentator

from src.cache.semantic_cache import SemanticCache, get_semantic_cache

logger = logging.getLogger(__name__)

# Pipeline decision counters
_PIPELINE_DECISIONS = Counter(
    "rag_pipeline_decisions_total",
    "Number of times the pipeline reached a specific termination decision",
    ["decision"]
)

def rag_counters(decision: str) -> None:
    """Increment a pipeline decision counter (e.g. answered, declined)."""
    _PIPELINE_DECISIONS.labels(decision=decision).inc()

# Cache counters are read from the live cache at scrape time rather than mirrored
# into separate Prometheus counters, so there is a single source of truth.
_CACHE_METRICS = ("hits", "misses", "errors", "writes", "hit_rate")


class CacheStatsCollector:
    """Publishes semantic-cache statistics as Prometheus gauges."""

    def __init__(self, cache: SemanticCache) -> None:
        self.cache = cache

    def collect(self) -> Iterator[GaugeMetricFamily]:
        stats = self.cache.stats.as_dict()
        for name in _CACHE_METRICS:
            gauge = GaugeMetricFamily(f"rag_cache_{name}", f"Semantic cache {name}")
            gauge.add_metric([], float(stats.get(name, 0)))
            yield gauge


def setup_metrics(app: FastAPI) -> Instrumentator:
    """Instrument HTTP metrics and publish semantic-cache statistics."""
    instrumentator = Instrumentator(excluded_handlers=["/metrics", "/health/live", "/health/ready"])
    instrumentator.instrument(app)

    cache = get_semantic_cache()
    if cache.stats.enabled:
        try:
            REGISTRY.register(CacheStatsCollector(cache))
        except ValueError as exc:
            logger.debug("Cache metrics collector already registered: %s", exc)

    return instrumentator
