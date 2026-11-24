"""LangSmith tracing integration.

LangSmith is optional. With ``LANGSMITH_API_KEY`` and ``LANGSMITH_TRACING=true`` set,
LangChain runs are traced to LangSmith and the ``@traceable`` decorator on the pipeline
creates one top-level run per query. Without a key the integration is a no-op, so there
is no observability regression when LangSmith is not configured.

Configuration is applied by setting the ``LANGSMITH_*`` environment variables that the
``langsmith`` / ``langchain-core`` SDKs read at call time, which is why this is safe to
run during startup rather than before imports.
"""

from __future__ import annotations

import logging
import os

from src.config.settings import Settings

logger = logging.getLogger(__name__)


def configure_langsmith(settings: Settings) -> bool:
    """Enable LangSmith tracing from the application settings.

    Returns ``True`` when tracing was enabled, ``False`` otherwise.
    """
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        logger.info("LangSmith tracing is disabled (set LANGSMITH_API_KEY and LANGSMITH_TRACING=true).")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    logger.info("LangSmith tracing enabled (project=%s).", settings.langsmith_project)
    return True
