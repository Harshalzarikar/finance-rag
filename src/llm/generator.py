"""Generation model factory."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return the process-wide chat model, constructing it on first use."""
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set — the generator cannot be initialised.")
    logger.info(
        "Initializing Groq chat model (%s, timeout=%ds).", settings.groq_model_name, settings.groq_request_timeout
    )
    return ChatGroq(
        model_name=settings.groq_model_name,
        temperature=0.0,
        groq_api_key=settings.groq_api_key,
        request_timeout=settings.groq_request_timeout,
    )
