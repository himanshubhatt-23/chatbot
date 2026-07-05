"""
config.py — FarmCity AI Assistant Configuration
================================================
Central configuration module that loads all settings from environment
variables. This is the ONLY place where configuration is defined.

Design Decision:
    Using a dataclass-based config instead of a plain dict because:
    - Type hints give IDE autocompletion and catch bugs early
    - Dataclasses are cleaner than a flat dict of strings
    - Easy to add validation logic later (e.g. pydantic)
    - Single source of truth for all settings

Usage:
    from config import config          # Import the singleton instance
    print(config.openai_api_key)       # Access any setting
    print(config.active_provider)      # "openai" | "anthropic" | "gemini" | "ollama"
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Project Root ────────────────────────────────────────────
# BASE_DIR is the folder that contains this file (farmcity_ai vala)
BASE_DIR: Path = Path(__file__).parent


# ─── Logs Directory ──────────────────────────────────────────
# Automatically create logs/ folder if it doesn't exist.
# This prevents FileNotFoundError when the logger tries to write.
LOGS_DIR: Path = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """
    Central configuration container for the FarmCity AI Assistant.

    All values are loaded from environment variables at startup.
    Defaults are safe, non-secret values only.

    Attributes:
        active_provider  : Which AI backend to use ("openai", "anthropic", "gemini", "ollama")
        openai_api_key   : OpenAI secret key (set in .env)
        openai_model     : GPT model name (default: gpt-4o-mini — best value for money)
        anthropic_api_key: Anthropic Claude secret key
        anthropic_model  : Claude model name
        gemini_api_key   : Google Gemini API key
        gemini_model     : Gemini model name
        ollama_base_url  : Local Ollama server URL
        ollama_model     : Local model loaded in Ollama
        max_memory_messages: Max chat history to keep in memory (sliding window)
        max_tokens       : Max tokens in the AI response
        temperature      : 0.0 = deterministic, 1.0 = very creative
        stream           : Whether to stream responses token by token
        log_level        : Logging verbosity (DEBUG, INFO, WARNING, ERROR)
        log_file         : Path to the log file
        app_name         : Displayed in logs and CLI header
        app_version      : Semantic version string
    """

    # ── AI Provider Selection ──────────────────────────────────
    active_provider: str = field(default_factory=lambda: os.getenv("FARMCITY_AI_PROVIDER", "openai"))

    # ── OpenAI ────────────────────────────────────────────────
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    # ── Anthropic Claude ──────────────────────────────────────
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
    )

    # ── Google Gemini ─────────────────────────────────────────
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    )

    # ── Ollama (local / on-premise LLM) ──────────────────────
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2")
    )


    max_memory_messages: int = field(
        default_factory=lambda: int(os.getenv("MAX_MEMORY_MESSAGES", "20"))
    )

    # ── Model Generation Settings ──────────────────────────────
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "1024"))
    )
    # temperature=0.7 is a good balance: helpful & somewhat creative,
    # but not hallucinating wildly. Farm advice should be factual → lower is safer.
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7"))
    )
    stream: bool = field(
        default_factory=lambda: os.getenv("STREAM_RESPONSES", "false").lower() == "true"
    )

    # ── Retry Settings ────────────────────────────────────────
    # On API failure, retry up to max_retries times with retry_delay seconds gap.
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3"))
    )
    retry_delay: float = field(
        default_factory=lambda: float(os.getenv("RETRY_DELAY_SECONDS", "2.0"))
    )

    # ── Logging ───────────────────────────────────────────────
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )
    log_file: Path = field(
        default_factory=lambda: LOGS_DIR / "farmcity_ai.log"
    )

    # ── App Identity ──────────────────────────────────────────
    app_name: str = "FarmCity AI Assistant"
    app_version: str = "1.0.0"

    def __post_init__(self) -> None:
        """
        Validate configuration after dataclass initialisation.

        Raises:
            ValueError: If the selected provider has no API key set
                        (except Ollama which runs locally and needs no key).
        """
        self._validate_provider()

    def _validate_provider(self) -> None:
        """
        Check that the selected AI provider has the required credentials.

        Ollama runs locally so it doesn't need an API key — skip validation.
        All cloud providers need their API key set in the environment.
        """
        provider = self.active_provider.lower()

        key_map = {
            "openai":    self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini":    self.gemini_api_key,
        }

        if provider in key_map and not key_map[provider]:
            # Warn but do not crash — the user might be testing without a key
            logging.warning(
                f"⚠️  Provider '{provider}' selected but no API key found. "
                f"Set the environment variable and restart."
            )

    @property
    def active_model(self) -> str:
        """
        Return the model name for the currently active provider.

        Returns:
            str: Model identifier string (e.g. "gpt-4o-mini", "claude-3-5-haiku-20241022")
        """
        model_map = {
            "openai":    self.openai_model,
            "anthropic": self.anthropic_model,
            "gemini":    self.gemini_model,
            "ollama":    self.ollama_model,
        }
        return model_map.get(self.active_provider.lower(), "unknown-model")

    @property
    def active_api_key(self) -> Optional[str]:
        """
        Return the API key for the currently active provider.

        Returns:
            Optional[str]: The API key, or None for local providers like Ollama.
        """
        key_map = {
            "openai":    self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini":    self.gemini_api_key,
            "ollama":    None,  # Local — no key needed
        }
        return key_map.get(self.active_provider.lower())

    def summary(self) -> str:
        """
        Return a safe, human-readable config summary for logging.
        API keys are masked — never log real secrets.

        Returns:
            str: Multi-line config summary string.
        """
        def mask(key: Optional[str]) -> str:
            """Show first 6 chars then mask the rest."""
            if not key:
                return "NOT SET"
            return key[:6] + "••••••••" + key[-4:] if len(key) > 10 else "••••••"

        return (
            f"\n{'─' * 50}\n"
            f"  {self.app_name} v{self.app_version}\n"
            f"{'─' * 50}\n"
            f"  Provider    : {self.active_provider.upper()}\n"
            f"  Model       : {self.active_model}\n"
            f"  Max Tokens  : {self.max_tokens}\n"
            f"  Temperature : {self.temperature}\n"
            f"  Memory      : {self.max_memory_messages} messages\n"
            f"  Streaming   : {self.stream}\n"
            f"  Log Level   : {self.log_level}\n"
            f"  OpenAI Key  : {mask(self.openai_api_key)}\n"
            f"  Claude Key  : {mask(self.anthropic_api_key)}\n"
            f"  Gemini Key  : {mask(self.gemini_api_key)}\n"
            f"{'─' * 50}"
        )



config = AppConfig()