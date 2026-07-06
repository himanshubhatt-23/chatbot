"""
models.py — AI Provider Abstraction Layer
==========================================
This module defines the abstract interface for all AI providers and
provides concrete implementations for:
    - OpenAI      (GPT-4o, GPT-4o-mini, etc.)
    - Anthropic   (Claude 3.5 Haiku, Sonnet, Opus)
    - Google      (Gemini 1.5 Flash, Pro)
    - Ollama      (Local LLMs: Llama 3.2, Mistral, Phi-3, etc.)

Architecture — Why Abstract Base Class (ABC)?
─────────────────────────────────────────────
The chatbot.py never imports OpenAI or Anthropic directly.
It only knows about AIProvider (the abstract class).
This means:
    1. You can swap providers by changing one line in config.py
    2. Adding a new provider (e.g. Cohere) only requires adding
       a new class here — zero changes to chatbot.py
    3. Unit tests can inject a MockProvider without real API calls
    4. The interface is a contract — every provider MUST implement
       `chat()` and `stream_chat()` or Python raises an error

This pattern is called the "Strategy Pattern" in software design.

Usage:
    from models import ProviderFactory
    provider = ProviderFactory.create()       # Creates from config
    response = provider.chat(messages)        # Works for any provider
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Generator, Optional

from config import config


# ─── Logger ──────────────────────────────────────────────────
# Use a module-level logger — best practice for library code.
# The root logger (in utils.py) configures the output format.
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# Using plain dicts for messages instead of dataclasses because:
# - All AI provider SDKs already expect dict format
# - No conversion overhead at call time
# - Simple and universally understood
#
# Message format (OpenAI standard, adopted by most providers):
#   {"role": "system",    "content": "You are FarmCity AI..."}
#   {"role": "user",      "content": "What is the best crop for clay soil?"}
#   {"role": "assistant", "content": "Clay soil works best with..."}
# ─────────────────────────────────────────────────────────────
Message = dict[str, str]      # {"role": str, "content": str}
MessageList = list[Message]   # Full conversation history


# ─────────────────────────────────────────────────────────────
# ABSTRACT BASE CLASS
# ─────────────────────────────────────────────────────────────

class AIProvider(ABC):
    """
    Abstract base class that all AI providers must inherit from.

    This defines the CONTRACT — any class that extends AIProvider
    MUST implement `chat()` and `stream_chat()` or Python will
    raise a TypeError when you try to instantiate it.

    Design Decision:
        Both `chat()` and `stream_chat()` are abstract.
        If a provider doesn't support streaming (e.g. a local model),
        it can raise NotImplementedError inside stream_chat() with a
        clear message — the chatbot will fall back to regular chat().

    Attributes:
        model    (str) : The model identifier string for this provider.
        max_tokens (int): Max tokens in the response.
        temperature (float): Creativity/randomness of the output.
        max_retries (int): How many times to retry on API failure.
        retry_delay (float): Seconds to wait between retries.
    """

    def __init__(self) -> None:
        """
        Initialise shared settings from the global config singleton.
        All providers share the same generation parameters so they
        produce comparable output regardless of which is active.
        """
        self.model:       str   = config.active_model
        self.max_tokens:  int   = config.max_tokens
        self.temperature: float = config.temperature
        self.max_retries: int   = config.max_retries
        self.retry_delay: float = config.retry_delay

    @abstractmethod
    def chat(self, messages: MessageList) -> str:
        """
        Send a list of messages and return the full response as a string.

        This is the PRIMARY method — every provider must implement it.

        Args:
            messages: Full conversation history including system prompt.
                      Format: [{"role": "system"|"user"|"assistant", "content": str}]

        Returns:
            str: The AI's response text. Never returns None — returns an
                 empty string or error message on failure.

        Raises:
            Should NOT raise — providers must handle errors internally
            and return a user-friendly error string instead.
        """
        ...

    @abstractmethod
    def stream_chat(self, messages: MessageList) -> Generator[str, None, None]:
        """
        Send messages and stream the response token by token.

        Args:
            messages: Same format as chat().

        Yields:
            str: Individual tokens or chunks as they arrive from the API.
                 The caller accumulates these into the full response.

        Example usage:
            for chunk in provider.stream_chat(messages):
                print(chunk, end="", flush=True)

        Raises:
            NotImplementedError: If the provider doesn't support streaming.
        """
        ...

    def _retry(self, func, *args, **kwargs):
        """
        Generic retry wrapper — applies to any callable.

        Retries up to self.max_retries times on any Exception,
        waiting self.retry_delay seconds between attempts.
        Uses exponential backoff: delay doubles on each retry.

        Why exponential backoff?
            On rate limit or server overload, hammering the API
            immediately makes things worse. Doubling the wait gives
            the server time to recover before each retry.

        Args:
            func    : The callable to retry (e.g. self._call_openai).
            *args   : Positional arguments forwarded to func.
            **kwargs: Keyword arguments forwarded to func.

        Returns:
            The return value of func on success.

        Raises:
            Exception: Re-raises the last exception after all retries fail.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as exc:
                last_exception = exc
                wait = self.retry_delay * (2 ** (attempt - 1))  # 2s, 4s, 8s

                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed: {exc}. "
                    f"Retrying in {wait:.1f}s..."
                )

                if attempt < self.max_retries:
                    time.sleep(wait)

        # All retries exhausted — re-raise so the provider can catch it
        raise last_exception

    @property
    def provider_name(self) -> str:
        """Human-readable name for logging. Overridden by each subclass."""
        return self.__class__.__name__


# ─────────────────────────────────────────────────────────────
# PROVIDER 1 — OPENAI
# ─────────────────────────────────────────────────────────────

class OpenAIProvider(AIProvider):
    """
    OpenAI GPT provider — supports GPT-4o, GPT-4o-mini, GPT-3.5-turbo, etc.

    Uses the official `openai` Python SDK (v1.x+).
    The SDK handles:
        - Connection pooling and timeouts
        - Automatic retries (we add our own layer on top)
        - Token usage reporting

    Design Decision:
        The client is created once in __init__ and reused for all calls.
        Creating a new client per request wastes connection setup time.

    Supported models (set OPENAI_MODEL in .env):
        gpt-4o          — Most capable, higher cost
        gpt-4o-mini     — Best value for money (default)
        gpt-3.5-turbo   — Fastest, cheapest, less capable
    """

    provider_name = "OpenAI"

    def __init__(self) -> None:
        super().__init__()

        # Import here (not at top of file) so users without the
        # openai package can still use other providers.
        # This is called "lazy importing" or "optional dependency" pattern.
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=config.openai_api_key)
            logger.info(f"✅ OpenAI provider ready — model: {self.model}")
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

    def chat(self, messages: MessageList) -> str:
        """
        Send messages to OpenAI and return the full response string.

        Args:
            messages: Conversation history with system prompt included.

        Returns:
            str: The assistant's response, or an error message string.
        """
        try:
            response = self._retry(
                self._client.chat.completions.create,
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content or ""
            logger.debug(
                f"OpenAI response received — "
                f"tokens used: {response.usage.total_tokens}"
            )
            return content.strip()

        except Exception as exc:
            logger.error(f"OpenAI chat failed after retries: {exc}")
            return (
                "I'm having trouble connecting to my AI system right now. "
                "Please try again in a moment. 🌾"
            )

    def stream_chat(self, messages: MessageList) -> Generator[str, None, None]:
        """
        Stream the response token by token from OpenAI.

        Yields:
            str: Text chunk (may be a word, partial word, or punctuation).
        """
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,  # Enable streaming
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except Exception as exc:
            logger.error(f"OpenAI stream failed: {exc}")
            yield "⚠️ Stream interrupted. Please try again."


# ─────────────────────────────────────────────────────────────
# PROVIDER 2 — ANTHROPIC CLAUDE
# ─────────────────────────────────────────────────────────────

class AnthropicProvider(AIProvider):
    """
    Anthropic Claude provider — supports Claude 3.5 Haiku, Sonnet, Opus.

    Key difference from OpenAI:
        Anthropic separates the system prompt from the messages list.
        The system prompt goes in a dedicated `system` parameter,
        not as {"role": "system"} inside messages.

        This class handles that conversion automatically — the chatbot
        always passes messages in the OpenAI format, and this provider
        extracts the system prompt before calling the API.

    Supported models (set ANTHROPIC_MODEL in .env):
        claude-3-5-haiku-20241022   — Fast & affordable (default)
        claude-3-5-sonnet-20241022  — Balanced capability
        claude-3-opus-20240229      — Most capable Claude model
    """

    provider_name = "Anthropic"

    def __init__(self) -> None:
        super().__init__()
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
            logger.info(f"✅ Anthropic provider ready — model: {self.model}")
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

    def _split_messages(
        self, messages: MessageList
    ) -> tuple[str, MessageList]:
        """
        Separate the system prompt from the conversation messages.

        Anthropic's API requires the system prompt in its own parameter.
        This method extracts it, leaving only user/assistant turns.

        Args:
            messages: Full message list including optional system message.

        Returns:
            tuple: (system_prompt_string, messages_without_system)
        """
        system_prompt = ""
        conversation: MessageList = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                conversation.append(msg)

        return system_prompt, conversation

    def chat(self, messages: MessageList) -> str:
        """
        Send messages to Anthropic Claude and return the response.

        Args:
            messages: Conversation history (system + user/assistant turns).

        Returns:
            str: Claude's response text, or an error message.
        """
        try:
            system_prompt, conversation = self._split_messages(messages)

            response = self._retry(
                self._client.messages.create,
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=conversation,
            )
            content = response.content[0].text if response.content else ""
            logger.debug(
                f"Anthropic response received — "
                f"input: {response.usage.input_tokens}, "
                f"output: {response.usage.output_tokens} tokens"
            )
            return content.strip()

        except Exception as exc:
            logger.error(f"Anthropic chat failed after retries: {exc}")
            return (
                "I'm having trouble reaching my AI system. "
                "Please try again shortly. 🌿"
            )

    def stream_chat(self, messages: MessageList) -> Generator[str, None, None]:
        """
        Stream the Claude response token by token.

        Yields:
            str: Text chunk from the streaming response.
        """
        try:
            system_prompt, conversation = self._split_messages(messages)

            with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=conversation,
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as exc:
            logger.error(f"Anthropic stream failed: {exc}")
            yield "⚠️ Stream interrupted. Please try again."


# ─────────────────────────────────────────────────────────────
# PROVIDER 3 — GOOGLE GEMINI
# ─────────────────────────────────────────────────────────────

class GeminiProvider(AIProvider):
    """
    Google Gemini provider — supports Gemini 1.5 Flash, Pro, Ultra.

    Key differences from OpenAI format:
        1. Gemini uses "parts" instead of "content" for message text.
        2. Roles are "user" and "model" (not "assistant").
        3. System prompt is passed as a system_instruction at model init.
        4. Conversation history uses a different structure.

    This class handles ALL those conversions transparently.

    Supported models (set GEMINI_MODEL in .env):
        gemini-1.5-flash   — Fast, great for most tasks (default)
        gemini-1.5-pro     — More capable, slower
        gemini-1.0-pro     — Older, stable
    """

    provider_name = "Gemini"

    def __init__(self) -> None:
        super().__init__()
        try:
            import google.generativeai as genai
            self._genai = genai
            genai.configure(api_key=config.gemini_api_key)
            logger.info(f"✅ Gemini provider ready — model: {self.model}")
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            )

    def _build_gemini_history(
        self, messages: MessageList
    ) -> tuple[str, list[dict]]:
        """
        Convert OpenAI-format messages to Gemini's conversation format.

        OpenAI format:
            {"role": "user",      "content": "Hello"}
            {"role": "assistant", "content": "Hi there!"}

        Gemini format:
            {"role": "user",  "parts": ["Hello"]}
            {"role": "model", "parts": ["Hi there!"]}

        Args:
            messages: OpenAI-format message list.

        Returns:
            tuple: (system_instruction_string, gemini_history_list)
        """
        system_instruction = ""
        history: list[dict] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "assistant":
                # Gemini calls the AI "model", not "assistant"
                history.append({"role": "model", "parts": [content]})
            else:
                history.append({"role": "user", "parts": [content]})

        return system_instruction, history

    def chat(self, messages: MessageList) -> str:
        """
        Send messages to Gemini and return the full response.

        Args:
            messages: Full conversation in OpenAI format.

        Returns:
            str: Gemini's response text.
        """
        try:
            system_instruction, history = self._build_gemini_history(messages)

            # The last message must be the user's turn — extract it
            if not history or history[-1]["role"] != "user":
                return "⚠️ No user message found to send."

            # The last message is the current user input
            current_message = history[-1]["parts"][0]
            past_history = history[:-1]  # Everything before the last user msg

            # Create model with system instruction baked in
            model = self._genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_instruction,
                generation_config={
                    "temperature":   self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )

            # Start a chat session with history, then send the new message
            chat_session = model.start_chat(history=past_history)
            response = self._retry(chat_session.send_message, current_message)

            logger.debug("Gemini response received.")
            return response.text.strip()

        except Exception as exc:
            logger.error(f"Gemini chat failed after retries: {exc}")
            return (
                "I'm having trouble connecting to Gemini right now. "
                "Please try again. 🌱"
            )

    def stream_chat(self, messages: MessageList) -> Generator[str, None, None]:
        """
        Stream Gemini's response token by token.

        Yields:
            str: Text chunk from the streaming response.
        """
        try:
            system_instruction, history = self._build_gemini_history(messages)

            if not history or history[-1]["role"] != "user":
                yield "⚠️ No user message found."
                return

            current_message = history[-1]["parts"][0]
            past_history = history[:-1]

            model = self._genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_instruction,
                generation_config={
                    "temperature":       self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )

            chat_session = model.start_chat(history=past_history)
            response = chat_session.send_message(current_message, stream=True)

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as exc:
            logger.error(f"Gemini stream failed: {exc}")
            yield "⚠️ Stream interrupted. Please try again."


# ─────────────────────────────────────────────────────────────
# PROVIDER 4 — OLLAMA (Local LLMs)
# ─────────────────────────────────────────────────────────────

class OllamaProvider(AIProvider):
    """
    Ollama provider — runs AI models 100% locally on your machine.

    Why Ollama?
        - Zero API cost — runs on your own hardware
        - Works offline — no internet required
        - Privacy — data never leaves your machine
        - Great for testing without burning API credits

    Requirements:
        1. Install Ollama: https://ollama.com/download
        2. Pull a model: `ollama pull llama3.2`
        3. Ollama auto-starts a server at localhost:11434
        4. Set OLLAMA_MODEL=llama3.2 in your .env

    Popular models to try (all free):
        llama3.2     — Meta's Llama 3.2 (3B or 11B) — excellent quality
        mistral      — Mistral 7B — fast and capable
        phi3         — Microsoft Phi-3 — great for small hardware
        gemma2       — Google Gemma 2 — strong reasoning

    API Format:
        Ollama exposes an OpenAI-compatible REST API at /api/chat.
        This means we can use the openai SDK with a custom base_url!
        No special Ollama SDK needed — massive simplification.
    """

    provider_name = "Ollama"

    def __init__(self) -> None:
        super().__init__()
        try:
            # Ollama speaks the OpenAI API format — reuse the OpenAI SDK
            # Just point it at the local Ollama server instead of api.openai.com
            from openai import OpenAI
            self._client = OpenAI(
                base_url=f"{config.ollama_base_url}/v1",
                api_key="ollama",  # Ollama doesn't check this — any string works
            )
            logger.info(
                f"✅ Ollama provider ready — model: {self.model} "
                f"at {config.ollama_base_url}"
            )
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai\n"
                "(We use the OpenAI SDK to talk to Ollama's compatible API)"
            )

    def chat(self, messages: MessageList) -> str:
        """
        Send messages to the local Ollama server and return the response.

        Args:
            messages: Conversation history (system + turns).

        Returns:
            str: The local model's response, or a helpful error if Ollama
                 isn't running or the model isn't pulled yet.
        """
        try:
            response = self._retry(
                self._client.chat.completions.create,
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content or ""
            logger.debug(f"Ollama response received from model: {self.model}")
            return content.strip()

        except Exception as exc:
            error_str = str(exc).lower()

            # Give specific, helpful messages for common Ollama problems
            if "connection refused" in error_str or "connect" in error_str:
                return (
                    "⚠️ Cannot connect to Ollama. Is it running?\n"
                    "Start it with: ollama serve"
                )
            elif "model" in error_str and "not found" in error_str:
                return (
                    f"⚠️ Model '{self.model}' not found in Ollama.\n"
                    f"Pull it with: ollama pull {self.model}"
                )
            else:
                logger.error(f"Ollama chat failed: {exc}")
                return (
                    "I'm having trouble with the local AI model. "
                    "Please check Ollama is running."
                )

    def stream_chat(self, messages: MessageList) -> Generator[str, None, None]:
        """
        Stream the Ollama response token by token.

        Ollama supports streaming natively through its OpenAI-compatible API.

        Yields:
            str: Text chunk from the local model.
        """
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except Exception as exc:
            logger.error(f"Ollama stream failed: {exc}")
            yield "⚠️ Local model stream interrupted."


# ─────────────────────────────────────────────────────────────
# PROVIDER FACTORY
# ─────────────────────────────────────────────────────────────

class ProviderFactory:
    """
    Factory class that creates the correct AI provider based on config.

    Design Decision — Why a Factory?
        Instead of writing this in chatbot.py:
            if config.provider == "openai":
                provider = OpenAIProvider()
            elif config.provider == "anthropic":
                ...

        We centralise all that logic here. chatbot.py just calls:
            provider = ProviderFactory.create()

        Benefits:
            1. Adding a new provider = add one entry to _PROVIDERS dict
            2. chatbot.py has zero if/elif provider logic
            3. The factory can be extended to support dynamic loading
               of provider plugins in the future

    Registry Pattern:
        _PROVIDERS maps string names → provider classes.
        To add a new provider, just add one line to _PROVIDERS.
    """

    # ── Provider Registry ─────────────────────────────────────
    # Maps config name → provider class.
    # To add a new provider: add ONE line here.
    _PROVIDERS: dict[str, type[AIProvider]] = {
        "openai":    OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini":    GeminiProvider,
        "ollama":    OllamaProvider,
    }

    @classmethod
    def create(cls, provider_name: Optional[str] = None) -> AIProvider:
        """
        Instantiate and return the appropriate AI provider.

        Args:
            provider_name: Override the config setting. Useful for testing.
                           If None, uses config.active_provider.

        Returns:
            AIProvider: A ready-to-use provider instance.

        Raises:
            ValueError: If the provider name is not in the registry.

        Example:
            # Use whatever is set in .env
            provider = ProviderFactory.create()

            # Force a specific provider (useful in tests)
            provider = ProviderFactory.create("ollama")
        """
        name = (provider_name or config.active_provider).lower().strip()

        if name not in cls._PROVIDERS:
            available = ", ".join(cls._PROVIDERS.keys())
            raise ValueError(
                f"Unknown AI provider: '{name}'. "
                f"Available providers: {available}. "
                f"Set FARMCITY_AI_PROVIDER in your .env file."
            )

        logger.info(f"Creating provider: {name.upper()}")
        return cls._PROVIDERS[name]()

    @classmethod
    def available_providers(cls) -> list[str]:
        """
        Return a list of all registered provider names.

        Returns:
            list[str]: e.g. ["openai", "anthropic", "gemini", "ollama"]
        """
        return list(cls._PROVIDERS.keys())

    @classmethod
    def register(cls, name: str, provider_class: type[AIProvider]) -> None:
        """
        Register a new provider at runtime.

        This enables the plugin/extensibility pattern — external code
        can add new providers without modifying this file.

        Args:
            name           : The string key (e.g. "cohere", "mistral_api").
            provider_class : A class that extends AIProvider.

        Example:
            class CohereProvider(AIProvider):
                ...

            ProviderFactory.register("cohere", CohereProvider)
            provider = ProviderFactory.create("cohere")
        """
        if not issubclass(provider_class, AIProvider):
            raise TypeError(
                f"{provider_class.__name__} must extend AIProvider"
            )
        cls._PROVIDERS[name.lower()] = provider_class
        logger.info(f"Registered new provider: {name}")