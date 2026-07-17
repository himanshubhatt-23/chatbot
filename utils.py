"""
utils.py -- FarmCity AI Utility System
=======================================
Provides all shared utilities used across the project:

    1. Logging Setup       -- Structured file + console logging
    2. Input Sanitiser     -- Clean and validate user input
    3. Response Formatter  -- Format AI responses for CLI display
    4. Streaming Printer   -- Print streamed tokens in real time
    5. Text Helpers        -- Word wrap, truncation, separators
    6. Tool Registry       -- Extensibility hooks for future tools
    7. Tool Stubs          -- Weather, Market, Database, RAG

Architecture -- Why a utils.py?
Every real project has shared code that does not belong to one specific
module. Instead of copy-pasting helpers or creating circular imports,
utils.py is the shared toolbox that anyone can import safely.

Rules for what belongs here:
    YES: Pure functions with no side effects
    YES: Logging setup (needed everywhere)
    YES: Tool stubs (future integrations)
    NO:  Business logic (goes in chatbot.py or services)
    NO:  AI provider code (goes in models.py)
    NO:  Memory management (goes in memory.py)

Usage:
    from utils import setup_logging, sanitise_input, print_response
    from utils import ToolRegistry, default_tool_registry

    logger = setup_logging()
    clean  = sanitise_input("  What crops grow in clay soil?  ")
    print_response("Here is the answer...", stream=False)
"""

import logging
import logging.handlers
import os
import re
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Generator, Optional

from config import config


# =====================================================
# 1. LOGGING SETUP
# =====================================================

def setup_logging(
    log_level:  Optional[str]  = None,
    log_file:   Optional[Path] = None,
    log_format: Optional[str]  = None,
) -> logging.Logger:
    """
    Configure and return the root logger for the entire application.

    Sets up TWO handlers simultaneously:
        1. Console handler -- coloured output to stdout for developers
        2. File handler    -- rotating log file for production audit trail

    Why RotatingFileHandler?
        Plain FileHandler grows forever. RotatingFileHandler caps the
        file at max_bytes and keeps backup_count old files.
        Total disk usage stays bounded at 30 MB (5 files x 5 MB each).

    Why configure the root logger once in main.py?
        Python logging is hierarchical. Configuring root ONCE propagates
        to ALL child loggers in every module automatically.

    Args:
        log_level  : Override config log level. e.g. "DEBUG", "INFO".
        log_file   : Override config log file path.
        log_format : Custom format string. Uses a sensible default.

    Returns:
        logging.Logger: Configured root logger instance.
    """
    level      = getattr(logging, (log_level or config.log_level).upper(), logging.INFO)
    file_path  = log_file or config.log_file
    fmt        = log_format or "[%(asctime)s] %(levelname)-8s %(name)-20s : %(message)s"
    date_fmt   = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove pre-existing handlers to prevent duplicate log lines
    root_logger.handlers.clear()

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    # Handler 1: Console output (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Handler 2: Rotating file (5 MB per file, 5 backups = 30 MB max)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except OSError as exc:
        root_logger.warning(f"Could not create log file at {file_path}: {exc}")

    return root_logger


# Module-level logger for this file
logger = logging.getLogger(__name__)


# =====================================================
# 2. INPUT SANITISER
# =====================================================

MAX_INPUT_LENGTH: int = 2000
MIN_INPUT_LENGTH: int = 2


def sanitise_input(user_input: str) -> str:
    """
    Clean and validate raw user input before sending it to the AI.

    Transformations applied in order:
        1. Strip leading/trailing whitespace
        2. Collapse multiple consecutive spaces into one
        3. Remove control characters (null bytes, escape sequences)
        4. Truncate to MAX_INPUT_LENGTH characters
        5. Validate minimum length

    Design Decision -- sanitise instead of reject:
        Sanitising is more user-friendly than hard errors.
        We only raise ValueError if the input is truly unusable.

    Args:
        user_input: Raw string from input() or API request body.

    Returns:
        str: Cleaned input string ready for the AI.

    Raises:
        ValueError: If the input is empty or too short after cleaning.

    Examples:
        sanitise_input("  What is DAP?  ")  ->  "What is DAP?"
        sanitise_input("Hello World")        ->  "Hello World"
        sanitise_input("")                   ->  raises ValueError
    """
    if not isinstance(user_input, str):
        raise ValueError(f"Input must be a string, got {type(user_input).__name__}")

    # Step 1: Strip surrounding whitespace
    cleaned = user_input.strip()

    # Step 2: Collapse multiple whitespace characters into single space
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Step 3: Remove control characters (keep normal printable chars)
    cleaned = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", cleaned)

    # Step 4: Truncate to maximum length
    if len(cleaned) > MAX_INPUT_LENGTH:
        cleaned = cleaned[:MAX_INPUT_LENGTH]
        logger.warning(f"Input truncated to {MAX_INPUT_LENGTH} chars.")

    # Step 5: Validate minimum length
    if not cleaned:
        raise ValueError("Input cannot be empty. Please type a question.")

    if len(cleaned) < MIN_INPUT_LENGTH:
        raise ValueError(
            f"Input too short. Please type at least {MIN_INPUT_LENGTH} characters."
        )

    return cleaned


# =====================================================
# 3. TEXT HELPERS
# =====================================================

try:
    TERMINAL_WIDTH: int = os.get_terminal_size().columns
except OSError:
    TERMINAL_WIDTH = 80

TERMINAL_WIDTH = min(TERMINAL_WIDTH, 100)


def separator(char: str = "-", width: Optional[int] = None) -> str:
    """
    Return a horizontal separator line for CLI display.

    Args:
        char : The character to repeat. Default: dash.
        width: Line width. Defaults to TERMINAL_WIDTH.

    Returns:
        str: A line of repeated characters.
    """
    return char * (width or TERMINAL_WIDTH)


def wrap_text(text: str, width: Optional[int] = None, indent: str = "") -> str:
    """
    Word-wrap a long string to fit within the terminal width.

    Args:
        text  : The text to wrap.
        width : Max line width. Defaults to TERMINAL_WIDTH.
        indent: Optional prefix for each wrapped line.

    Returns:
        str: Wrapped text with newlines inserted.
    """
    return textwrap.fill(
        text,
        width=width or TERMINAL_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=True,
    )


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to max_length characters, appending a suffix.

    Args:
        text      : The text to truncate.
        max_length: Maximum character count including the suffix.
        suffix    : Appended when truncation occurs.

    Returns:
        str: Original text if short enough, else truncated with suffix.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Return a human-readable timestamp string.

    Args:
        dt: datetime object. Defaults to now if not provided.

    Returns:
        str: e.g. "15 Nov 2024, 09:30 AM"
    """
    target = dt or datetime.now()
    return target.strftime("%d %b %Y, %I:%M %p")


# =====================================================
# 4. RESPONSE FORMATTER AND PRINTER
# =====================================================

def print_response(
    text:       str,
    label:      str  = "FarmCity AI",
    stream:     bool = False,
    add_border: bool = True,
) -> None:
    """
    Print a complete AI response to the terminal with consistent formatting.

    Used when streaming is OFF. The full response is printed at once.

    Format:
        --------------------------------------------------
        FarmCity AI
        --------------------------------------------------
        Your answer here, word-wrapped to fit
        the terminal width neatly.
        --------------------------------------------------

    Args:
        text      : The AI response text to display.
        label     : The speaker label. Default: "FarmCity AI"
        stream    : If True, skip the border (streaming handles its own display).
        add_border: Whether to draw separator lines around the response.
    """
    if stream:
        print(text)
        return

    sep = separator()

    if add_border:
        print(f"\n{sep}")
        print(f"  FarmCity AI")
        print(sep)

    for paragraph in text.split("\n"):
        if paragraph.strip():
            print(wrap_text(paragraph))
        else:
            print()

    if add_border:
        print(sep)


def print_stream_start(label: str = "FarmCity AI") -> None:
    """
    Print the header before streaming begins.

    Called once before the first token arrives.

    Args:
        label: The speaker label displayed in the header.
    """
    sep = separator()
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)


def print_stream_token(token: str) -> None:
    """
    Print a single streaming token to the terminal without a newline.

    flush=True is critical -- without it Python buffers the output
    and nothing appears until the buffer is full.

    Args:
        token: A single text chunk from the AI stream generator.
    """
    print(token, end="", flush=True)


def print_stream_end() -> None:
    """
    Print the footer after all streaming tokens have been received.
    Called once after the generator is exhausted.
    """
    sep = separator()
    print(f"\n{sep}\n")


def stream_and_collect(
    generator: Generator[str, None, None],
    label:     str = "FarmCity AI",
) -> str:
    """
    Stream tokens to the terminal AND collect them into a full string.

    This solves the key streaming problem: the chatbot needs to:
        1. Show the user each token as it arrives (good UX)
        2. Store the complete response in memory for conversation history

    This function does both in one pass over the generator.

    Args:
        generator: A token generator from provider.stream_chat().
        label    : Speaker label for the output header.

    Returns:
        str: The complete response text (all tokens concatenated).

    Example:
        generator = provider.stream_chat(messages)
        full_response = stream_and_collect(generator)
        memory.add_assistant_message(full_response)
    """
    print_stream_start(label)

    collected_tokens: list[str] = []

    for token in generator:
        print_stream_token(token)
        collected_tokens.append(token)

    print_stream_end()

    return "".join(collected_tokens)


# =====================================================
# 5. CLI DISPLAY HELPERS
# =====================================================

def print_welcome(app_name: str, version: str, provider: str, model: str) -> None:
    """
    Print the FarmCity AI welcome banner when the CLI starts.

    Args:
        app_name : The application name string.
        version  : Semantic version (e.g. "1.0.0").
        provider : Active AI provider name (e.g. "OpenAI").
        model    : Active model name (e.g. "gpt-4o-mini").
    """
    sep    = separator("=")
    sep_sm = separator("-")

    print(f"\n{sep}")
    print(f"  FarmCity AI  v{version}")
    print(sep_sm)
    print(f"  Provider : {provider.upper()}")
    print(f"  Model    : {model}")
    print(f"  Time     : {format_timestamp()}")
    print(sep_sm)
    print("  Type your farming question and press Enter.")
    print("  Commands: /help  /clear  /memory  /variant  /quit")
    print(f"{sep}\n")


def print_help() -> None:
    """Print the CLI help text listing all available commands."""
    sep = separator("-")
    print(f"\n{sep}")
    print("  FarmCity AI -- Available Commands")
    print(sep)
    print("  /help              Show this help message")
    print("  /clear             Clear conversation history")
    print("  /reset             Full reset including system prompt")
    print("  /memory            Show current memory usage")
    print("  /variant <name>    Switch prompt style:")
    print("                       default  -- Full FarmCity expert")
    print("                       concise  -- Short answers only")
    print("                       expert   -- Technical depth")
    print("                       hindi    -- Respond in Hindi")
    print("                       seller   -- FarmCity seller onboarding")
    print("  /history           Print the last 5 messages from memory")
    print("  /quit or /exit     Exit the chatbot")
    print(sep)
    print("  For farming questions, just type naturally!")
    print(f"{sep}\n")


def print_memory_stats(
    message_count: int,
    turn_count:    int,
    max_messages:  int,
    session_id:    str,
) -> None:
    """
    Print a memory usage summary to the terminal.

    Args:
        message_count: Total messages in memory including system prompt.
        turn_count   : Number of complete user-assistant turns.
        max_messages : The configured maximum message limit.
        session_id   : Current session identifier.
    """
    sep = separator("-")
    used_pct = int((turn_count * 2 / max(max_messages, 1)) * 100)
    bar = "#" * (used_pct // 10) + "." * (10 - used_pct // 10)

    print(f"\n{sep}")
    print("  Memory Status")
    print(sep)
    print(f"  Session ID    : {session_id}")
    print(f"  Total Messages: {message_count}")
    print(f"  Turns         : {turn_count} / {max_messages // 2}")
    print(f"  Memory Used   : {used_pct}%  [{bar}]")
    print(f"{sep}\n")


def print_error(message: str) -> None:
    """Print an error message in a clearly visible format."""
    print(f"\n[ERROR] {message}\n")


def print_info(message: str) -> None:
    """Print an informational message to the terminal."""
    print(f"\n[INFO] {message}\n")


def print_success(message: str) -> None:
    """Print a success confirmation message."""
    print(f"\n[OK] {message}\n")


# =====================================================
# 6. PERFORMANCE TIMER
# =====================================================

class Timer:
    """
    Simple context manager for timing code blocks.

    Design Decision -- Context Manager:
        Using a context manager is cleaner than manual start/stop calls.
        The elapsed time is always recorded even if an exception occurs.

    Attributes:
        elapsed (float): Seconds elapsed between __enter__ and __exit__.

    Usage:
        with Timer() as t:
            response = provider.chat(messages)
        print(f"API call took {t.elapsed:.2f}s")
    """

    def __init__(self) -> None:
        self._start:  float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"{self.elapsed:.2f}s"


# =====================================================
# 7. TOOL REGISTRY -- Extensibility System
# =====================================================

class ToolRegistry:
    """
    Registry for external tool functions that the chatbot can call.

    Design Pattern -- Registry / Plugin System:
        Tools are plain Python functions registered with a string name.
        The chatbot calls any tool by name without knowing its
        implementation details.

    A Tool is any function that:
        - Accepts a string query
        - Returns a string result formatted for prompt injection
        - Is self-contained with no side effects on chatbot state

    Pre-registered stubs:
        weather       -- Fetch weather for a location
        market_price  -- Fetch MSP or mandi price for a crop
        db_search     -- Query the FarmCity PostgreSQL database
        rag_search    -- Semantic search in vector database

    Future tools to add:
        soil_report   -- Fetch soil health card data by district
        scheme_lookup -- Look up government scheme eligibility
        image_analyse -- Analyse crop disease from uploaded photo
        voice_input   -- Transcribe audio to text

    Usage:
        registry = ToolRegistry()
        registry.register("weather", get_weather)
        result = registry.call("weather", "Ludhiana, Punjab")
    """

    def __init__(self) -> None:
        """Initialise an empty tool registry."""
        self._tools: dict[str, Callable[[str], str]] = {}
        logger.debug("ToolRegistry initialised.")

    def register(self, name: str, func: Callable[[str], str]) -> None:
        """
        Register a tool function under a given name.

        Args:
            name: Tool identifier (e.g. "weather", "market_price").
            func: Callable that accepts a query string and returns a
                  result string. Signature: (query: str) -> str

        Raises:
            ValueError: If name is empty or func is not callable.
        """
        if not name or not name.strip():
            raise ValueError("Tool name cannot be empty.")
        if not callable(func):
            raise ValueError(f"Tool '{name}' must be a callable function.")

        self._tools[name.lower().strip()] = func
        logger.info(f"Tool registered: '{name}'")

    def call(self, name: str, query: str) -> Optional[str]:
        """
        Call a registered tool by name and return its result.

        Returns None if the tool is not registered or raises an exception.
        Returning None lets the caller skip the tool result gracefully
        rather than crashing the entire chatbot response.

        Args:
            name : Tool name (must have been registered first).
            query: Input string passed to the tool function.

        Returns:
            str  : The tool result string ready for prompt injection.
            None : If the tool is not registered or fails.
        """
        tool_name = name.lower().strip()

        if tool_name not in self._tools:
            logger.warning(f"Tool '{name}' not found in registry.")
            return None

        try:
            with Timer() as t:
                result = self._tools[tool_name](query)
            logger.info(f"Tool '{name}' executed in {t}")
            return result

        except Exception as exc:
            logger.error(f"Tool '{name}' failed with error: {exc}")
            return None

    def list_tools(self) -> list[str]:
        """Return a list of all registered tool names."""
        return list(self._tools.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a tool is registered by name."""
        return name.lower().strip() in self._tools

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry. Silent if not found."""
        removed = self._tools.pop(name.lower().strip(), None)
        if removed:
            logger.info(f"Tool unregistered: '{name}'")

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_tools()})"


# =====================================================
# 8. TOOL IMPLEMENTATIONS (Stubs -- Ready to Fill In)
# =====================================================
# Each function is a STUB returning example data.
#
# HOW TO REPLACE A STUB WITH A REAL IMPLEMENTATION:
#   1. Install the required library (shown in each docstring)
#   2. Replace the stub return statement with a real API call
#   3. The chatbot.py code that calls these tools does NOT change
# =====================================================

def get_weather(location: str) -> str:
    """
    Fetch current weather for a farming location.

    STUB -- Replace with real implementation.

    HOW TO IMPLEMENT (OpenWeatherMap, free tier):
        pip install requests
        API docs: https://openweathermap.org/api

        import requests
        api_key = os.getenv("OPENWEATHER_API_KEY")
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": location, "appid": api_key, "units": "metric"}
        r = requests.get(url, params=params, timeout=5)
        d = r.json()
        return (
            f"Location: {location} | "
            f"Temp: {d['main']['temp']}C | "
            f"{d['weather'][0]['description']} | "
            f"Humidity: {d['main']['humidity']}%"
        )

    Args:
        location: City or district name (e.g. "Ludhiana, Punjab").

    Returns:
        str: Formatted weather string for prompt injection.
    """
    logger.info(f"[STUB] Weather tool called for: {location}")
    return (
        f"Location: {location} | Temp: 28C | Humidity: 65% | "
        f"Condition: Partly cloudy | "
        f"Forecast: Light rain expected tomorrow | "
        f"[STUB -- connect OpenWeatherMap or IMD API]"
    )


def get_market_price(crop: str) -> str:
    """
    Fetch current MSP and mandi prices for a crop.

    STUB -- Replace with real implementation.

    HOW TO IMPLEMENT (data.gov.in Open Government Data):
        pip install requests
        API docs: https://api.data.gov.in/

        import requests
        api_key = os.getenv("DATA_GOV_API_KEY")
        url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        params = {
            "api-key": api_key,
            "format": "json",
            "filters[commodity]": crop,
            "limit": 5,
        }
        r = requests.get(url, params=params, timeout=5)
        records = r.json().get("records", [])
        lines = [f"{r['market']}: Rs {r['modal_price']}/quintal" for r in records]
        return " | ".join(lines) if lines else f"No price data found for {crop}"

    Args:
        crop: Crop name (e.g. "Wheat", "Tomato", "Onion").

    Returns:
        str: Formatted price data string for prompt injection.
    """
    logger.info(f"[STUB] Market price tool called for: {crop}")
    return (
        f"Crop: {crop} | MSP: Rs 2,275/quintal (2024-25) | "
        f"Mandi Range: Rs 2,100 - Rs 2,400/quintal | "
        f"[STUB -- connect Agmarknet or eNAM API]"
    )


def search_database(query: str) -> str:
    """
    Search the FarmCity PostgreSQL database for user-specific data.

    STUB -- Replace with real implementation.

    HOW TO IMPLEMENT (psycopg2):
        pip install psycopg2-binary

        import psycopg2
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, quantity, price FROM products "
            "WHERE name ILIKE %s LIMIT 5",
            (f"%{query}%",)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return " | ".join([f"{r[0]}: {r[1]} units @ Rs{r[2]}" for r in rows])

    FUTURE RAG (ChromaDB):
        pip install chromadb
        collection = chromadb.Client().get_collection("farmcity_knowledge")
        results = collection.query(query_texts=[query], n_results=3)
        return " ".join(results["documents"][0])

    Args:
        query: Search query extracted from the user's message.

    Returns:
        str: Formatted database results for prompt injection.
    """
    logger.info(f"[STUB] Database search for: '{query}'")
    return (
        f"DB search: '{query}' | No live database connected. | "
        f"[STUB -- connect PostgreSQL with DATABASE_URL env variable]"
    )


def search_knowledge_base(query: str) -> str:
    """
    Perform semantic search in the agricultural knowledge base.

    STUB -- Replace with real implementation.

    HOW TO IMPLEMENT (ChromaDB -- easiest local vector DB):
        pip install chromadb sentence-transformers

        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("farm_knowledge")
        results = collection.query(query_texts=[query], n_results=3)
        docs = results["documents"][0]
        return " | ".join(docs)

    HOW TO IMPLEMENT (Pinecone -- cloud hosted):
        pip install pinecone-client

    HOW TO IMPLEMENT (FAISS -- for large scale):
        pip install faiss-cpu sentence-transformers

    Args:
        query: User question or search terms.

    Returns:
        str: Relevant knowledge base excerpts for prompt injection.
    """
    logger.info(f"[STUB] Knowledge base search for: '{query}'")
    return (
        f"KB search: '{query}' | No vector database connected. | "
        f"[STUB -- connect ChromaDB, FAISS, or Pinecone]"
    )


# =====================================================
# 9. DEFAULT TOOL REGISTRY INSTANCE
# =====================================================
# Pre-loaded with all stub tools. chatbot.py imports this instance.
#
# To replace a stub with a real implementation:
#   from utils import default_tool_registry
#   default_tool_registry.register("weather", my_real_weather_fn)
# This replaces the stub without changing any other code.

default_tool_registry = ToolRegistry()
default_tool_registry.register("weather",      get_weather)
default_tool_registry.register("market_price", get_market_price)
default_tool_registry.register("db_search",    search_database)
default_tool_registry.register("rag_search",   search_knowledge_base)