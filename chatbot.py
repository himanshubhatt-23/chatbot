"""
chatbot.py -- FarmCity AI Chatbot Engine
==========================================
The central orchestrator that wires all modules together:

    config.py   --> Settings and environment variables
    models.py   --> AI provider (OpenAI / Anthropic / Gemini / Ollama)
    memory.py   --> Conversation history and session management
    prompts.py  --> System prompt assembly with dynamic context
    utils.py    --> Logging, input sanitisation, tool registry

Architecture -- Facade Pattern:
    FarmCityChatbot is a Facade. It presents one clean interface
    to the outside world (main.py, Flask, FastAPI, tests) while
    hiding the complexity of coordinating five different modules.

    External code only needs to know:
        bot = FarmCityChatbot()
        response = bot.chat("What fertilizer is best for wheat?")

    It does NOT need to know about providers, memory trimming,
    prompt building, token streaming, or tool execution.

Design -- Single Responsibility per Method:
    chat()          --> The one public method for getting a response
    _build_prompt() --> Assembles system prompt (calls prompts.py)
    _run_tools()    --> Calls external tools (calls utils.py registry)
    _send()         --> Calls the AI provider (calls models.py)
    _handle_cmd()   --> Processes slash commands (/clear, /help, etc.)

Thread Safety:
    Each FarmCityChatbot instance has its OWN memory object.
    For web APIs with multiple users, create one instance per session:
        bot = FarmCityChatbot(session_id=user_id)
    Or use the ChatbotFactory for managed multi-session handling.

Usage (CLI):
    bot = FarmCityChatbot()
    response = bot.chat("What crops grow in clay soil?")
    print(response)

Usage (Flask/FastAPI):
    bot = FarmCityChatbot(session_id=request.user.id)
    response = bot.chat(request.json["message"])
    return {"response": response}

Usage (Streaming):
    bot = FarmCityChatbot(stream=True)
    for token in bot.stream("What is DAP fertilizer?"):
        print(token, end="", flush=True)
"""

import logging
from dataclasses import dataclass, field
from typing import Generator, Iterator, Optional

from config import config
from memory import ConversationMemory, InMemoryStore, MemoryStore
from models import AIProvider, ProviderFactory
from prompts import PromptBuilder
from utils import (
    Timer,
    default_tool_registry,
    sanitise_input,
    ToolRegistry,
)

# Module-level logger
logger = logging.getLogger(__name__)


# =====================================================
# CHATBOT RESPONSE DATACLASS
# =====================================================

@dataclass
class ChatResponse:
    """
    Structured container for a single chatbot response.

    Design Decision -- Dataclass instead of plain string:
        Returning a plain string is simpler, but returning a dataclass
        means callers get metadata for free (elapsed time, session ID,
        provider used). A Flask API can serialise this to JSON directly.
        A CLI can show the timing. Tests can assert on specific fields.

    Attributes:
        text          : The AI's response text (main payload).
        session_id    : Which session produced this response.
        provider      : Which AI provider was used (e.g. "OpenAI").
        model         : Which model was used (e.g. "gpt-4o-mini").
        elapsed_sec   : How long the AI call took in seconds.
        is_command    : True if the input was a slash command (/help etc.).
        is_error      : True if the response is an error message.
        tools_used    : List of tool names called to build this response.

    Usage:
        resp = bot.chat("What is the MSP for wheat?")
        print(resp.text)             # The answer
        print(resp.elapsed_sec)      # e.g. 1.24
        print(resp.provider)         # e.g. "OpenAI"

        # JSON serialisation for Flask/FastAPI:
        return resp.to_dict()
    """

    text:        str
    session_id:  str        = "default"
    provider:    str        = ""
    model:       str        = ""
    elapsed_sec: float      = 0.0
    is_command:  bool       = False
    is_error:    bool       = False
    tools_used:  list[str]  = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Serialise this response to a plain dictionary.

        Use this when returning responses from a Flask or FastAPI endpoint.
        All fields are JSON-serialisable (str, float, bool, list).

        Returns:
            dict: JSON-ready dictionary with all response fields.
        """
        return {
            "text":        self.text,
            "session_id":  self.session_id,
            "provider":    self.provider,
            "model":       self.model,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "is_command":  self.is_command,
            "is_error":    self.is_error,
            "tools_used":  self.tools_used,
        }

    def __str__(self) -> str:
        """Return just the text for simple print() usage."""
        return self.text


# =====================================================
# MAIN CHATBOT CLASS
# =====================================================

class FarmCityChatbot:
    """
    FarmCity AI Chatbot -- the central engine.

    Responsibilities:
        1. Create and hold the AI provider instance
        2. Manage conversation memory for this session
        3. Build the system prompt (base + dynamic context)
        4. Optionally call external tools before each response
        5. Send messages to the provider and return the response
        6. Handle streaming responses
        7. Process slash commands (/help, /clear, /memory, etc.)

    One instance = one conversation session.
    For multiple users, create multiple instances (one per user/session).

    Attributes:
        session_id   : Unique identifier for this conversation.
        provider_name: Which AI backend is active.
        model        : Which model is active.
        stream       : Whether to stream responses by default.
        memory       : The ConversationMemory for this session.
        provider     : The AIProvider instance.
        tool_registry: The ToolRegistry for external tool calls.
        _prompt_variant: Which system prompt style is active.
        _user_context  : Dict of user context for prompt personalisation.

    Args:
        session_id    : Unique session identifier. Default: "default".
        provider_name : Override the config provider. Default: uses config.
        stream        : Enable streaming responses. Default: uses config.
        memory_store  : Custom MemoryStore backend. Default: InMemoryStore.
        tool_registry : Custom ToolRegistry. Default: default_tool_registry.
        prompt_variant: Starting prompt style. Default: "default".
        auto_tools    : Whether to auto-call tools on keywords. Default: False.
    """

    # Slash commands recognised by the CLI
    COMMANDS: frozenset[str] = frozenset({
        "/help", "/clear", "/reset", "/memory",
        "/history", "/variant", "/quit", "/exit",
        "/tools", "/provider",
    })

    def __init__(
        self,
        session_id:     str                   = "default",
        provider_name:  Optional[str]         = None,
        stream:         Optional[bool]        = None,
        memory_store:   Optional[MemoryStore] = None,
        tool_registry:  Optional[ToolRegistry]= None,
        prompt_variant: str                   = "default",
        auto_tools:     bool                  = False,
    ) -> None:
        """
        Initialise the FarmCity AI chatbot.

        Args:
            session_id    : Unique ID for this conversation session.
            provider_name : AI provider override ("openai", "anthropic",
                            "gemini", "ollama"). Defaults to config value.
            stream        : Enable streaming. Defaults to config value.
            memory_store  : Storage backend for conversation history.
                            Defaults to InMemoryStore (dict in RAM).
            tool_registry : External tool registry for weather, prices etc.
                            Defaults to the pre-loaded default_tool_registry.
            prompt_variant: System prompt style. One of:
                            "default", "concise", "expert", "hindi", "seller"
            auto_tools    : If True, the chatbot automatically calls tools
                            based on keywords in the user message.
                            e.g. "weather in Punjab" triggers the weather tool.
        """
        self.session_id     = session_id
        self.stream         = stream if stream is not None else config.stream
        self.auto_tools     = auto_tools
        self._prompt_variant = prompt_variant

        # User context for prompt personalisation (set via set_user_context())
        self._user_context: dict[str, Optional[str]] = {
            "user_name":  None,
            "location":   None,
            "farm_type":  None,
            "language":   None,
            "season":     None,
        }

        # Initialise the AI provider
        # ProviderFactory reads the provider name and creates the right class
        self.provider: AIProvider = ProviderFactory.create(provider_name)
        self.provider_name: str   = self.provider.provider_name
        self.model: str           = self.provider.model

        # Initialise conversation memory
        # get_or_create() returns existing session or creates a new one
        store = memory_store or InMemoryStore()
        self.memory: ConversationMemory = store.get_or_create(session_id)

        # Set the initial system prompt
        # This is done here (not in _build_prompt) so the memory always
        # has a valid system prompt from the first message onward
        initial_prompt = PromptBuilder.build(
            variant=self._prompt_variant,
            **self._user_context,
        )
        self.memory.set_system_prompt(initial_prompt)

        # Tool registry for external integrations
        self.tool_registry: ToolRegistry = tool_registry or default_tool_registry

        logger.info(
            f"FarmCityChatbot initialised -- "
            f"session: '{session_id}' | "
            f"provider: {self.provider_name} | "
            f"model: {self.model} | "
            f"stream: {self.stream} | "
            f"variant: '{prompt_variant}'"
        )

    # =====================================================
    # PUBLIC INTERFACE -- The methods callers actually use
    # =====================================================

    def chat(self, user_input: str) -> ChatResponse:
        """
        Send a user message and return a structured ChatResponse.

        This is the PRIMARY method. It handles the complete pipeline:
            1. Sanitise and validate the input
            2. Check for slash commands (/help, /clear etc.)
            3. Call external tools if auto_tools is enabled
            4. Rebuild the system prompt with any fresh context
            5. Add user message to memory
            6. Send all messages to the AI provider
            7. Add AI response to memory
            8. Return a ChatResponse with text + metadata

        Args:
            user_input: The raw user message string.

        Returns:
            ChatResponse: Structured response with text and metadata.
                          Never raises -- errors are returned as ChatResponse
                          objects with is_error=True.

        Example:
            bot = FarmCityChatbot()
            resp = bot.chat("What fertilizer for wheat?")
            print(resp.text)
            print(f"Took {resp.elapsed_sec:.2f}s using {resp.provider}")
        """
        # Step 1: Sanitise input
        try:
            clean_input = sanitise_input(user_input)
        except ValueError as exc:
            return self._error_response(str(exc))

        # Step 2: Handle slash commands
        if clean_input.startswith("/"):
            return self._handle_command(clean_input)

        # Step 3: Auto-call tools based on keywords (if enabled)
        tools_used: list[str] = []
        tool_context: dict[str, Optional[str]] = {
            "weather_data": None,
            "market_data":  None,
            "db_results":   None,
            "rag_results":  None,
        }

        if self.auto_tools:
            tool_context, tools_used = self._run_tools(clean_input)

        # Step 4: Rebuild system prompt with latest context + tool data
        fresh_prompt = PromptBuilder.build(
            variant=self._prompt_variant,
            **self._user_context,
            **tool_context,
        )
        self.memory.set_system_prompt(fresh_prompt)

        # Step 5: Add user message to memory
        self.memory.add_user_message(clean_input)

        # Step 6: Send to AI provider and measure elapsed time
        messages = self.memory.get_messages()

        with Timer() as timer:
            response_text = self._send(messages)

        # Step 7: Store AI response in memory
        self.memory.add_assistant_message(response_text)

        logger.info(
            f"Response generated in {timer} -- "
            f"session: '{self.session_id}' | "
            f"turns: {self.memory.turn_count} | "
            f"tools: {tools_used or 'none'}"
        )

        return ChatResponse(
            text=response_text,
            session_id=self.session_id,
            provider=self.provider_name,
            model=self.model,
            elapsed_sec=timer.elapsed,
            is_command=False,
            is_error=False,
            tools_used=tools_used,
        )

    def stream(self, user_input: str) -> Iterator[str]:
        """
        Send a user message and stream the response token by token.

        This is the streaming counterpart to chat(). It yields each
        text token as it arrives from the AI provider.

        IMPORTANT: Unlike chat(), stream() does NOT return a ChatResponse.
        The caller is responsible for:
            1. Collecting all tokens into a full string
            2. Calling memory.add_assistant_message(full_response)
               after the stream is exhausted

        main.py handles this automatically via stream_and_collect().

        Args:
            user_input: The raw user message string.

        Yields:
            str: Individual text tokens from the AI stream.

        Example:
            for token in bot.stream("What is organic farming?"):
                print(token, end="", flush=True)
        """
        # Sanitise
        try:
            clean_input = sanitise_input(user_input)
        except ValueError as exc:
            yield f"Error: {exc}"
            return

        # Commands are not streamed -- return immediately
        if clean_input.startswith("/"):
            resp = self._handle_command(clean_input)
            yield resp.text
            return

        # Tool execution
        tool_context: dict[str, Optional[str]] = {
            "weather_data": None,
            "market_data":  None,
            "db_results":   None,
            "rag_results":  None,
        }
        if self.auto_tools:
            tool_context, _ = self._run_tools(clean_input)

        # Rebuild prompt
        fresh_prompt = PromptBuilder.build(
            variant=self._prompt_variant,
            **self._user_context,
            **tool_context,
        )
        self.memory.set_system_prompt(fresh_prompt)

        # Add user message
        self.memory.add_user_message(clean_input)
        messages = self.memory.get_messages()

        # Stream tokens and collect full response simultaneously
        collected: list[str] = []
        try:
            for token in self.provider.stream_chat(messages):
                collected.append(token)
                yield token
        finally:
            # Always store the response even if stream is interrupted
            full_response = "".join(collected)
            if full_response.strip():
                self.memory.add_assistant_message(full_response)
                logger.info(
                    f"Stream complete -- "
                    f"session: '{self.session_id}' | "
                    f"tokens streamed: {len(collected)}"
                )

    def set_user_context(
        self,
        user_name:  Optional[str] = None,
        location:   Optional[str] = None,
        farm_type:  Optional[str] = None,
        language:   Optional[str] = None,
        season:     Optional[str] = None,
    ) -> None:
        """
        Set user profile context for prompt personalisation.

        After calling this, every subsequent chat() call will include
        this context in the system prompt, making responses more
        relevant and personalised.

        Only non-None values are updated -- existing values are preserved.

        Args:
            user_name : Farmer's display name (e.g. "Ramesh Kumar").
            location  : State or district (e.g. "Ludhiana, Punjab").
            farm_type : Farm description (e.g. "Wheat farm, 8 acres").
            language  : Preferred language (e.g. "Hindi", "Punjabi").
            season    : Current season (e.g. "Rabi 2024-25").

        Example:
            bot.set_user_context(
                user_name="Ramesh",
                location="Punjab",
                language="Hindi",
            )
            # Now every response will be personalised for Ramesh in Punjab
        """
        if user_name  is not None: self._user_context["user_name"]  = user_name
        if location   is not None: self._user_context["location"]   = location
        if farm_type  is not None: self._user_context["farm_type"]  = farm_type
        if language   is not None: self._user_context["language"]   = language
        if season     is not None: self._user_context["season"]     = season

        # Rebuild and apply the updated system prompt immediately
        updated_prompt = PromptBuilder.build(
            variant=self._prompt_variant,
            **self._user_context,
        )
        self.memory.set_system_prompt(updated_prompt)

        logger.info(
            f"User context updated -- session: '{self.session_id}' | "
            f"context: {self._non_null_context()}"
        )

    def set_variant(self, variant: str) -> None:
        """
        Switch the active system prompt variant at runtime.

        This lets the user change the AI's communication style
        mid-conversation without losing their chat history.

        Args:
            variant: New prompt variant name. One of:
                     "default", "concise", "expert", "hindi", "seller"

        Raises:
            ValueError: If the variant name is not recognised.

        Example:
            bot.set_variant("hindi")    # Switch to Hindi responses
            bot.set_variant("concise")  # Switch to short answers
        """
        # This will raise ValueError if invalid -- let it propagate to caller
        new_prompt = PromptBuilder.build(
            variant=variant,
            **self._user_context,
        )
        self._prompt_variant = variant
        self.memory.set_system_prompt(new_prompt)
        logger.info(f"Prompt variant changed to '{variant}' -- session: '{self.session_id}'")

    def clear_history(self, keep_system_prompt: bool = True) -> None:
        """
        Clear the conversation history.

        Args:
            keep_system_prompt: If True (default), the system prompt and
                                user context are preserved. The AI keeps
                                its instructions but the chat starts fresh.
                                If False, everything is wiped completely.
        """
        self.memory.clear(keep_system_prompt=keep_system_prompt)
        logger.info(
            f"History cleared -- session: '{self.session_id}' | "
            f"kept_prompt: {keep_system_prompt}"
        )

    def get_history(self, last_n: int = 5) -> list[dict]:
        """
        Return the last N messages from conversation memory.

        Useful for displaying recent history in the CLI or building
        a history view in a web UI.

        Args:
            last_n: Number of most recent messages to return.
                    Includes all roles (system, user, assistant).
                    Default: 5.

        Returns:
            list[dict]: Last N messages as dicts with "role" and "content".
        """
        all_messages = self.memory.get_messages()
        return all_messages[-last_n:] if len(all_messages) > last_n else all_messages

    def get_stats(self) -> dict:
        """
        Return current session statistics as a dictionary.

        Useful for monitoring, dashboards, or the /memory CLI command.

        Returns:
            dict: Session statistics including message counts and config.
        """
        return {
            "session_id":    self.session_id,
            "provider":      self.provider_name,
            "model":         self.model,
            "variant":       self._prompt_variant,
            "message_count": self.memory.message_count,
            "turn_count":    self.memory.turn_count,
            "max_messages":  self.memory.max_messages,
            "streaming":     self.stream,
            "auto_tools":    self.auto_tools,
            "tools_available": self.tool_registry.list_tools(),
            "user_context":  self._non_null_context(),
        }

    # =====================================================
    # PRIVATE METHODS -- Internal pipeline steps
    # =====================================================

    def _send(self, messages: list[dict]) -> str:
        """
        Send the message list to the AI provider and return the response.

        This is the only place in the codebase where the AI provider
        is actually called. Keeping it isolated makes it easy to:
            - Add request logging around it
            - Mock it in tests
            - Add rate limiting or quota checks

        Args:
            messages: Full conversation history including system prompt.

        Returns:
            str: The AI's response text.
        """
        logger.debug(
            f"Sending {len(messages)} messages to {self.provider_name} "
            f"(model: {self.model})"
        )
        return self.provider.chat(messages)

    def _run_tools(
        self,
        user_input: str,
    ) -> tuple[dict[str, Optional[str]], list[str]]:
        """
        Determine which tools to call based on the user's message,
        call them, and return their results plus the list of tools used.

        This uses simple keyword detection. Future improvement:
        use an LLM to determine which tools to call (tool-calling/
        function-calling pattern used by GPT-4 and Claude).

        Keyword rules:
            weather keywords  -> call "weather" tool
            price keywords    -> call "market_price" tool
            database keywords -> call "db_search" tool
            knowledge keywords-> call "rag_search" tool

        Args:
            user_input: The cleaned user message.

        Returns:
            tuple:
                dict: Tool results keyed by context field name.
                list: Names of tools that were called.
        """
        lower = user_input.lower()
        tool_context: dict[str, Optional[str]] = {
            "weather_data": None,
            "market_data":  None,
            "db_results":   None,
            "rag_results":  None,
        }
        tools_called: list[str] = []

        # Weather tool -- detect location + weather keywords
        weather_keywords = {
            "weather", "rain", "temperature", "humid", "forecast",
            "monsoon", "drought", "flood", "climate", "mausam",
        }
        if any(kw in lower for kw in weather_keywords):
            # Extract a location hint from user context or message
            location_hint = (
                self._user_context.get("location")
                or self._extract_location(user_input)
                or "India"
            )
            result = self.tool_registry.call("weather", location_hint)
            if result:
                tool_context["weather_data"] = result
                tools_called.append("weather")

        # Market price tool -- detect crop + price keywords
        price_keywords = {
            "price", "msp", "mandi", "rate", "cost", "sell",
            "market", "profit", "earning", "income", "bhaav",
        }
        if any(kw in lower for kw in price_keywords):
            crop_hint = self._extract_crop(user_input) or "crop"
            result = self.tool_registry.call("market_price", crop_hint)
            if result:
                tool_context["market_data"] = result
                tools_called.append("market_price")

        # Database search -- detect user-specific data keywords
        db_keywords = {
            "my order", "my farm", "my product", "last order",
            "my history", "i bought", "i sold", "my account",
        }
        if any(kw in lower for kw in db_keywords):
            result = self.tool_registry.call("db_search", user_input)
            if result:
                tool_context["db_results"] = result
                tools_called.append("db_search")

        # RAG search -- detect knowledge-intensive questions
        rag_keywords = {
            "disease", "pest", "treatment", "cure", "spray",
            "symptom", "organic", "certification", "scheme", "how to",
        }
        if any(kw in lower for kw in rag_keywords):
            result = self.tool_registry.call("rag_search", user_input)
            if result:
                tool_context["rag_results"] = result
                tools_called.append("rag_search")

        if tools_called:
            logger.info(f"Tools called: {tools_called}")

        return tool_context, tools_called

    def _extract_location(self, text: str) -> Optional[str]:
        """
        Attempt to extract a location name from the user's message.

        Simple heuristic: look for known Indian state names or common
        location words. A production system would use NER (Named Entity
        Recognition) from spaCy or a similar library.

        Args:
            text: User's message text.

        Returns:
            Optional[str]: Extracted location string or None.
        """
        # List of major Indian states and cities for basic detection
        indian_locations = [
            "punjab", "haryana", "uttar pradesh", "up", "maharashtra",
            "karnataka", "gujarat", "rajasthan", "madhya pradesh", "mp",
            "bihar", "west bengal", "andhra pradesh", "telangana",
            "kerala", "tamil nadu", "delhi", "odisha", "assam",
            "ludhiana", "amritsar", "patna", "pune", "nashik",
            "nagpur", "jaipur", "ahmedabad", "bhopal", "lucknow",
        ]
        lower = text.lower()
        for location in indian_locations:
            if location in lower:
                return location.title()
        return None

    def _extract_crop(self, text: str) -> Optional[str]:
        """
        Attempt to extract a crop name from the user's message.

        Simple keyword matching. Future improvement: use an NER model
        trained on agricultural text.

        Args:
            text: User's message text.

        Returns:
            Optional[str]: Extracted crop name or None.
        """
        crops = [
            "wheat", "rice", "maize", "corn", "cotton", "sugarcane",
            "groundnut", "soybean", "mustard", "gram", "barley",
            "tomato", "onion", "potato", "mango", "banana", "grapes",
            "broccoli", "cauliflower", "cabbage", "spinach", "carrot",
            "turmeric", "ginger", "garlic", "chilli", "lentil",
        ]
        lower = text.lower()
        for crop in crops:
            if crop in lower:
                return crop.capitalize()
        return None

    def _handle_command(self, command_input: str) -> ChatResponse:
        """
        Process a slash command and return a ChatResponse.

        Slash commands allow the user to control the chatbot from the CLI
        without sending a message to the AI. This keeps the conversation
        history clean (commands are not added to memory).

        Supported commands:
            /help              -- Show available commands
            /clear             -- Clear history, keep system prompt
            /reset             -- Full reset including system prompt
            /memory            -- Show memory usage statistics
            /history           -- Show last 5 messages
            /variant <name>    -- Switch prompt style
            /tools             -- List available tools
            /provider          -- Show current provider and model
            /quit or /exit     -- Exit signal (handled by main.py)

        Args:
            command_input: The raw command string (e.g. "/variant hindi").

        Returns:
            ChatResponse: Response with is_command=True and the result text.
        """
        # Split into command and optional argument
        parts = command_input.strip().split(maxsplit=1)
        cmd   = parts[0].lower()
        arg   = parts[1].strip() if len(parts) > 1 else ""

        # ── /help ────────────────────────────────────────────
        if cmd == "/help":
            text = (
                "Available Commands:\n"
                "  /help              -- Show this message\n"
                "  /clear             -- Clear conversation history\n"
                "  /reset             -- Full reset (clears everything)\n"
                "  /memory            -- Show memory usage\n"
                "  /history           -- Show last 5 messages\n"
                "  /variant <name>    -- Switch style: default, concise,\n"
                "                        expert, hindi, seller\n"
                "  /tools             -- List available external tools\n"
                "  /provider          -- Show current AI provider info\n"
                "  /quit or /exit     -- Exit the chatbot"
            )

        # ── /clear ───────────────────────────────────────────
        elif cmd == "/clear":
            self.clear_history(keep_system_prompt=True)
            text = "Conversation cleared. System prompt preserved. Starting fresh!"

        # ── /reset ───────────────────────────────────────────
        elif cmd == "/reset":
            self.clear_history(keep_system_prompt=False)
            # Restore the system prompt from scratch
            fresh_prompt = PromptBuilder.build(variant=self._prompt_variant)
            self.memory.set_system_prompt(fresh_prompt)
            text = "Full reset complete. Everything cleared and restarted."

        # ── /memory ──────────────────────────────────────────
        elif cmd == "/memory":
            stats = self.get_stats()
            used_pct = int(
                (stats["turn_count"] * 2 / max(stats["max_messages"], 1)) * 100
            )
            bar = "#" * (used_pct // 10) + "." * (10 - used_pct // 10)
            text = (
                f"Memory Status:\n"
                f"  Session ID    : {stats['session_id']}\n"
                f"  Messages      : {stats['message_count']}\n"
                f"  Turns         : {stats['turn_count']} / "
                f"{stats['max_messages'] // 2}\n"
                f"  Memory Used   : {used_pct}% [{bar}]\n"
                f"  Provider      : {stats['provider']}\n"
                f"  Model         : {stats['model']}\n"
                f"  Variant       : {stats['variant']}\n"
                f"  Streaming     : {stats['streaming']}\n"
                f"  Auto Tools    : {stats['auto_tools']}"
            )

        # ── /history ─────────────────────────────────────────
        elif cmd == "/history":
            messages = self.get_history(last_n=6)
            if not messages:
                text = "No conversation history yet."
            else:
                lines = ["Last messages:"]
                for msg in messages:
                    role    = msg["role"].upper()
                    content = msg["content"][:120]
                    if len(msg["content"]) > 120:
                        content += "..."
                    lines.append(f"  [{role}] {content}")
                text = "\n".join(lines)

        # ── /variant <name> ──────────────────────────────────
        elif cmd == "/variant":
            if not arg:
                from prompts import PromptBuilder as PB
                available = ", ".join(PB.list_variants())
                text = (
                    f"Current variant: '{self._prompt_variant}'\n"
                    f"Available variants: {available}\n"
                    f"Usage: /variant <name>"
                )
            else:
                try:
                    self.set_variant(arg)
                    text = f"Switched to '{arg}' prompt style."
                except ValueError as exc:
                    text = f"Error: {exc}"

        # ── /tools ───────────────────────────────────────────
        elif cmd == "/tools":
            tools = self.tool_registry.list_tools()
            if tools:
                tool_lines = "\n".join(f"  - {t}" for t in tools)
                text = (
                    f"Available tools ({len(tools)}):\n{tool_lines}\n"
                    f"Auto-tools: {'ON' if self.auto_tools else 'OFF'}\n"
                    f"To enable: create bot with auto_tools=True"
                )
            else:
                text = "No tools registered."

        # ── /provider ────────────────────────────────────────
        elif cmd == "/provider":
            text = (
                f"Current AI Provider:\n"
                f"  Provider  : {self.provider_name}\n"
                f"  Model     : {self.model}\n"
                f"  Streaming : {self.stream}\n"
                f"  Variant   : {self._prompt_variant}"
            )

        # ── /quit or /exit ───────────────────────────────────
        elif cmd in {"/quit", "/exit"}:
            text = "QUIT"   # main.py detects this string to exit the loop

        # ── Unknown command ──────────────────────────────────
        else:
            text = (
                f"Unknown command: '{cmd}'. Type /help to see all commands."
            )

        return ChatResponse(
            text=text,
            session_id=self.session_id,
            provider=self.provider_name,
            model=self.model,
            is_command=True,
        )

    def _error_response(self, message: str) -> ChatResponse:
        """
        Build a ChatResponse for error conditions.

        Args:
            message: Human-readable error description.

        Returns:
            ChatResponse: Response with is_error=True.
        """
        logger.warning(f"Error response generated: {message}")
        return ChatResponse(
            text=message,
            session_id=self.session_id,
            provider=self.provider_name,
            model=self.model,
            is_error=True,
        )

    def _non_null_context(self) -> dict:
        """
        Return only the non-None user context fields.

        Used in log messages to avoid cluttering logs with None values.

        Returns:
            dict: User context fields that have been set.
        """
        return {k: v for k, v in self._user_context.items() if v is not None}

    def __repr__(self) -> str:
        return (
            f"FarmCityChatbot("
            f"session='{self.session_id}', "
            f"provider={self.provider_name}, "
            f"model={self.model}, "
            f"turns={self.memory.turn_count})"
        )


# =====================================================
# CHATBOT FACTORY -- Multi-session management
# =====================================================

class ChatbotFactory:
    """
    Manages multiple FarmCityChatbot instances for multi-user scenarios.

    In a CLI, one chatbot instance per run is fine.
    In a Flask or FastAPI web server, you need one instance PER USER
    so that their conversation history is isolated from other users.

    This factory creates, caches, and destroys chatbot instances
    keyed by session_id (typically the user's ID or a UUID).

    Design Decision -- Why a Factory instead of a global dict?
        A plain dict of bots is fine but has problems:
            - No cleanup mechanism (memory grows forever)
            - No shared configuration enforcement
            - No way to inject a custom memory backend for all sessions

        The Factory adds:
            - get_or_create() -- one-line session management
            - destroy()       -- explicit cleanup
            - session_count   -- monitoring
            - Shared config   -- all bots get the same provider/settings

    Usage (Flask):
        factory = ChatbotFactory()

        @app.route("/chat", methods=["POST"])
        def chat():
            user_id = request.user.id
            bot = factory.get_or_create(user_id)
            resp = bot.chat(request.json["message"])
            return jsonify(resp.to_dict())

        @app.route("/session", methods=["DELETE"])
        def end_session():
            factory.destroy(request.user.id)
            return {"status": "cleared"}
    """

    def __init__(
        self,
        provider_name:  Optional[str] = None,
        prompt_variant: str           = "default",
        auto_tools:     bool          = False,
    ) -> None:
        """
        Initialise the factory with shared configuration.

        Args:
            provider_name : AI provider for all created bots.
            prompt_variant: Default prompt style for all created bots.
            auto_tools    : Whether to enable auto-tools for all bots.
        """
        self._bots:          dict[str, FarmCityChatbot] = {}
        self._provider_name  = provider_name
        self._prompt_variant = prompt_variant
        self._auto_tools     = auto_tools
        logger.debug("ChatbotFactory initialised.")

    def get_or_create(self, session_id: str) -> FarmCityChatbot:
        """
        Return an existing chatbot for this session, or create a new one.

        Args:
            session_id: Unique user or session identifier.

        Returns:
            FarmCityChatbot: The bot for this session.
        """
        if session_id not in self._bots:
            self._bots[session_id] = FarmCityChatbot(
                session_id=session_id,
                provider_name=self._provider_name,
                prompt_variant=self._prompt_variant,
                auto_tools=self._auto_tools,
            )
            logger.info(f"ChatbotFactory: new session created '{session_id}'")
        return self._bots[session_id]

    def destroy(self, session_id: str) -> None:
        """
        Remove and discard a chatbot session.

        Args:
            session_id: Session to destroy. Silent if not found.
        """
        removed = self._bots.pop(session_id, None)
        if removed:
            logger.info(f"ChatbotFactory: session destroyed '{session_id}'")

    def destroy_all(self) -> None:
        """Remove all active sessions. Use on server shutdown."""
        count = len(self._bots)
        self._bots.clear()
        logger.info(f"ChatbotFactory: all {count} sessions destroyed.")

    @property
    def session_count(self) -> int:
        """Number of currently active chatbot sessions."""
        return len(self._bots)

    @property
    def session_ids(self) -> list[str]:
        """List of all active session IDs."""
        return list(self._bots.keys())

    def __repr__(self) -> str:
        return f"ChatbotFactory(sessions={self.session_count})"