"""
memory.py — Conversation Memory System
=======================================
Manages conversation history for the FarmCity AI Assistant.

This module provides:
    1. ConversationMemory  — Core in-memory storage (default, works today)
    2. MemoryStore (ABC)   — Abstract interface for swappable backends
    3. InMemoryStore       — Dict-based implementation (built-in, no deps)
    4. FileMemoryStore     — JSON file persistence (survives restarts)

Architecture — Why Two Layers?
────────────────────────────────
The memory system is split into two responsibilities:

    Layer 1 — ConversationMemory:
        - Holds ONE conversation's message list
        - Handles trimming (sliding window)
        - Knows about roles (system/user/assistant)
        - Used directly by chatbot.py

    Layer 2 — MemoryStore (ABC + implementations):
        - Manages MULTIPLE sessions (one per user)
        - Stores/loads ConversationMemory by session_id
        - Backend-agnostic interface

    This means:
        Today  → InMemoryStore   (dict in RAM, fast, no setup)
        Later  → RedisStore      (persistent, scalable, multi-server)
        Later  → PostgresStore   (full history, queryable, analytics)
        Later  → MongoStore      (flexible schema, document-based)

    Swapping backends requires changing ONE line in chatbot.py.
    Everything else stays identical.

Design Decision — Sliding Window vs Token Counting:
    We use a message-count sliding window (keep last N messages).
    Alternative: Token counting with tiktoken.
    Why message-count?
        - Provider-agnostic (works with all 4 providers)
        - No extra dependency
        - Predictable behavior
        - Easy to explain to non-engineers
    Upgrade path: Replace _trim() with token-based logic later.

Usage:
    # Basic usage
    memory = ConversationMemory(max_messages=20)
    memory.add_user_message("What crops grow in clay soil?")
    memory.add_assistant_message("Clay soil works well with...")
    messages = memory.get_messages()  # Full history for API call

    # Multi-session usage
    store = InMemoryStore()
    memory = store.get_or_create("user_123")
    memory.add_user_message("Hello!")
"""

import json
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import config


# ─── Logger ──────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Type Alias ──────────────────────────────────────────────
# A single chat message: {"role": "user"|"assistant"|"system", "content": str}
Message = dict[str, str]


# ═════════════════════════════════════════════════════════════
# CORE MEMORY CLASS
# ═════════════════════════════════════════════════════════════

class ConversationMemory:
    """
    Manages the message history for a single conversation session.

    Responsibilities:
        - Stores the ordered list of messages (system + user + assistant)
        - Enforces a maximum message count (sliding window)
        - Provides helper methods for adding and reading messages
        - Tracks metadata: session ID, creation time, message count

    The system prompt is ALWAYS kept at position 0.
    When trimming, only user/assistant messages are removed from the front.
    This ensures the AI always has its instructions — context window or not.

    Attributes:
        session_id   (str)           : Unique ID for this conversation.
        max_messages (int)           : Maximum user+assistant messages to keep.
        created_at   (datetime)      : When this session started.
        _messages    (list[Message]) : Internal message list (mutable).

    Example:
        memory = ConversationMemory(session_id="user_42", max_messages=10)
        memory.set_system_prompt("You are FarmCity AI...")
        memory.add_user_message("What is the best fertilizer for wheat?")
        memory.add_assistant_message("For wheat, NPK 12-32-16 is ideal...")
        print(memory.message_count)   # 3 (system + user + assistant)
        print(memory.turn_count)      # 1 (one full turn = user + assistant)
    """

    def __init__(
        self,
        session_id:   str = "default",
        max_messages: int = None,
    ) -> None:
        """
        Initialise an empty conversation memory.

        Args:
            session_id  : Unique identifier for this session.
                          Use user ID or UUID for multi-user systems.
                          Defaults to "default" for single-user CLI.
            max_messages: Maximum number of user+assistant messages to keep.
                          System prompt is EXCLUDED from this count.
                          Defaults to config.max_memory_messages (from .env).
        """
        self.session_id:   str      = session_id
        self.max_messages: int      = max_messages or config.max_memory_messages
        self.created_at:   datetime = datetime.now()
        self._messages:    list[Message] = []

        logger.debug(
            f"ConversationMemory created — "
            f"session: '{session_id}', max_messages: {self.max_messages}"
        )

    # ─── System Prompt ────────────────────────────────────────

    def set_system_prompt(self, content: str) -> None:
        """
        Set or replace the system prompt for this conversation.

        The system prompt is always placed at index 0 in the message list.
        If a system message already exists, it is replaced (not duplicated).
        If none exists, it is inserted at the front.

        Why always at index 0?
            All major AI providers expect the system message first.
            Moving it would break the conversation format.

        Args:
            content: The full system prompt text.
        """
        system_message: Message = {"role": "system", "content": content}

        if self._messages and self._messages[0]["role"] == "system":
            # Replace existing system prompt
            self._messages[0] = system_message
            logger.debug("System prompt updated.")
        else:
            # Insert at the beginning (before any user/assistant messages)
            self._messages.insert(0, system_message)
            logger.debug("System prompt set.")

    def get_system_prompt(self) -> Optional[str]:
        """
        Return the current system prompt text, or None if not set.

        Returns:
            Optional[str]: System prompt content, or None.
        """
        if self._messages and self._messages[0]["role"] == "system":
            return self._messages[0]["content"]
        return None

    # ─── Adding Messages ──────────────────────────────────────

    def add_user_message(self, content: str) -> None:
        """
        Append a user message to the conversation history.

        After adding, automatically trim old messages if the history
        exceeds max_messages. Trimming NEVER removes the system prompt.

        Args:
            content: The user's input text.
        """
        if not content or not content.strip():
            logger.warning("Attempted to add empty user message — skipped.")
            return

        self._messages.append({
            "role":    "user",
            "content": content.strip(),
        })
        logger.debug(f"User message added. Total messages: {len(self._messages)}")
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        """
        Append an assistant (AI) message to the conversation history.

        Args:
            content: The AI's response text.
        """
        if not content or not content.strip():
            logger.warning("Attempted to add empty assistant message — skipped.")
            return

        self._messages.append({
            "role":    "assistant",
            "content": content.strip(),
        })
        logger.debug(
            f"Assistant message added. Total messages: {len(self._messages)}"
        )

    def add_message(self, role: str, content: str) -> None:
        """
        Generic message adder — use when the role is determined at runtime.

        Args:
            role   : "user", "assistant", or "system".
            content: Message text.

        Raises:
            ValueError: If role is not one of the three allowed values.
        """
        allowed_roles = {"user", "assistant", "system"}
        if role not in allowed_roles:
            raise ValueError(
                f"Invalid role '{role}'. Must be one of: {allowed_roles}"
            )

        if role == "system":
            self.set_system_prompt(content)
        elif role == "user":
            self.add_user_message(content)
        else:
            self.add_assistant_message(content)

    # ─── Reading Messages ─────────────────────────────────────

    def get_messages(self) -> list[Message]:
        """
        Return a deep copy of the full message list.

        Why deep copy?
            Prevents external code from accidentally mutating the
            internal message list (defensive programming).
            The AI provider receives the list and must not change it.

        Returns:
            list[Message]: Full conversation history ready for API call.
        """
        return deepcopy(self._messages)

    def get_last_user_message(self) -> Optional[str]:
        """
        Return the content of the most recent user message.

        Useful for logging or displaying "what you asked" in the UI.

        Returns:
            Optional[str]: Last user message text, or None if history is empty.
        """
        for msg in reversed(self._messages):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """
        Return the content of the most recent assistant response.

        Returns:
            Optional[str]: Last assistant response text, or None.
        """
        for msg in reversed(self._messages):
            if msg["role"] == "assistant":
                return msg["content"]
        return None

    # ─── Memory Management ────────────────────────────────────

    def _trim(self) -> None:
        """
        Enforce the max_messages limit using a sliding window strategy.

        How it works:
            1. Separate the system prompt from user/assistant messages
            2. Count only user/assistant messages against the limit
            3. If over limit, drop the OLDEST user/assistant messages
            4. Always keep the system prompt at position 0

        Example (max_messages=4):
            Before trim: [system, user1, ai1, user2, ai2, user3]
            After trim:  [system, user2, ai2, user3]
            (user1 and ai1 are oldest → removed first)

        Design Decision:
            We remove messages in pairs (user + its assistant response)
            to avoid orphaned turns. An orphaned assistant message with
            no preceding user message would confuse the AI.
        """
        # Separate system prompt from conversation turns
        system_msgs  = [m for m in self._messages if m["role"] == "system"]
        conversation = [m for m in self._messages if m["role"] != "system"]

        # Only trim if over the limit
        if len(conversation) <= self.max_messages:
            return

        # How many messages to remove from the front
        excess = len(conversation) - self.max_messages

        # Always remove in pairs to keep turns aligned
        # Round up to nearest even number so we don't leave orphaned messages
        if excess % 2 != 0:
            excess += 1

        trimmed = conversation[excess:]

        # Rebuild: system prompt first, then trimmed conversation
        self._messages = system_msgs + trimmed

        logger.debug(
            f"Memory trimmed — removed {excess} messages. "
            f"Remaining: {len(self._messages)}"
        )

    def clear(self, keep_system_prompt: bool = True) -> None:
        """
        Clear the conversation history.

        Args:
            keep_system_prompt: If True (default), the system prompt is
                                preserved. The AI keeps its instructions
                                but forgets the conversation history.
                                If False, everything is cleared.

        Example:
            memory.clear()          # Reset conversation, keep instructions
            memory.clear(False)     # Complete wipe — fresh start
        """
        if keep_system_prompt:
            system_prompt = self.get_system_prompt()
            self._messages = []
            if system_prompt:
                self.set_system_prompt(system_prompt)
            logger.info(f"Memory cleared (system prompt kept) — session: {self.session_id}")
        else:
            self._messages = []
            logger.info(f"Memory fully cleared — session: {self.session_id}")

    # ─── Properties ───────────────────────────────────────────

    @property
    def message_count(self) -> int:
        """Total number of messages including system prompt."""
        return len(self._messages)

    @property
    def turn_count(self) -> int:
        """
        Number of complete conversation turns (user + assistant pairs).

        A "turn" is one full exchange: user asks, assistant answers.
        Incomplete turns (user message with no response yet) count as 0.5
        but this property returns only COMPLETE turns.

        Returns:
            int: Number of complete user→assistant exchanges.
        """
        user_count      = sum(1 for m in self._messages if m["role"] == "user")
        assistant_count = sum(1 for m in self._messages if m["role"] == "assistant")
        return min(user_count, assistant_count)

    @property
    def is_empty(self) -> bool:
        """
        True if there are no user or assistant messages (only system prompt or nothing).

        Returns:
            bool: True if conversation has not started yet.
        """
        return all(m["role"] == "system" for m in self._messages)

    # ─── Serialisation ────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Serialise this memory object to a plain dictionary.

        Use this to save memory to a database, Redis, or JSON file.

        Returns:
            dict: Serialisable representation of this memory.
        """
        return {
            "session_id":   self.session_id,
            "max_messages": self.max_messages,
            "created_at":   self.created_at.isoformat(),
            "messages":     deepcopy(self._messages),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        """
        Restore a ConversationMemory from a serialised dictionary.

        Use this when loading a session from a database or file.

        Args:
            data: Dictionary previously created by to_dict().

        Returns:
            ConversationMemory: Restored memory object.
        """
        memory = cls(
            session_id=   data.get("session_id", "default"),
            max_messages= data.get("max_messages", config.max_memory_messages),
        )
        memory._messages = data.get("messages", [])

        # Restore created_at timestamp
        created_str = data.get("created_at")
        if created_str:
            try:
                memory.created_at = datetime.fromisoformat(created_str)
            except ValueError:
                memory.created_at = datetime.now()

        logger.debug(f"Memory restored from dict — session: {memory.session_id}")
        return memory

    def __repr__(self) -> str:
        return (
            f"ConversationMemory("
            f"session_id='{self.session_id}', "
            f"messages={self.message_count}, "
            f"turns={self.turn_count})"
        )

    def __len__(self) -> int:
        """Allow len(memory) to return total message count."""
        return self.message_count


# ═════════════════════════════════════════════════════════════
# ABSTRACT MEMORY STORE — Backend-agnostic interface
# ═════════════════════════════════════════════════════════════

class MemoryStore(ABC):
    """
    Abstract base class for multi-session memory storage backends.

    This defines the contract that ALL storage backends must follow.
    The chatbot only knows about this interface — not the implementation.

    Implementations provided:
        InMemoryStore   → dict in RAM (default, no dependencies)
        FileMemoryStore → JSON files (survives restarts, no dependencies)

    Future implementations (plug in without changing chatbot.py):
        RedisStore      → pip install redis
        PostgresStore   → pip install asyncpg
        MongoStore      → pip install pymongo

    Design Decision — Why a MemoryStore separate from ConversationMemory?
        ConversationMemory handles ONE conversation.
        MemoryStore manages MANY conversations (one per user/session).
        Separating these two concerns makes each class simpler and
        easier to test, swap, or extend independently.
    """

    @abstractmethod
    def get(self, session_id: str) -> Optional[ConversationMemory]:
        """
        Retrieve an existing session by ID.

        Args:
            session_id: Unique session identifier.

        Returns:
            ConversationMemory if found, None if session doesn't exist.
        """
        ...

    @abstractmethod
    def save(self, memory: ConversationMemory) -> None:
        """
        Persist a ConversationMemory to the backend.

        Args:
            memory: The memory object to save.
        """
        ...

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """
        Delete a session permanently from the backend.

        Args:
            session_id: The session to remove.
        """
        ...

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """
        Return all known session IDs.

        Returns:
            list[str]: List of session ID strings.
        """
        ...

    def get_or_create(
        self,
        session_id:    str,
        system_prompt: Optional[str] = None,
    ) -> ConversationMemory:
        """
        Get an existing session or create a new one if it doesn't exist.

        This is the PRIMARY method used by chatbot.py.
        It handles the "find or create" pattern in one call.

        Args:
            session_id   : Unique session identifier.
            system_prompt: If creating a new session, set this system prompt.

        Returns:
            ConversationMemory: Existing or newly created memory object.
        """
        memory = self.get(session_id)

        if memory is None:
            # Brand new session
            memory = ConversationMemory(session_id=session_id)
            if system_prompt:
                memory.set_system_prompt(system_prompt)
            self.save(memory)
            logger.info(f"New session created: '{session_id}'")
        else:
            logger.debug(f"Existing session loaded: '{session_id}'")

        return memory


# ═════════════════════════════════════════════════════════════
# IMPLEMENTATION 1 — IN-MEMORY STORE (Default)
# ═════════════════════════════════════════════════════════════

class InMemoryStore(MemoryStore):
    """
    In-memory session store using a plain Python dictionary.

    Best for:
        - Single-user CLI chatbot (default)
        - Development and testing
        - Applications where history doesn't need to survive restart

    Limitations:
        - Sessions lost on process restart
        - No persistence between runs
        - Single process only (can't scale horizontally)

    Upgrade path:
        Replace with RedisStore when you need:
        - Persistence across restarts
        - Multiple server instances
        - Session expiry (TTL)

    Thread Safety:
        Python's GIL makes dict operations atomic for single reads/writes.
        For production multi-threaded use, add threading.Lock around
        _sessions access.
    """

    def __init__(self) -> None:
        """Initialise an empty session store."""
        self._sessions: dict[str, ConversationMemory] = {}
        logger.debug("InMemoryStore initialised.")

    def get(self, session_id: str) -> Optional[ConversationMemory]:
        """
        Retrieve a session from the in-memory dict.

        Args:
            session_id: Session to look up.

        Returns:
            ConversationMemory or None.
        """
        return self._sessions.get(session_id)

    def save(self, memory: ConversationMemory) -> None:
        """
        Store a ConversationMemory in the dict.

        Args:
            memory: The memory object to store. Keyed by session_id.
        """
        self._sessions[memory.session_id] = memory

    def delete(self, session_id: str) -> None:
        """
        Remove a session from the dict.

        Args:
            session_id: Session to remove. Silent if not found.
        """
        removed = self._sessions.pop(session_id, None)
        if removed:
            logger.info(f"Session deleted: '{session_id}'")

    def list_sessions(self) -> list[str]:
        """
        Return all session IDs currently in memory.

        Returns:
            list[str]: All active session IDs.
        """
        return list(self._sessions.keys())

    @property
    def session_count(self) -> int:
        """Number of active sessions in memory."""
        return len(self._sessions)

    def __repr__(self) -> str:
        return f"InMemoryStore(sessions={self.session_count})"


# ═════════════════════════════════════════════════════════════
# IMPLEMENTATION 2 — FILE-BASED STORE (Persistent)
# ═════════════════════════════════════════════════════════════

class FileMemoryStore(MemoryStore):
    """
    File-based session store — persists conversations to JSON files.

    Each session is saved as a separate JSON file:
        memory_dir/
            session_user_123.json
            session_user_456.json
            session_default.json

    Best for:
        - Single-user applications that need persistence
        - Development where you want conversation history to survive restarts
        - Debugging — you can open the JSON files and read the history

    Limitations:
        - Not suitable for high-concurrency (file I/O is slow)
        - No expiry (files accumulate unless manually cleaned)
        - Not suitable for production multi-user systems → use RedisStore

    Design Decision — One file per session:
        Alternative: One big JSON file with all sessions.
        Problem: Reading/writing the whole file for each message is slow
                 and creates race conditions.
        Solution: One file per session → only affected file is read/written.
    """

    def __init__(self, memory_dir: Optional[Path] = None) -> None:
        """
        Initialise the file store.

        Args:
            memory_dir: Directory to store session JSON files.
                        Defaults to ./logs/memory/ (auto-created).
        """
        self.memory_dir: Path = memory_dir or (Path("logs") / "memory")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"FileMemoryStore initialised at: {self.memory_dir}")

    def _file_path(self, session_id: str) -> Path:
        """
        Return the JSON file path for a given session ID.

        Sanitises the session_id to be filesystem-safe.

        Args:
            session_id: Raw session identifier.

        Returns:
            Path: Safe file path for this session.
        """
        # Replace characters that are invalid in filenames
        safe_id = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in session_id
        )
        return self.memory_dir / f"session_{safe_id}.json"

    def get(self, session_id: str) -> Optional[ConversationMemory]:
        """
        Load a session from its JSON file.

        Args:
            session_id: Session to load.

        Returns:
            ConversationMemory if file exists, None otherwise.
        """
        path = self._file_path(session_id)

        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            memory = ConversationMemory.from_dict(data)
            logger.debug(f"Session loaded from file: {path.name}")
            return memory

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error(f"Failed to load session '{session_id}': {exc}")
            return None

    def save(self, memory: ConversationMemory) -> None:
        """
        Write a session to its JSON file (overwrite if exists).

        Args:
            memory: The memory object to persist.
        """
        path = self._file_path(memory.session_id)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(memory.to_dict(), f, indent=2, ensure_ascii=False)
            logger.debug(f"Session saved to file: {path.name}")

        except OSError as exc:
            logger.error(f"Failed to save session '{memory.session_id}': {exc}")

    def delete(self, session_id: str) -> None:
        """
        Delete the JSON file for a session.

        Args:
            session_id: Session to delete. Silent if file doesn't exist.
        """
        path = self._file_path(session_id)
        if path.exists():
            path.unlink()
            logger.info(f"Session file deleted: {path.name}")

    def list_sessions(self) -> list[str]:
        """
        Return all session IDs by scanning the memory directory.

        Returns:
            list[str]: All session IDs (derived from filenames).
        """
        sessions = []
        for file in self.memory_dir.glob("session_*.json"):
            # Reverse the filename → session_id mapping
            session_id = file.stem.replace("session_", "", 1)
            sessions.append(session_id)
        return sorted(sessions)

    def __repr__(self) -> str:
        count = len(self.list_sessions())
        return f"FileMemoryStore(dir='{self.memory_dir}', sessions={count})"


# ═════════════════════════════════════════════════════════════
# FUTURE BACKEND TEMPLATE
# ═════════════════════════════════════════════════════════════

class RedisStore(MemoryStore):
    """
    Redis-backed session store — TEMPLATE for future implementation.

    When you're ready to add Redis:
        1. pip install redis
        2. Uncomment and complete this class
        3. Change ONE line in chatbot.py:
               store = RedisStore(host="localhost", port=6379)

    Why Redis for production?
        - Survives server restarts
        - TTL support (sessions auto-expire after X days)
        - Works across multiple server instances (horizontal scaling)
        - Extremely fast (sub-millisecond reads/writes)
        - Native support in cloud platforms (AWS ElastiCache, etc.)

    Implementation sketch (uncomment when ready):
    ─────────────────────────────────────────────
    def __init__(self, host="localhost", port=6379, db=0, ttl_seconds=86400):
        import redis
        self._redis = redis.Redis(host=host, port=port, db=db)
        self._ttl = ttl_seconds
        self._prefix = "farmcity:memory:"

    def get(self, session_id):
        data = self._redis.get(self._prefix + session_id)
        if data is None:
            return None
        return ConversationMemory.from_dict(json.loads(data))

    def save(self, memory):
        key = self._prefix + memory.session_id
        self._redis.setex(key, self._ttl, json.dumps(memory.to_dict()))

    def delete(self, session_id):
        self._redis.delete(self._prefix + session_id)

    def list_sessions(self):
        keys = self._redis.keys(self._prefix + "*")
        return [k.decode().replace(self._prefix, "") for k in keys]
    """

    def get(self, session_id: str) -> Optional[ConversationMemory]:
        raise NotImplementedError("RedisStore not yet implemented. See docstring.")

    def save(self, memory: ConversationMemory) -> None:
        raise NotImplementedError("RedisStore not yet implemented. See docstring.")

    def delete(self, session_id: str) -> None:
        raise NotImplementedError("RedisStore not yet implemented. See docstring.")

    def list_sessions(self) -> list[str]:
        raise NotImplementedError("RedisStore not yet implemented. See docstring.")