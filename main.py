"""
main.py -- FarmCity AI Assistant Entry Point
=============================================
The command-line interface (CLI) for the FarmCity AI chatbot.

This file is the ONLY file you run directly:
    python main.py
    python main.py --provider ollama
    python main.py --variant hindi
    python main.py --stream
    python main.py --session my_farm_session
    python main.py --no-tools

Responsibilities:
    1. Parse command-line arguments (argparse)
    2. Set up logging (via utils.setup_logging)
    3. Validate environment (API keys present, provider reachable)
    4. Create the FarmCityChatbot instance
    5. Run the interactive chat loop
    6. Handle Ctrl+C and EOF gracefully
    7. Print structured responses using utils display helpers

Design -- Separation of Concerns:
    main.py handles ONLY the CLI layer:
        - How the user types input
        - How the response is displayed
        - How the program starts and exits

    It does NOT contain:
        - Any AI provider logic       (models.py)
        - Any memory management       (memory.py)
        - Any prompt construction     (prompts.py)
        - Any business logic          (chatbot.py)

    This means main.py can be replaced by a Flask route or a
    FastAPI endpoint without touching any other file.

Usage Examples:
    # Basic start (uses .env settings)
    python main.py

    # Force a specific provider
    python main.py --provider anthropic
    python main.py --provider ollama
    python main.py --provider gemini

    # Use a specific prompt variant
    python main.py --variant hindi
    python main.py --variant concise
    python main.py --variant expert

    # Enable streaming responses
    python main.py --stream

    # Enable auto-tool calling (weather, prices, etc.)
    python main.py --auto-tools

    # Named session (history survives within the run)
    python main.py --session ramesh_farm

    # Override the AI model
    python main.py --model gpt-4o

    # Set debug logging
    python main.py --log-level debug

    # Combine options
    python main.py --provider ollama --variant hindi --stream --auto-tools
"""

import argparse
import sys
import os
from pathlib import Path

# ── Ensure the farmcity_ai package is importable ──────────────
# When run as `python main.py` from any directory, Python needs
# to find the other .py files. Adding the script's directory to
# sys.path ensures imports always work regardless of where you
# run the command from.
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from chatbot import FarmCityChatbot, ChatResponse
from utils import (
    setup_logging,
    print_welcome,
    print_help,
    print_memory_stats,
    print_response,
    print_stream_start,
    print_stream_end,
    print_stream_token,
    print_error,
    print_info,
    print_success,
    separator,
    stream_and_collect,
)


# =====================================================
# ARGUMENT PARSER
# =====================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser.

    Using argparse (built-in) instead of click or typer because:
        - Zero extra dependencies
        - Built into Python standard library
        - Automatic --help generation
        - Sufficient for this project's needs

    Returns:
        argparse.ArgumentParser: Configured parser ready for parsing.
    """
    parser = argparse.ArgumentParser(
        prog="farmcity_ai",
        description=(
            "FarmCity AI Assistant -- Expert agricultural chatbot\n"
            "Powered by OpenAI, Anthropic, Gemini, or local Ollama models."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --provider ollama\n"
            "  python main.py --variant hindi --stream\n"
            "  python main.py --provider anthropic --auto-tools\n"
            "\n"
            "Environment variables (set in .env file):\n"
            "  FARMCITY_AI_PROVIDER  -- openai | anthropic | gemini | ollama\n"
            "  OPENAI_API_KEY        -- Your OpenAI secret key\n"
            "  ANTHROPIC_API_KEY     -- Your Anthropic secret key\n"
            "  GEMINI_API_KEY        -- Your Google Gemini key\n"
            "  OLLAMA_MODEL          -- Model name for local Ollama\n"
        ),
    )

    # ── Provider Selection ────────────────────────────────────
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "gemini", "ollama"],
        default=None,
        metavar="NAME",
        help=(
            "AI provider to use. Overrides FARMCITY_AI_PROVIDER env var.\n"
            "Choices: openai, anthropic, gemini, ollama\n"
            f"Current default: {config.active_provider}"
        ),
    )

    # ── Model Override ────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL_NAME",
        help=(
            "Override the AI model name.\n"
            "Examples: gpt-4o, gpt-4o-mini, claude-3-5-haiku-20241022,\n"
            "          gemini-1.5-flash, llama3.2\n"
            f"Current default: {config.active_model}"
        ),
    )

    # ── Prompt Variant ────────────────────────────────────────
    parser.add_argument(
        "--variant",
        type=str,
        choices=["default", "concise", "expert", "hindi", "seller"],
        default="default",
        metavar="STYLE",
        help=(
            "System prompt style.\n"
            "  default  -- Full FarmCity expert assistant\n"
            "  concise  -- Short, direct answers only\n"
            "  expert   -- Technical depth for professionals\n"
            "  hindi    -- Always respond in Hindi\n"
            "  seller   -- FarmCity seller onboarding focus\n"
            "Default: default"
        ),
    )

    # ── Streaming ─────────────────────────────────────────────
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help=(
            "Stream the AI response token by token.\n"
            "Text appears as it is generated, like ChatGPT.\n"
            "Default: off (full response shown at once)"
        ),
    )

    # ── Auto Tools ────────────────────────────────────────────
    parser.add_argument(
        "--auto-tools",
        action="store_true",
        default=False,
        dest="auto_tools",
        help=(
            "Automatically call external tools based on keywords.\n"
            "Weather keywords -> weather tool\n"
            "Price keywords   -> market price tool\n"
            "Default: off (tools not called automatically)"
        ),
    )

    # ── Session ID ────────────────────────────────────────────
    parser.add_argument(
        "--session",
        type=str,
        default="default",
        metavar="SESSION_ID",
        help=(
            "Named session identifier.\n"
            "Use the same name across runs to continue a conversation.\n"
            "Different names = different conversation histories.\n"
            "Default: 'default'"
        ),
    )

    # ── User Context ──────────────────────────────────────────
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        metavar="YOUR_NAME",
        help="Your name for personalised responses. e.g. --name Ramesh",
    )

    parser.add_argument(
        "--location",
        type=str,
        default=None,
        metavar="LOCATION",
        help=(
            "Your location for localised advice.\n"
            "e.g. --location 'Ludhiana, Punjab'"
        ),
    )

    parser.add_argument(
        "--farm-type",
        type=str,
        default=None,
        dest="farm_type",
        metavar="DESCRIPTION",
        help=(
            "Describe your farm for tailored advice.\n"
            "e.g. --farm-type 'Wheat farm, 5 acres, Punjab'"
        ),
    )

    # ── Logging ───────────────────────────────────────────────
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="warning",
        dest="log_level",
        metavar="LEVEL",
        help=(
            "Logging verbosity.\n"
            "  debug   -- All internal details (very verbose)\n"
            "  info    -- Key events (provider calls, tool usage)\n"
            "  warning -- Only warnings and errors (default)\n"
            "  error   -- Only errors\n"
            "Default: warning (keeps the CLI clean)"
        ),
    )

    # ── Version ───────────────────────────────────────────────
    parser.add_argument(
        "--version",
        action="version",
        version=f"FarmCity AI v{config.app_version}",
    )

    return parser


# =====================================================
# ENVIRONMENT VALIDATOR
# =====================================================

def validate_environment(provider: str) -> tuple[bool, str]:
    """
    Check that the selected provider has the required configuration.

    Called at startup before creating the chatbot. Catches missing API
    keys early with a clear error message instead of a confusing
    authentication error later during the first chat() call.

    Args:
        provider: The provider name to validate (e.g. "openai").

    Returns:
        tuple[bool, str]:
            bool -- True if the environment is valid.
            str  -- Error message if invalid, empty string if valid.
    """
    provider = provider.lower()

    # Ollama runs locally -- no API key needed
    if provider == "ollama":
        ollama_url = config.ollama_base_url
        return True, f"Ollama at {ollama_url} (ensure `ollama serve` is running)"

    # Cloud providers need API keys
    key_map = {
        "openai":    ("OPENAI_API_KEY",    config.openai_api_key),
        "anthropic": ("ANTHROPIC_API_KEY", config.anthropic_api_key),
        "gemini":    ("GEMINI_API_KEY",    config.gemini_api_key),
    }

    if provider not in key_map:
        return False, f"Unknown provider '{provider}'. Choose: openai, anthropic, gemini, ollama"

    env_var_name, api_key = key_map[provider]

    if not api_key:
        return False, (
            f"No API key found for provider '{provider}'.\n"
            f"  Set environment variable: {env_var_name}=your_key_here\n"
            f"  Or add it to your .env file and restart.\n"
            f"  Get a key at:\n"
            f"    openai    -> https://platform.openai.com/api-keys\n"
            f"    anthropic -> https://console.anthropic.com/\n"
            f"    gemini    -> https://aistudio.google.com/app/apikey"
        )

    return True, ""


# =====================================================
# RESPONSE DISPLAY
# =====================================================

def display_response(
    response:   ChatResponse,
    use_stream: bool = False,
) -> None:
    """
    Display a ChatResponse to the terminal in the correct format.

    Handles three display cases:
        1. Command response  -- plain text, no borders
        2. Error response    -- [ERROR] prefix
        3. Normal response   -- bordered block with timing info

    Args:
        response   : The ChatResponse object from bot.chat().
        use_stream : If True, streaming was used (response already printed).
    """
    # Commands print their own output (no border needed)
    if response.is_command:
        if response.text == "QUIT":
            return  # Exit signal -- handled by the chat loop
        sep = separator("-")
        print(f"\n{sep}")
        print(response.text)
        print(f"{sep}\n")
        return

    # Error responses get a clear prefix
    if response.is_error:
        print_error(response.text)
        return

    # Normal response -- streaming already printed it token by token
    # We still show the metadata footer
    if use_stream:
        # Content already printed by stream_and_collect()
        # Just add the timing/tools footer
        _print_response_footer(response)
        return

    # Non-streaming -- print the full formatted response
    print_response(response.text, add_border=True)
    _print_response_footer(response)


def _print_response_footer(response: ChatResponse) -> None:
    """
    Print the metadata line below each AI response.

    Shows timing and which tools were used. Keeps it subtle so it
    does not distract from the actual content.

    Args:
        response: The ChatResponse with metadata fields.
    """
    tools_str = (
        f" | Tools: {', '.join(response.tools_used)}"
        if response.tools_used
        else ""
    )
    print(
        f"  [{response.provider} / {response.model} | "
        f"{response.elapsed_sec:.2f}s{tools_str}]\n"
    )


# =====================================================
# CHAT LOOP
# =====================================================

def run_chat_loop(bot: FarmCityChatbot, use_stream: bool) -> None:
    """
    Run the main interactive chat loop.

    This is the core of the CLI. It:
        1. Prints the input prompt
        2. Reads user input
        3. Calls bot.chat() or bot.stream()
        4. Displays the response
        5. Repeats until /quit or Ctrl+C

    Design -- Why a separate function?
        Separating the chat loop from main() makes it:
        - Testable (can call run_chat_loop with a mock bot)
        - Reusable (could embed in a Tkinter GUI later)
        - Readable (main() stays short and high-level)

    Args:
        bot       : The configured FarmCityChatbot instance.
        use_stream: Whether to use streaming mode for responses.
    """
    while True:
        try:
            # ── Get user input ────────────────────────────────
            # The prompt shows the session name so multi-session users
            # always know which conversation they are in.
            user_input = input(f"\n[{bot.session_id}] You: ").strip()

            # Skip empty input silently (user just pressed Enter)
            if not user_input:
                continue

            # ── Streaming mode ────────────────────────────────
            if use_stream and not user_input.startswith("/"):
                # Print header
                print_stream_start("FarmCity AI")

                # Stream tokens + collect full response for memory
                full_response = ""
                try:
                    for token in bot.stream(user_input):
                        print_stream_token(token)
                        full_response += token
                finally:
                    print_stream_end()

                # Print metadata footer manually (stream() returns no ChatResponse)
                print(
                    f"  [{bot.provider_name} / {bot.model} | streaming]\n"
                )
                continue

            # ── Normal (non-streaming) mode ───────────────────
            response = bot.chat(user_input)

            # Check for quit signal
            if response.is_command and response.text == "QUIT":
                print_success("Goodbye! Happy farming!")
                break

            # Display the response
            display_response(response, use_stream=False)

            # ── Handle /memory command specially ─────────────
            # (print_memory_stats gives a nicer visual than plain text)
            if user_input.strip().lower() == "/memory":
                stats = bot.get_stats()
                print_memory_stats(
                    message_count=stats["message_count"],
                    turn_count=stats["turn_count"],
                    max_messages=stats["max_messages"],
                    session_id=stats["session_id"],
                )

        except KeyboardInterrupt:
            # Ctrl+C -- graceful exit
            print("\n\nInterrupted. Goodbye! Happy farming!")
            break

        except EOFError:
            # Ctrl+D or piped input ended -- graceful exit
            print("\nInput ended. Goodbye!")
            break

        except Exception as exc:
            # Catch-all for unexpected errors
            # Never let the chat loop crash -- show the error and continue
            print_error(f"Unexpected error: {exc}")
            import logging
            logging.getLogger(__name__).exception("Unexpected error in chat loop")
            continue


# =====================================================
# MAIN ENTRY POINT
# =====================================================

def main() -> None:
    """
    Main entry point for the FarmCity AI CLI.

    Execution flow:
        1. Parse command-line arguments
        2. Set up logging
        3. Load .env file if present
        4. Apply model override if provided
        5. Validate the environment (API key present?)
        6. Create the FarmCityChatbot instance
        7. Set user context if provided via CLI args
        8. Print the welcome banner
        9. Run the interactive chat loop
        10. Exit cleanly

    This function intentionally calls sys.exit(1) on fatal errors
    (missing API key, unknown provider) so the shell gets a non-zero
    exit code for scripting/automation.
    """
    # ── Step 1: Parse arguments ───────────────────────────────
    parser = build_argument_parser()
    args   = parser.parse_args()

    # ── Step 2: Set up logging ────────────────────────────────
    # Use the log level from args (overrides config)
    setup_logging(log_level=args.log_level.upper())
    import logging
    logger = logging.getLogger(__name__)

    # ── Step 3: Load .env file ────────────────────────────────
    # Try to load python-dotenv if available.
    # If not installed, fall back to system environment variables.
    # This means .env works automatically without making python-dotenv
    # a hard requirement (but we still list it in requirements.txt).
    try:
        from dotenv import load_dotenv
        # Look for .env in the same directory as main.py
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f".env file loaded from {env_path}")
        else:
            logger.debug(".env file not found -- using system environment variables")
    except ImportError:
        logger.debug(
            "python-dotenv not installed. "
            "Using system environment variables only. "
            "Install with: pip install python-dotenv"
        )

    # ── Step 4: Determine provider and apply model override ───
    provider_name = args.provider or config.active_provider

    # If a model override is provided, set it in the environment
    # so the config picks it up when the provider is created.
    # This is the cleanest way to override without restructuring config.
    if args.model:
        model_env_map = {
            "openai":    "OPENAI_MODEL",
            "anthropic": "ANTHROPIC_MODEL",
            "gemini":    "GEMINI_MODEL",
            "ollama":    "OLLAMA_MODEL",
        }
        env_key = model_env_map.get(provider_name.lower())
        if env_key:
            os.environ[env_key] = args.model
            logger.info(f"Model overridden to '{args.model}' via --model flag")

    # ── Step 5: Validate environment ──────────────────────────
    is_valid, error_msg = validate_environment(provider_name)
    if not is_valid:
        print_error(f"Configuration error:\n{error_msg}")
        sys.exit(1)

    # ── Step 6: Create the chatbot ────────────────────────────
    try:
        bot = FarmCityChatbot(
            session_id=args.session,
            provider_name=provider_name,
            stream=args.stream,
            prompt_variant=args.variant,
            auto_tools=args.auto_tools,
        )
    except ImportError as exc:
        print_error(
            f"Missing package for provider '{provider_name}':\n"
            f"  {exc}\n"
            f"  Run: pip install -r requirements.txt"
        )
        sys.exit(1)
    except ValueError as exc:
        print_error(f"Configuration error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print_error(f"Failed to start chatbot: {exc}")
        logger.exception("Chatbot initialisation failed")
        sys.exit(1)

    # ── Step 7: Set user context from CLI args ────────────────
    # Only set fields that were explicitly provided
    if any([args.name, args.location, args.farm_type]):
        bot.set_user_context(
            user_name=args.name,
            location=args.location,
            farm_type=args.farm_type,
        )
        logger.info(
            f"User context set from CLI args: "
            f"name={args.name}, location={args.location}, "
            f"farm_type={args.farm_type}"
        )

    # ── Step 8: Print welcome banner ──────────────────────────
    print_welcome(
        app_name=config.app_name,
        version=config.app_version,
        provider=bot.provider_name,
        model=bot.model,
    )

    # Show active settings if non-default options are in use
    active_flags = []
    if args.stream:       active_flags.append("streaming ON")
    if args.auto_tools:   active_flags.append("auto-tools ON")
    if args.variant != "default":
        active_flags.append(f"variant={args.variant}")
    if args.name:         active_flags.append(f"name={args.name}")
    if args.location:     active_flags.append(f"location={args.location}")

    if active_flags:
        print_info("Active options: " + " | ".join(active_flags))

    # ── Step 9: Run the chat loop ─────────────────────────────
    run_chat_loop(bot=bot, use_stream=args.stream)

    # ── Step 10: Clean exit ───────────────────────────────────
    logger.info(
        f"Session ended -- session: '{args.session}' | "
        f"turns: {bot.memory.turn_count}"
    )
    sys.exit(0)



# FLASK EXAMPLE:
# ──────────────
# from flask import Flask, request, jsonify
# from chatbot import ChatbotFactory
#
# app = Flask(__name__)
# factory = ChatbotFactory(provider_name="openai", auto_tools=True)
#
# @app.route("/chat", methods=["POST"])
# def chat():
#     data       = request.get_json()
#     session_id = data.get("session_id", "anonymous")
#     message    = data.get("message", "")
#
#     bot      = factory.get_or_create(session_id)
#     response = bot.chat(message)
#     return jsonify(response.to_dict())
#
# @app.route("/session/<session_id>", methods=["DELETE"])
# def clear_session(session_id):
#     factory.destroy(session_id)
#     return jsonify({"status": "cleared"})
#
# if __name__ == "__main__":
#     setup_logging()
#     app.run(debug=True, port=5000)
#
#
# FASTAPI EXAMPLE:
# ────────────────
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# from chatbot import ChatbotFactory
#
# app     = FastAPI(title="FarmCity AI API")
# factory = ChatbotFactory(provider_name="openai", auto_tools=True)
#
# class ChatRequest(BaseModel):
#     session_id: str = "default"
#     message:    str
#
# @app.post("/chat")
# async def chat(req: ChatRequest):
#     bot      = factory.get_or_create(req.session_id)
#     response = bot.chat(req.message)
#     return response.to_dict()
#
# @app.post("/chat/stream")
# async def chat_stream(req: ChatRequest):
#     bot = factory.get_or_create(req.session_id)
#     def generate():
#         for token in bot.stream(req.message):
#             yield token
#     return StreamingResponse(generate(), media_type="text/plain")
#
# @app.delete("/session/{session_id}")
# async def delete_session(session_id: str):
#     factory.destroy(session_id)
#     return {"status": "cleared"}


# =====================================================
# SCRIPT ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()