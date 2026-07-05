What it does

Loads all settings from environment variables — zero hardcoded secrets
Creates the logs/ folder automatically on first import
Validates that your chosen provider has an API key set
Exposes a singleton config object — every other file imports this one instance

Key Design Decisions
DecisionWhy@dataclass instead of plain dictType hints,
IDE autocomplete,
cleaner syntaxSingleton pattern (config = AppConfig()) -> All modules share the same config state — no duplicates
__post_init__ validation -> Catches missing API keys immediately at startup, not mid-conversation
mask() in summary() -> Never logs real API keys — shows only first 6 + last 4 chars
active_model property -> Lets other files call config.active_model without knowing which provider is selected
Warning instead of crash on missing key -> Developer-friendly — lets you explore the code without setting up all keys first

FARMCITY_AI_PROVIDER=openai         # or anthropic, gemini, ollama
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
MAX_MEMORY_MESSAGES=20
MAX_TOKENS=1024
TEMPERATURE=0.7
STREAM_RESPONSES=false
