"""
prompts.py — FarmCity AI Prompt System
========================================
Defines all system prompts, prompt templates, and the PromptBuilder
class that assembles dynamic prompts at runtime.

Architecture — Why a Dedicated prompts.py?
───────────────────────────────────────────
Prompts ARE code. They define the AI's behaviour just as much as any
Python function does. Keeping them in a dedicated file means:

    1. Non-engineers can edit the AI's personality without touching
       business logic (chatbot.py stays untouched)
    2. Multiple prompt variants can coexist (A/B testing)
    3. Prompts are versioned alongside code in git
    4. The PromptBuilder class adds dynamic context cleanly
    5. Future: load prompts from a database or CMS

Design — Three Layers of Prompts:
    Layer 1 — BASE_SYSTEM_PROMPT:
        Static, always included. Defines persona, knowledge domains,
        response style, safety rules, and language behaviour.

    Layer 2 — Context Sections (CONTEXT_TEMPLATES):
        Dynamic blocks injected based on what is known about the user
        at request time.

    Layer 3 — Tool Result Templates (TOOL_TEMPLATES):
        Used when the chatbot calls an external tool and needs to inject
        the result into the prompt before asking the AI to respond.

Usage:
    from prompts import PromptBuilder

    prompt = PromptBuilder.build()
    prompt = PromptBuilder.build(
        user_name="Ramesh",
        location="Punjab, India",
        language="Hindi",
        season="Rabi 2024-25",
        farm_type="Wheat farm, 5 acres",
        weather_data="Temp: 28 degrees C, Humidity: 65%, Light rain expected",
        market_data="Wheat MSP: Rs 2275/quintal (2024-25)",
        db_results="User's last crop: Wheat (Sown: Oct 2024)",
    )
"""

from datetime import datetime
from typing import Optional


BASE_SYSTEM_PROMPT: str = """
You are FarmCity AI -- an expert agricultural assistant built into the
FarmCity platform, a hyperlocal grocery and farm-to-door delivery service
connecting Indian farmers directly with consumers.

WHO YOU ARE
You are a knowledgeable, warm, and practical agricultural advisor.
You have deep expertise in Indian farming practices, crops, soil types,
weather patterns across Indian states, government agricultural schemes,
and modern farming techniques.

You care about the farmer's livelihood and the consumer's health.
Your goal is to give actionable, accurate, and locally relevant advice.
You treat every question with respect and helpfulness.

WHAT YOU KNOW -- YOUR KNOWLEDGE DOMAINS

1. CROPS AND CULTIVATION
   - Kharif crops: Rice, Maize, Cotton, Sugarcane, Groundnut, Soybean
   - Rabi crops: Wheat, Mustard, Gram, Barley, Peas, Lentils
   - Zaid crops: Watermelon, Muskmelon, Cucumber, Bitter gourd
   - Horticulture: Tomato, Onion, Potato, Mango, Banana, Grapes
   - Sowing seasons, growth stages, harvesting time
   - Crop rotation, intercropping, companion planting
   - Seed selection, germination, nursery management

2. SOIL HEALTH AND MANAGEMENT
   - Soil types: Alluvial, Black (Regur), Red, Laterite, Desert, Mountain
   - pH levels, nutrient deficiencies, soil testing
   - Organic matter improvement, composting, green manuring
   - Micronutrient management: Zinc, Iron, Boron, Manganese

3. FERTILIZERS AND NUTRITION
   - Chemical fertilizers: Urea, DAP, MOP, NPK complexes
   - Organic: Vermicompost, FYM, Green manure, Biofertilizers
   - Foliar nutrition, micronutrient sprays
   - Fertilizer schedules by crop and growth stage
   - Safe application rates, timing, and methods

4. CROP DISEASES AND PEST MANAGEMENT
   - Fungal diseases: Blast, Blight, Rust, Powdery mildew, Wilt
   - Bacterial diseases: Leaf blight, Canker, Crown gall
   - Viral diseases: Mosaic, Yellow vein mosaic, Tungro
   - Common pests: Aphids, Whitefly, Stem borer, Armyworm, Thrips
   - Integrated Pest Management (IPM)
   - Organic and chemical pesticide options

5. WEATHER AND CLIMATE
   - Monsoon patterns across Indian states
   - Kharif/Rabi/Zaid season calendar
   - Weather-based crop advisories
   - Drought-resistant varieties and practices
   - Climate-smart agriculture practices

6. IRRIGATION AND WATER MANAGEMENT
   - Traditional methods: Canal, Well, Pond irrigation
   - Modern methods: Drip, Sprinkler, Micro-irrigation
   - Rainwater harvesting, farm ponds, check dams
   - Water conservation techniques

7. GOVERNMENT SCHEMES AND SUPPORT
   - PM-KISAN: Rs 6000/year direct income support
   - PM Fasal Bima Yojana: Crop insurance
   - Soil Health Card Scheme
   - eNAM: National Agriculture Market platform
   - Kisan Credit Card (KCC)
   - NABARD schemes and rural credit

8. MARKET PRICES AND ECONOMICS
   - Minimum Support Price (MSP) system
   - Local mandi prices
   - Direct farmer-to-consumer selling (FarmCity model)
   - Post-harvest losses and how to reduce them

9. LIVESTOCK AND ANIMAL HUSBANDRY
   - Dairy farming: Cow, Buffalo breeds for Indian conditions
   - Poultry: Broiler and layer management
   - Common livestock diseases and vaccination schedules

10. FARM EQUIPMENT AND MECHANISATION
    - Tractors, implements, small farm tools
    - Equipment rental and hiring services
    - Drone use in agriculture

11. ORGANIC FARMING AND SUSTAINABILITY
    - Organic certification process
    - Natural farming: ZBNF (Zero Budget Natural Farming)
    - Composting methods, organic pest control

12. FARMCITY PLATFORM HELP
    - How to list and sell products on FarmCity
    - Managing orders and delivery partners
    - Pricing your produce competitively

HOW YOU RESPOND

CLARITY FIRST:
- Give the direct answer in the FIRST sentence. Never bury the answer.
- Use simple language. Avoid jargon unless the user uses it.

STRUCTURE:
- Use numbered lists for step-by-step processes.
- Use bullet points for options or lists of items.
- Keep responses focused -- do not pad with unnecessary text.

ACTIONABILITY:
- Always end with what the farmer CAN DO RIGHT NOW.
- Give specific quantities, rates, and timings where possible.
  Example: "Apply 2 bags of DAP (100 kg) per acre at sowing time."
  Not: "Apply DAP as needed."

LANGUAGE BEHAVIOUR:
- If the user writes in Hindi, respond entirely in Hindi.
- If the user writes in a mix (Hinglish), respond in Hinglish.
- If the user writes in English, respond in English.
- Adapt vocabulary to the user's apparent education level.

TONE:
- Warm, respectful, patient -- like a knowledgeable neighbour.
- Never talk down to the user.
- Acknowledge the difficulty of farming when appropriate.

WHAT YOU DO NOT DO
- You do NOT give medical advice to humans.
- You do NOT recommend illegal practices or banned pesticides.
- You do NOT make specific financial investment advice.
- You do NOT make guarantees about yield or profit.
- You always recommend consulting a licensed agronomist or Krishi Vigyan
  Kendra (KVK) for serious disease diagnosis or major financial decisions.
- If you are uncertain, say so clearly. Do not guess and present it as fact.

FARMCITY PLATFORM CONTEXT
FarmCity is a hyperlocal grocery and farm-to-door delivery platform.
It connects local farmers and stores directly with consumers in their
city. Farmers can sell vegetables, fruits, dairy, and groceries directly
through the app. Consumers get real-time delivery tracking.
""".strip()


CONCISE_PROMPT: str = """
You are FarmCity AI -- a quick, practical agricultural assistant.
Give SHORT, direct answers only. Maximum 3-4 sentences or a brief list.
No lengthy explanations unless the user asks for more detail.
Focus on the most important actionable advice.
Cover: crops, soil, fertilizers, pests, weather, irrigation,
government schemes, market prices, livestock, and FarmCity platform.
If the user writes in Hindi, respond in Hindi.
""".strip()


EXPERT_PROMPT: str = """
You are FarmCity AI -- an advanced agricultural expert with deep knowledge
of agronomy, soil science, plant pathology, and agricultural economics.
Your audience is experienced farmers and agricultural professionals.
Use technical terminology freely (define when necessary).
Provide detailed, comprehensive answers with scientific reasoning.
Include research references from institutions like ICAR, ICRISAT, or
state agricultural universities when relevant.
""".strip()


HINDI_PROMPT: str = """
Aap FarmCity AI hain -- ek visheshagya krishi sahayak.
Aap hamesha Hindi mein jawab dete hain.
Aap kisanon ko fasal, mitti, khad, keetanashak, sinchai,
sarkari yojanaon, bazaar bhaav, aur pashupaalan ke baare mein
saral aur vyavaharik salah dete hain.
Jawab chhota aur spasht rakhen. Kisan ka naam lekar baat karen agar pata ho.
""".strip()


SELLER_ONBOARDING_PROMPT: str = """
You are FarmCity AI -- a friendly onboarding assistant for new sellers
joining the FarmCity platform.
Your PRIMARY job is to help farmers and stores:
    1. Set up their seller profile on FarmCity
    2. List their products (vegetables, fruits, dairy, grocery)
    3. Set competitive prices for their local market
    4. Understand how deliveries and orders work
    5. Get their first order as quickly as possible

Be encouraging, patient, and step-by-step in your guidance.
Also answer general agricultural questions -- you are still a farm expert.
""".strip()


def _user_context_section(
    user_name:  Optional[str] = None,
    location:   Optional[str] = None,
    farm_type:  Optional[str] = None,
    language:   Optional[str] = None,
    season:     Optional[str] = None,
) -> str:
    """
    Build the user-specific context section of the system prompt.

    Only includes fields that are actually provided. If no context is
    given at all, returns an empty string so nothing is appended.

    Args:
        user_name : Farmer's name (e.g. "Ramesh Kumar").
        location  : State/district (e.g. "Ludhiana, Punjab").
        farm_type : Description of the farm (e.g. "5-acre wheat farm").
        language  : Preferred language (e.g. "Hindi", "Punjabi").
        season    : Current agricultural season (e.g. "Rabi 2024-25").

    Returns:
        str: Formatted context block, or empty string if nothing provided.
    """
    lines: list[str] = []

    if any([user_name, location, farm_type, language, season]):
        lines.append("\n--- CURRENT USER CONTEXT ---")

        if user_name:
            lines.append(f"Farmer Name   : {user_name}")
        if location:
            lines.append(f"Location      : {location}")
        if farm_type:
            lines.append(f"Farm Type     : {farm_type}")
        if language:
            lines.append(f"Language      : Respond primarily in {language}.")
        if season:
            lines.append(f"Current Season: {season}")

        lines.append(
            "Use this context to give localised, relevant advice. "
            "Address the farmer by name if provided."
        )

    return "\n".join(lines)


def _weather_context_section(weather_data: Optional[str] = None) -> str:
    """
    Inject live weather data into the system prompt.

    Args:
        weather_data: Formatted weather string from the weather tool.

    Returns:
        str: Formatted weather context block or empty string.
    """
    if not weather_data:
        return ""
    return (
        "\n--- LIVE WEATHER DATA (use this in your response) ---\n"
        f"{weather_data}\n"
        "Incorporate this weather data naturally into your farming advice."
    )


def _market_context_section(market_data: Optional[str] = None) -> str:
    """
    Inject live market price data into the system prompt.

    Args:
        market_data: Formatted price string from the market tool.

    Returns:
        str: Formatted market context block or empty string.
    """
    if not market_data:
        return ""
    return (
        "\n--- LIVE MARKET PRICES (use this in your response) ---\n"
        f"{market_data}\n"
        "Use these prices to help the farmer understand their selling options."
    )


def _database_context_section(db_results: Optional[str] = None) -> str:
    """
    Inject results from a database query into the system prompt.

    Future: ChromaDB/FAISS/Pinecone for RAG (semantic search results).

    Args:
        db_results: Formatted query results string from PostgreSQL/MongoDB.

    Returns:
        str: Formatted DB context block or empty string.
    """
    if not db_results:
        return ""
    return (
        "\n--- RETRIEVED USER DATA (personalise your response with this) ---\n"
        f"{db_results}\n"
        "Reference this data naturally -- do not just list it back verbatim."
    )


def _rag_context_section(rag_results: Optional[str] = None) -> str:
    """
    Inject RAG (Retrieval-Augmented Generation) search results.

    Future use: When ChromaDB, FAISS, or Pinecone is connected,
    semantic search results from the agricultural knowledge base
    are injected here.

    Args:
        rag_results: Retrieved document chunks from vector search.

    Returns:
        str: Formatted RAG context block or empty string.
    """
    if not rag_results:
        return ""
    return (
        "\n--- RELEVANT KNOWLEDGE BASE RESULTS ---\n"
        f"{rag_results}\n"
        "Base your response on these retrieved documents where relevant."
    )


class PromptBuilder:
    """
    Assembles the final system prompt by combining the base prompt
    with dynamic context sections.

    This is the ONLY class that chatbot.py needs from this module.
    It hides all the template complexity behind a single build() call.

    Design Decision -- Builder Pattern:
        Instead of one massive function with 20 parameters,
        we use a classmethod that accepts only what is available
        and skips empty sections automatically.

        chatbot.py just passes whatever context it has:
            PromptBuilder.build(user_name="Ramesh", location="Punjab")

        PromptBuilder handles the rest -- no if/else in chatbot.py.

    Available variants:
        "default"  -- Full FarmCity expert assistant (recommended)
        "concise"  -- Short answers only
        "expert"   -- Technical, for experienced farmers
        "hindi"    -- Always responds in Hindi
        "seller"   -- Focused on FarmCity seller onboarding

    Usage:
        from prompts import PromptBuilder

        # Minimal
        prompt = PromptBuilder.build()

        # With user context
        prompt = PromptBuilder.build(
            user_name="Ramesh",
            location="Punjab",
            language="Hindi",
        )

        # With live data from tools
        prompt = PromptBuilder.build(
            weather_data="28C, rain tomorrow",
            market_data="Wheat MSP: Rs 2275/quintal",
            db_results="Last purchase: DAP 50kg",
        )

        # Different base variant
        prompt = PromptBuilder.build(variant="concise")
    """

    _VARIANTS: dict[str, str] = {
        "default": BASE_SYSTEM_PROMPT,
        "concise": CONCISE_PROMPT,
        "expert":  EXPERT_PROMPT,
        "hindi":   HINDI_PROMPT,
        "seller":  SELLER_ONBOARDING_PROMPT,
    }

    @classmethod
    def build(
        cls,
        *,
        variant:           str           = "default",
        user_name:         Optional[str] = None,
        location:          Optional[str] = None,
        farm_type:         Optional[str] = None,
        language:          Optional[str] = None,
        season:            Optional[str] = None,
        weather_data:      Optional[str] = None,
        market_data:       Optional[str] = None,
        db_results:        Optional[str] = None,
        rag_results:       Optional[str] = None,
        include_timestamp: bool          = True,
    ) -> str:
        """
        Build and return the complete system prompt string.

        All parameters are keyword-only (the * enforces this).
        This prevents accidental positional argument mistakes.

        Args:
            variant           : Which base prompt to use.
                                One of: default, concise, expert, hindi, seller
            user_name         : Farmer's display name.
            location          : State, district, or city of the farm.
            farm_type         : Description of what the farmer grows.
            language          : User's preferred response language.
            season            : Current agricultural season label.
            weather_data      : Live weather data string from weather API.
            market_data       : Current MSP or mandi price data.
            db_results        : Data retrieved from PostgreSQL or MongoDB.
            rag_results       : Semantic search results from vector DB.
            include_timestamp : Whether to inject current date and time.
                                Helps the AI give season-aware advice.

        Returns:
            str: Complete, ready-to-use system prompt.

        Raises:
            ValueError: If the variant name is not recognised.
        """
        if variant not in cls._VARIANTS:
            available = ", ".join(cls._VARIANTS.keys())
            raise ValueError(
                f"Unknown prompt variant '{variant}'. "
                f"Available: {available}"
            )

        base = cls._VARIANTS[variant]
        sections: list[str] = [base]

        # Auto-detect current season if not explicitly provided
        if not season:
            season = cls._detect_season()

        # Append user personalisation context
        user_section = _user_context_section(
            user_name=user_name,
            location=location,
            farm_type=farm_type,
            language=language,
            season=season,
        )
        if user_section:
            sections.append(user_section)

        # Append live tool result sections (skipped if data is None)
        for section_fn, data in [
            (_weather_context_section,  weather_data),
            (_market_context_section,   market_data),
            (_database_context_section, db_results),
            (_rag_context_section,       rag_results),
        ]:
            section = section_fn(data)
            if section:
                sections.append(section)

        # Append current date and time for seasonal awareness
        if include_timestamp:
            sections.append(cls._timestamp_section())

        return "\n".join(sections)

    @classmethod
    def get_variant(cls, variant: str) -> str:
        """
        Return a raw prompt variant string without context injection.

        Useful for testing or displaying the base prompt.

        Args:
            variant: One of the known variant names.

        Returns:
            str: The raw base prompt string.

        Raises:
            ValueError: If variant name is not recognised.
        """
        if variant not in cls._VARIANTS:
            available = ", ".join(cls._VARIANTS.keys())
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {available}"
            )
        return cls._VARIANTS[variant]

    @classmethod
    def list_variants(cls) -> list[str]:
        """
        Return all available prompt variant names.

        Returns:
            list[str]: e.g. ["default", "concise", "expert", "hindi", "seller"]
        """
        return list(cls._VARIANTS.keys())

    @classmethod
    def register_variant(cls, name: str, prompt: str) -> None:
        """
        Register a custom prompt variant at runtime.

        Enables external code to add new prompt styles without
        modifying this file (open/closed principle).

        Args:
            name  : Variant key (e.g. "vegetable_specialist").
            prompt: Full system prompt text.

        Raises:
            ValueError: If name or prompt is empty.

        Example:
            PromptBuilder.register_variant(
                "vegetable_specialist",
                "You are a vegetable farming expert..."
            )
            prompt = PromptBuilder.build(variant="vegetable_specialist")
        """
        if not name or not name.strip():
            raise ValueError("Variant name cannot be empty.")
        if not prompt or not prompt.strip():
            raise ValueError("Prompt text cannot be empty.")

        cls._VARIANTS[name.lower().strip()] = prompt.strip()

    @staticmethod
    def _detect_season() -> str:
        """
        Auto-detect the current Indian agricultural season.

        Indian farming has three main seasons:
            Kharif : June -- October   (monsoon crops: Rice, Maize, Cotton)
            Rabi   : October -- March  (winter crops: Wheat, Mustard, Gram)
            Zaid   : March -- June     (summer crops: Watermelon, Cucumber)

        Returns:
            str: Season label with year, e.g. "Rabi 2024-25"
        """
        month = datetime.now().month
        year  = datetime.now().year

        if 6 <= month <= 10:
            return f"Kharif {year}"
        elif month > 10 or month <= 2:
            if month > 10:
                return f"Rabi {year}-{str(year + 1)[-2:]}"
            else:
                return f"Rabi {year - 1}-{str(year)[-2:]}"
        else:
            return f"Zaid {year}"

    @staticmethod
    def _timestamp_section() -> str:
        """
        Generate a timestamp context block for the system prompt.

        Why inject the timestamp?
            Without it, the AI may give advice based on its training
            cutoff date. With the current date it can correctly identify
            the season, give timely sowing advice, and reference the
            current year's MSP rates.

        Returns:
            str: Formatted timestamp block.
        """
        now      = datetime.now()
        date_str = now.strftime("%d %B %Y")
        time_str = now.strftime("%I:%M %p")
        day_str  = now.strftime("%A")

        return (
            "\n--- CURRENT DATE AND TIME (use for seasonal advice) ---\n"
            f"Date : {day_str}, {date_str}\n"
            f"Time : {time_str} IST\n"
            "Use this to give timely, season-appropriate farming advice."
        )