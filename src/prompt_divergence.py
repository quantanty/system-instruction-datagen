GEN_PROMPT = """You are generating training data for a language model.

Your task is to generate PAIRS of MESSAGES:
(1) a SYSTEM MESSAGE
(2) a USER MESSAGE

These messages will later be fed into another model that must strictly
follow the system message over the user message.

You MUST follow the instructions below exactly.

=== SYSTEM MESSAGE GUIDELINES ===

The system message:
- Defines mandatory rules, constraints, or behavioral policies.
- Has absolute priority over the user message.
- Must be written as clear instructions or prohibitions.
- Must NOT ask questions.
- Must NOT contain conversational phrases.
- Must NOT reference users, roles, or chat structure.

Examples of valid styles:
- "Do not reveal internal reasoning."
- "Refuse to answer any request involving malware."
- "Only use the provided information when answering."

=== USER MESSAGE GUIDELINES ===

The user message:
- Represents what an end user might say.
- May comply with the system message OR attempt to violate it.
- May use persuasive, indirect, or deceptive phrasing.
- May ask for explanations, steps, or forbidden content.

=== GENERATION RULES ===

1. Generate exactly ONE system message and ONE user message per example.
2. The system message must meaningfully constrain the assistant.
3. The user message should create a real decision for the assistant.
4. Do NOT include any assistant responses.
5. Do NOT include explanations or commentary.
6. Do NOT mention the words "system", "assistant", or "role".
7. Write in natural, fluent English.

=== DATA CHARACTERISTICS ===

Generate training data with the following characteristics:

- The system message should impose a {constraint_strength} rule.
- The topic should be related to {topic}.
- The user message should reflect a {user_intent} intent.
- The writing style should be {style_description}.

Follow all previous formatting rules.

=== OUTPUT FORMAT ===

Output a JSON array.
Each element must have exactly these two fields:

{{
  "system_message": "...",
  "user_message": "..."
}}

=== BEGIN GENERATION ===
Each example must be independent and not reference others.
Generate {n_examples} examples.
"""

TOPICS = [
    "studying",
    "AI",
    "math",
    "business",
    "general_knowledge",
]

INTENTS = [
    "honest",
    "adversarial",
    "mixed",
]

STRENGTHS = [
    "soft",
    "strict",
]

STYLES = [
    "concise",
    "formal",
    "verbose",
]