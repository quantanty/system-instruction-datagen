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
- Should sound like a practical instruction, not like a legal policy or academic disclaimer.

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
- Avoid directly using words or phrases that mirror the system constraints.
- Prefer user messages where part of the request can be satisfied and part conflicts with the system message.

=== GENERATION RULES ===

1. Generate exactly ONE system message and ONE user message per example.
2. The system message must meaningfully constrain the assistant.
3. The user message should create a real decision for the assistant.
4. Do NOT include any assistant responses.
5. Do NOT include explanations or commentary.
6. Do NOT mention the words "system", "assistant", or "role".
7. Write in natural, fluent English.
8. If the user message can be correctly answered without reading the system message, the example is invalid.
9. When the user message violates the system message, the violation must be implicit rather than explicit.
10. If the user request is trivially rejectable without reasoning, the example is low quality and should be avoided.
{check_self_contained_explanation}
=== DATA CHARACTERISTICS ===

Generate training data with the following characteristics:

- The system message should impose {constraint_strength_description}.
- The topic should be related to {topic_description}.
- The user message should reflect {user_intent_description}.
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

TOPIC_WORDINGS = {
    "studying": (
        "learning processes, education, skill development, studying methods, "
        "or academic activities encountered by learners"
    ),
    "AI": (
        "artificial intelligence, machine learning, language models, "
        "or the development and application of intelligent systems"
    ),
    "math": (
        "mathematical concepts, reasoning, problem solving, proofs, "
        "or quantitative analysis"
    ),
    "business": (
        "business practices, economics, management, finance, markets, "
        "or organizational decision-making"
    ),
    "general_knowledge": (
        "widely known topics such as history, science, culture, "
        "or commonly taught academic subjects"
    ),
}

INTENT_WORDINGS = {
    "honest": (
        "a sincere and straightforward request that appears cooperative "
        "and aligned with the given instructions"
    ),
    "adversarial": (
        "a request that subtly pressures, tempts, or probes the boundaries "
        "of the given instructions without being explicit"
    ),
    "mixed": (
        "a request that begins cooperatively but gradually introduces elements "
        "that challenge or conflict with the given instructions"
    ),
}

STRENGTH_WORDINGS = {
    "soft": (
        "a flexible or preference-like constraint that guides behavior "
        "without absolute prohibition"
    ),
    "strict": (
        "a firm and non-negotiable constraint that must be followed exactly "
        "without exceptions"
    ),
}

STYLE_WORDINGS = {
    "concise": (
        "brief, direct, and efficient, avoiding unnecessary elaboration"
    ),
    "formal": (
        "professional, neutral, and academic in tone"
    ),
    "verbose": (
        "detailed, thorough, and explanatory, with rich context"
    ),
}