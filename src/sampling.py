import random

from src.prompt_divergence import INTENTS, STRENGTHS, STYLES, TOPICS

topic_s2i = {
    k: i for i, k in enumerate(TOPICS)
}

# Constraint strengths of system message
strength_s2i = {
    k: i for i, k in enumerate(STRENGTHS)
}

intent_s2i = {
    k: i for i, k in enumerate(INTENTS)
}

style_s2i = {
    k: i for i, k in enumerate(STYLES)
}

p_topic_from_str = {
    "studying": 0.2,
    "AI": 0.2,
    "math": 0.2,
    "business": 0.2,
    "general_knowledge": 0.2,
}
p_topic_from_id = [0.0] * len(TOPICS)
def _calculate_p_topic_from_id():
    total_prob = 0
    n_defined = len(p_topic_from_str.keys())
    for topic, prob in p_topic_from_str.items():
        topic_id = topic_s2i[topic]
        p_topic_from_id[topic_id] = prob
        total_prob += prob
    
    if n_defined < len(TOPICS):
        remaining_probs = 1 - total_prob
        prob = remaining_probs / (len(TOPICS) - n_defined)
        for i in range(len(TOPICS)):
            if p_topic_from_id[i] == 0:
                p_topic_from_id[i] = prob
                p_topic_from_str[topic] = prob

_calculate_p_topic_from_id()

def interaction_tuple_2_idx(intent, strength):
    intent_id = intent_s2i[intent]
    strength_id = strength_s2i[strength]
    interaction_idx = intent_id * len(STRENGTHS) + strength_id
    return interaction_idx

def interaction_id_2_tuple(idx):
    intent_id = int(idx / len(STRENGTHS))
    strength_id = idx % len(STRENGTHS)
    return INTENTS[intent_id], STRENGTHS[strength_id]

def interaction_id_2_tuple_id(idx):
    intent_id = int(idx / len(STRENGTHS))
    strength_id = idx % len(STRENGTHS)
    return intent_id, strength_id
    

p_interaction_from_tuple = {
    ("adversarial", "strict"): 0.35,
    ("mixed", "strict"): 0.25,
    ("honest", "strict"): 0.15,
    # ("mixed", "soft"): 0.10,
    # ("honest", "soft"): 0.10,
    ("adversarial", "soft"): 0.05,
}

p_interaction_from_idx = [0.0] * len(INTENTS) * len(STRENGTHS)
def _calculate_p_interaction_from_idx():
    total_prob = 0
    n_defined = len(p_interaction_from_tuple.keys())
    for (intent, strength), prob in p_interaction_from_tuple.items():
        interaction_idx = interaction_tuple_2_idx(intent, strength)
        p_interaction_from_idx[interaction_idx] = prob
        total_prob += prob
    if n_defined < len(p_interaction_from_idx):
        remaining_probs = 1 - total_prob
        prob = remaining_probs / (len(p_interaction_from_idx) - n_defined)
        for i in range(len(p_interaction_from_idx)):
            if p_interaction_from_idx[i] == 0:
                p_interaction_from_idx[i] = prob
                interaction_tuple = interaction_id_2_tuple(i)
                p_interaction_from_tuple[interaction_tuple] = prob

_calculate_p_interaction_from_idx()

interaction_indices = list(range(len(p_interaction_from_idx)))
def sample_interaction(n_samples):
    choices = random.choices(interaction_indices, p_interaction_from_idx, k=n_samples)
    return choices

topic_indices = list(range(len(TOPICS)))
def sample_topic(n_samples):
    choices = random.choices(topic_indices, p_topic_from_id, k=n_samples)
    return choices

style_indices = list(range(len(STYLES)))
def sample_style(n_samples):
    choices = random.choices(style_indices, k=n_samples)
    return choices

def get_ratio(choices, criteria):
    occurrencies = [0] * len(criteria)
    for c in choices:
        occurrencies[c] += 1
    return [x / len(choices) for x in occurrencies]


if __name__ == '__main__':
    # print(p_interaction_from_idx)
    print("Expected:")
    print("=== topic probs ===")
    print(p_topic_from_str)
    # print(p_topic_from_id)
    print("=== interaction probs ===")
    print(p_interaction_from_tuple)

    n_samples = 1000
    topic_choices = sample_topic(n_samples)
    interaction_choices = sample_interaction(n_samples)
    topic_ratio = get_ratio(topic_choices, TOPICS)
    interaction_ratio = get_ratio(interaction_choices, p_interaction_from_idx)
    
    print("\nSampling result:")
    print("=== topic ratio ===")
    for i, topic in enumerate(TOPICS):
        print(f"{topic}: {topic_ratio[i]}")

    print("=== (user intent x contraint strengths) ratio ===")
    for i in range(len(p_interaction_from_idx)):
        intent, strength = interaction_id_2_tuple(i)
        print(f"{intent} + {strength}: {interaction_ratio[i]}")


    style_choices = sample_style(n_samples)
    style_ratio = get_ratio(style_choices, STYLES)
    print("=== style ratio ===")
    for i, style in enumerate(STYLES):
        print(f"{style}: {style_ratio[i]}")