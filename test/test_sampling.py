from src.prompt_divergence import GEN_PROMPT, INTENTS, STRENGTHS, STYLES, TOPICS
from src.sampling import (
    get_ratio,
    interaction_id_2_tuple,
    p_interaction_from_idx,
    p_interaction_from_tuple,
    p_topic_from_str,
    sample_interaction,
    sample_topic,
)

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