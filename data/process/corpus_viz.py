import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


def analyze_corpus_statistics(corpus_path: str):
    """
    Load a pickled corpus and compute statistics:
    - Total number of words (tokens)
    - Unique word count
    - Unique ratio
    - Per-position token frequency (open, high, low, close) with bar plots
    - Number of unique 2-grams
    """
    corpus_df = pd.read_pickle(corpus_path)
    all_tokens = []
    for tokens_list in corpus_df["text"]:
        all_tokens.extend(tokens_list)

    total_words = len(all_tokens)
    unique_words = len(set(all_tokens))
    unique_ratio = unique_words / total_words if total_words > 0 else 0

    print(f"Total number of words (tokens): {total_words}")
    print(f"Unique word count: {unique_words}")
    print(f"Unique ratio: {unique_ratio:.4f}")

    # Separate tokens by position (open, high, low, close)
    open_tokens = [token[0] for token in all_tokens]
    high_tokens = [token[1] for token in all_tokens]
    low_tokens = [token[2] for token in all_tokens]
    close_tokens = [token[3] for token in all_tokens]

    positions = ["Open", "High", "Low", "Close"]
    tokens_by_position = [open_tokens, high_tokens, low_tokens, close_tokens]

    for pos_name, tokens in zip(positions, tokens_by_position):
        freq = Counter(tokens)
        most_common = freq.most_common(20)
        labels, counts = zip(*most_common)

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(labels)), counts, tick_label=labels)
        plt.title(f"Top 20 {pos_name} Token Frequencies")
        plt.xlabel(f"{pos_name} Token")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Count unique 2-grams
    two_grams = []
    for tokens_list in corpus_df["text"]:
        for i in range(len(tokens_list) - 1):
            two_grams.append((tokens_list[i], tokens_list[i + 1]))
    unique_two_grams = len(set(two_grams))
    print(f"Number of unique 2-grams: {unique_two_grams}")
