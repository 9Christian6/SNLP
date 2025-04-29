from typing import Union
import numpy as np
import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

nltk.download('brown')


def load_corpus():
    """Load `Brown` corpus from NLTK, tokenize and lowercase it"""
    tokens = brown.words()
    tokens = [token.lower() for token in tokens]
    return tokens


def get_bigram_freqs(text: list):
    """ Get bigram frequencies for the provided text."""
    bigram_frequencies = {}

    for i in range(len(text) - 1):
        w1=text[i]
        w2=text[i + 1]
        if w1 not in bigram_frequencies:
            bigram_frequencies[w1] = {}
        if w2 not in bigram_frequencies[w1]:
            bigram_frequencies[w1][w2] = 0
        bigram_frequencies[w1][w2] += 1
    return bigram_frequencies

def get_top_n_probabilities(text: list, context_words: Union[str, list], n: int):
    """ Get top `n` following words to `context_words` and their probabilities """
    if isinstance(context_words, str):
        context_words = [context_words]

    bigram_frequencies = get_bigram_freqs(text)
    results = {}

    for context_word in context_words:
        next_words = bigram_frequencies.get(context_word, {}) #ifnot empty dict
        total_count = sum(next_words.values())
        if total_count == 0:
            results[context_word] = {}
            continue

        probs = {word: count / total_count for word, count in next_words.items()} #make it into probs
        top_n = {}
        probs_temp = probs.copy()

        for a in range(n):
            max_word = None
            max_prob = -1
            for word, prob in probs_temp.items():
                if prob > max_prob:
                    max_prob = prob
                    max_word = word
            if max_word is not None:
                top_n[max_word] = probs_temp.pop(max_word)

        results[context_word] = top_n
    return results


def get_entropy(top_n_dict: dict):
    """ Get entropy of distribution of top `n` bigrams """
    entropies = {}

    for context_word, probs in top_n_dict.items():
        entropy = 0
        for p in probs.values():
            if p > 0:
                entropy += -p * np.log2(p)
        entropies[context_word] = entropy

    return entropies


def plot_top_n(top_n_dict: dict):
    """ Plot top `n` """
    for context, probs in top_n_dict.items():
        plt.figure(figsize=(20, 6))
        plt.bar(probs.keys(), probs.values())
        plt.ylabel('Probability')
        plt.xticks(rotation=90)
        plt.title(f"likelihood of words that come after '{context}'")
        plt.show()

def get_perplexity(phrase: str, bigram_probs=None):
    """Calculate perplexity of a phrase using bigram probabilities"""
    text = phrase.lower().split()

    if bigram_probs is None:
        corpus = load_corpus()
        bigram_frequencies = get_bigram_freqs(corpus)
        bigram_probs = {}
        for w1, counter in bigram_frequencies.items():
            total_count = sum(counter.values())
            bigram_probs[w1] = {}
            for w2, count in counter.items():
                bigram_probs[w1][w2] = count / total_count

    log_probability_sum = 0
    n = 0
    for i in range(len(text) - 1):
        w1=text[i]
        w2=text[i + 1]
        prob = bigram_probs.get(w1, {}).get(w2, 0)

        if prob == 0:

            print(f"Phrase: '{phrase}'  Perplexity: {float('inf')}")
            return float('inf')

        log_probability_sum += np.log2(prob)
        n += 1

    avg_log_prob = log_probability_sum / n
    perplexity = 2 ** (-avg_log_prob)
    print(f"Phrase: '{phrase}'  Perplexity: {perplexity}")
    return perplexity


def get_mean_rank(phrase: str, bigram_probs=None):
    """Calculate mean rank of a phrase using bigram probabilities"""
    tokens = phrase.lower().split()

    if bigram_probs is None:
        corpus = load_corpus()
        bigram_frequencies = get_bigram_freqs(corpus)
        bigram_probs = {}
        for w1, counter in bigram_frequencies.items():
            total_count = sum(counter.values())
            bigram_probs[w1] = {}
            for w2, count in counter.items():
                bigram_probs[w1][w2] = count / total_count

    ranks = []

    for i in range(len(tokens) - 1):
        w1 = tokens[i]
        w2 = tokens[i + 1]

        if w1 in bigram_probs:
            following_words = list(bigram_probs[w1].items())
            following_words.sort(key=lambda pair: pair[1], reverse=True)

            rank = float('inf')  #ifnot found
            index = 1
            for pair in following_words:
                word = pair[0]
                if word == w2:
                    rank = index
                    break
                index += 1

            ranks.append(rank)
        else:
            ranks.append(float('inf'))

    if all(r == float('inf') for r in ranks):
        mean_rank = float('inf')
    else:
        mean_rank = np.mean(ranks)

    print(f"Phrase: '{phrase}'  Mean Rank: {mean_rank}")
    return mean_rank