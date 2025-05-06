# TODO: Add your necessary imports here
from typing import Dict, List, Union
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown
from collections import Counter, defaultdict
import math
nltk.download('brown')
from typing import Union

def load_corpus():
    corpus = brown.words()
    corpus = [word.lower() for word in corpus if word.isalpha()]
    return corpus


def get_bigram_freqs(text: list):
    """ Get bigram frequencies for the provided text.

    Args:
    text -- A `list` containing the tokenized text to be
            used to calculate the frequencies of bigrams
    """
    bigram_freq = Counter()
    for i in range(1, len(text)):
        bigram_freq[(text[i-1], text[i])] += 1
    return bigram_freq

def get_top_n_probabilities(text: List[str], context_words: Union[str, List[str]], n: int) -> Dict[str, float]:
    """
    Get top `n` following words to `context_words` and their probabilities.

    Args:
        text -- A list of tokens to be used as the corpus text.
        context_words -- A string or list of strings to be considered as context.
        n -- An integer indicating how many top tokens to return.

    Returns:
        A dictionary of the top `n` words that follow the context word(s),
        with their associated probabilities.
    """
    bigrams = get_bigram_freqs(text)
    context_words = [context_words] if isinstance(context_words, str) else context_words

    following_freq = Counter()

    for (w1, w2), count in bigrams.items():
        if w1 in context_words:
            following_freq[w2] += count

    total = sum(following_freq.values())
    if total == 0:
        return {}

    probs = {word: count / total for word, count in following_freq.items()}
    top_n = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True)[:n])
    return top_n

#def get_top_n_probabilities(text: list, context_words: Union[str, list], n: int):
#    """ Get top `n` following words to `context_words` and their probabilities
#
#    Args:
#    text -- A list of tokens to be used as the corpus text
#    context_words -- A `str` containing the context word(s) to be considered
#    n    -- An `int` that indicates how many tokens to evaluate
#    """
#    bigrams = get_bigram_freqs(text)
#    context_freq = Counter()
#    following_freq = Counter()
#
#    for (w1, w2), count in bigrams.items():
#        if w1 == context_words:
#            context_freq[w1] += count
#            following_freq[w2] += count
#
#    total = sum(following_freq.values())
#    probs = {word: count / total for word, count in following_freq.items()}
#    top_n = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True)[:n])
#    return top_n

def get_entropy(top_n_dict: dict):
    """ Get entropy of distribution of top `n` bigrams """
    return -sum(p * math.log2(p) for p in top_n_dict.values() if p > 0)


def plot_top_n(top_n_dict: dict):
    """ Plot top `n` """
    words = list(top_n_dict.keys())
    values = list(top_n_dict.values())
    plt.figure(figsize=(14, 5))
    plt.bar(words, values)
    plt.title("Top probabilities")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def get_perplexity(prob_dist: dict, tokens: str):
    entropy = get_entropy(prob_dist)
    print(2 ** entropy)
    return 2 ** entropy


# def get_mean_rank(prob_dist: dict, bigram: str):
#     sorted_probs = sorted(prob_dist.items(), key=lambda item: item[1], reverse=True)
#     mean_rank = 0.0
#     for rank, (word, prob) in enumerate(sorted_probs, start=1):
#         mean_rank += prob * rank
#     return mean_rank

def get_mean_rank(prob_dist: dict, bigram: str):
    """
    Calculate the mean rank of the second word in a bigram across matching context distributions.

    Args:
        prob_dist -- A dict where keys are context words and values are dicts of following word probabilities.
        bigram -- A string in the form "w1 w2".

    Returns:
        The mean rank (1-based) of the second word w2 when sorted by descending probability from w1.
        If the context word doesn't exist or the second word isn't found, returns float('inf').
    """
    w1, w2 = bigram.split()
    
    if w1 not in prob_dist:
        return float('inf')

    next_word_probs = prob_dist[w1]
    sorted_words = sorted(next_word_probs.items(), key=lambda item: item[1], reverse=True)
    
    for idx, (word, _) in enumerate(sorted_words, start=1):
        if word == w2:
            return float(idx)
    
    return float('inf')  # w2 not found in the follow-ups of w1