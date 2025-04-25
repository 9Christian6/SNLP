# TODO: Add your necessary imports here
from typing import Union

import numpy as np

def load_corpus():
    """Load `Brown` corpus from NLTK"""
    raise NotImplementedError


def get_bigram_freqs(text: list):
    """ Get bigram frequencies for the provided text.

    Args:
    text -- A `list` containing the tokenized text to be
            used to calculate the frequencies of bigrams
    """
    raise NotImplementedError

def get_top_n_probabilities(text: list, context_words: Union[str, list], n: int):
    """ Get top `n` following words to `context_words` and their probabilities

    Args:
    text -- A list of tokens to be used as the corpus text
    context_words -- A `str` containing the context word(s) to be considered
    n    -- An `int` that indicates how many tokens to evaluate
    """
    raise NotImplementedError


def get_entropy(top_n_dict: dict):
    """ Get entropy of distribution of top `n` bigrams """
    raise NotImplementedError


def plot_top_n(top_n_dict: dict):
    """ Plot top `n` """
    raise NotImplementedError


def get_perplexity():
    raise NotImplementedError


def get_mean_rank():
    raise NotImplementedError

