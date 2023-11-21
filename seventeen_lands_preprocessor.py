"""
This file contains helper functions for extracting decklists from 17Lands data. We use the "Game Data" files sourced
from 17Lands. Each row corresponds to 1 game, and hence 1 deck. Note that this means any particular deck will appear
several times in the dataset which shouldn't be *too* problematic when it comes to issues of bias.

One row in the "Game Data" .csv is essentially an N-hot encdoded vector of length S, where S is the number of cards in
the set this game was played in. N is the number of copies of that card that were present in the decklist.
"""

# Imports

import csv
import re
import itertools
import numpy as np


def gen_training_pairs(game_data_csv):
    """
    Generates a list of (input_card, output_card) one-hot training pairs from a 17Lands "Game Data" object. Pairs are to
    be used to generate Word2Vec-like Skip-Gram embeddings.

    Args:
        game_data_csv (file) : open file context manager of a game data .csv file

    Return:
        training_pairs () :
    """
    csv_reader = csv.reader(game_data_csv)
    header = next(csv_reader)

    # Extract indices of "deck_" columns -- these contain info on which cards were present in the deck
    card_names = [s[len(re.match("^deck_", s).group(0)):] for s in header if re.match("^deck_", s) is not None]
    deck_idxs = [idx for idx, s in enumerate(header) if re.match("^deck_", s) is not None]

    training_pairs = []

    draft_id = None
    for game in csv_reader:
        # Check the draft ID -- we only select one game (i.e. one deck) per draft
        if game[2] != draft_id:
            # When draft ID does not equal prev draft ID we are looking at a new deck
            draft_id = game[2]
            deck = [int(float(game[i])) for i in deck_idxs]
            one_hots = deck_counts_to_one_hots(deck)
            training_pairs.append(
                [pair for pair in itertools.combinations(one_hots, 2)]  # Each 40-card deck yields 780 training pairs
            )
            
    return training_pairs


def deck_counts_to_one_hots(deck_counts: list):
    """
    Utility function that converts a list of card counts (as contained in 17Lands Game Data) to a a list of one-hot
    encodings.

    Arg:
        deck_counts (list) : list of deck card counts, like [0, 2, 1, 0, 0, 5, 0, ...]
    """
    set_size = len(deck_counts)
    one_hots = []
    for idx, val in enumerate(deck_counts):
        for c in range(val):
            vec = np.zeros(set_size)
            vec[idx] = 1
            one_hots.append(vec)
    return np.array(one_hots)
