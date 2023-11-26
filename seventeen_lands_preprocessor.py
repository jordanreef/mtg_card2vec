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
import torch


def gen_training_pairs(game_data_csv, num_decks, sample):
    """
    Generates a list of (input_card, output_card) one-hot training pairs from a 17Lands "Game Data" object. Pairs are to
    be used to generate Word2Vec-like Skip-Gram embeddings.

    Args:
        game_data_csv (file) : open file context manager of a game data .csv file
        num_decks (int)      : maximum number of decks to sample from game data .csv file
        sample (float)       : hyperparameter controlling the strength of subsampling

    Return:
        card_names (list<tuple>)  : human-friendly card names packaged with their idx in the one-hot encoding
        training_pairs (Tensor)   : (N, 2, D) tensor, of N D-sized training pairs
    """
    csv_reader = csv.reader(game_data_csv)
    header = next(csv_reader)

    # Extract indices of "deck_" columns -- these contain info on which cards were present in the deck
    card_names = [s[len(re.match("^deck_", s).group(0)):] for s in header if re.match("^deck_", s) is not None]
    deck_idxs = [idx for idx, s in enumerate(header) if re.match("^deck_", s) is not None]

    ######################
    # Subsampling counts #
    ######################
    card_counts = np.zeros(len(deck_idxs), dtype=int)

    decks_found = 0
    draft_id = None
    for game in csv_reader:
        # Check the draft ID -- we only select one game (i.e. one deck) per draft
        if game[2] != draft_id:
            # When draft ID does not equal prev draft ID we are looking at a new deck
            draft_id = game[2]

            deck = [int(float(game[i])) for i in deck_idxs]
            card_counts = np.add(card_counts, deck)

            decks_found += 1
            if decks_found >= num_decks:
                break

    # Subsampling rates are proportional to card occurrences in the corpus
    card_probs = card_counts / np.sum(card_counts)
    subsample_probs = torch.tensor((np.sqrt(card_probs / sample) + 1) * (sample / card_probs))
    subsample_probs = torch.clamp(subsample_probs, 0.0, 1.0)

    ##################
    # Training pairs #
    ##################
    training_pairs = []

    # Reset csv_reader
    csv_reader = csv.reader(game_data_csv)
    header = next(csv_reader)

    decks_found = 0

    draft_id = None
    for game in csv_reader:
        # Check the draft ID -- we only select one game (i.e. one deck) per draft
        if game[2] != draft_id:
            # When draft ID does not equal prev draft ID we are looking at a new deck
            draft_id = game[2]

            deck = [int(float(game[i])) for i in deck_idxs]

            # Generate one-hot encodings for each deck
            one_hots = deck_counts_to_one_hots(deck)

            # Apply subsampling
            one_hots = subsample_one_hots(one_hots, subsample_probs)

            # Generate training pairs using subsampled deck lists
            for pair in itertools.combinations(one_hots, 2):
                training_pairs.append(torch.stack(pair))

            decks_found += 1

            if decks_found >= num_decks:
                break

    return list(enumerate(card_names)), torch.stack(training_pairs)


def deck_counts_to_one_hots(deck_counts: list):
    """
    Utility function that converts a list of card counts (as contained in 17Lands Game Data) to a a list of one-hot
    encodings.

    Arg:
        deck_counts (list) : list of deck card counts, like [0, 2, 1, 0, 0, 5, 0, ...]
    Returns:
        one_hots (Tensor)  : Tensor of shape (N, D) -- with N being deck size and D set size
    """
    set_size = len(deck_counts)
    one_hots = []
    for idx, val in enumerate(deck_counts):
        for c in range(val):
            vec = torch.zeros(set_size, dtype=torch.bool)
            vec[idx] = 1
            one_hots.append(vec)
    return torch.stack(one_hots, dim=0)


def subsample_one_hots(one_hot_encoding, probs):
    """
    Utility function that subsamples entries of a one-hot encoding based on probabilities contained in probs. The
    subsampling scheme follows the method laid out in Mikolov et al. in which indvidual tokens are selected based on
    their frequency in the corpus. Tokens that occur exceedingly frequently as selected with a low probability, whereas
    tokens that occur exceedingly rarely are selected with probability approaching 1.0.

    Practically speaking for us, this makes samples involving basic land cards (i.e. Plains, Island, Swamp, Mountain,
    and Forest) much less frequent.

    Args:
        one_hot_encoding (Tensor) : (N, D) Tensor of "raw" one-hot vectors -- essentially a ~40 card deck-list
        probs (Tensor)            : (D,) Tensor of probabilities for each token in the 'vocab' (i.e. set)
    """
    mask = torch.bernoulli(probs.unsqueeze(0).expand_as(one_hot_encoding)).to(one_hot_encoding.dtype)
    subsample = one_hot_encoding * mask
    subsample = subsample[torch.any(subsample, dim=1)]  # Only return rows that contain a "True" value

    return subsample
