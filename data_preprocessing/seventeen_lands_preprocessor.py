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

NAME_2_ID = 0
ID_2_NAME = 1


def gen_card_names(game_data_csv):
    """ Generates two dicts, for converting between card names and one-hot vectors

    Return:
        name_to_one_hot : keys are card names, values are one-hot vector Tensors
        idx_to_names    : keys are integer indexes within the one-hot encoding, values are card names
    """
    csv_reader = csv.reader(game_data_csv)
    header = next(csv_reader)

    card_names = [s[len(re.match("^deck_", s).group(0)):] for s in header if re.match("^deck_", s) is not None]

    name_to_idx = {name: idx for idx, name in enumerate(card_names)}
    idx_to_name = {idx: name for idx, name in enumerate(card_names)}

    return name_to_idx, idx_to_name


def gen_training_pairs(game_data_csv, num_decks, sample):
    """
    Generates a list of (input_card, output_card) one-hot training pairs from a 17Lands "Game Data" object. Pairs are to
    be used to generate Word2Vec-like Skip-Gram embeddings.

    Args:
        game_data_csv (file) : open file context manager of a game data .csv file
        num_decks (int)      : maximum number of decks to sample from game data .csv file
        sample (float)       : hyperparameter controlling the strength of subsampling

    Return:
        training_pairs (Tensor)   : (N, 2) N training pairs
    """

    # Get subsampling probabilities
    # Subsampling rates are proportional to card occurrences in the corpus
    card_probs = generate_card_probs(game_data_csv, num_decks)
    subsample_probs = torch.tensor((np.sqrt(card_probs / sample) + 1) * (sample / card_probs))
    subsample_probs = torch.clamp(subsample_probs, 0.0, 1.0)

    ##################
    # Training pairs #
    ##################

    csv_reader = csv.reader(game_data_csv)
    header = next(csv_reader)

    # Extract indices of "deck_" columns -- these contain info on which cards were present in the deck
    deck_idxs = [idx for idx, s in enumerate(header) if re.match("^deck_", s) is not None]

    training_pairs = []
    decks_found = 0

    draft_id = None
    for game in csv_reader:
        # Check the draft ID -- we only select one game (i.e. one deck) per draft
        if game[2] != draft_id:
            # When draft ID does not equal prev draft ID we are looking at a new deck
            draft_id = game[2]

            deck = [int(float(game[i])) for i in deck_idxs]

            # Generate one-hot encodings for each deck
            deck_list = deck_counts_to_idxs(deck)

            # Apply subsampling
            deck_list = subsample_deck_idxs(deck_list, subsample_probs)

            # Generate training pairs using subsampled deck lists
            for pair in itertools.permutations(deck_list, 2):
                training_pairs.append(torch.stack(pair))

            decks_found += 1
            if decks_found >= num_decks:
                break

    return torch.stack(training_pairs)


def generate_card_probs(game_data_csv, num_decks):
    """
    Generate the probability distribution of each card in the given game data file

    Args:
        game_data_csv (file) : open file context manager for a 17lands game data .csv
        num_decks (int)      : maximum number of decks to sample from game data .csv file
    """
    csv_reader = csv.reader(game_data_csv)
    header = next(csv_reader)

    # Extract indices of "deck_" columns -- these contain info on which cards were present in the deck
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

    game_data_csv.seek(0)  # Seek to the beginning of file after extracting probs

    return card_probs


def deck_counts_to_idxs(deck_counts: list):
    """
    Utility function that converts a list of card counts (as contained in 17Lands Game Data) to a list of indexes (i.e.
    one-hot encodings).

    Arg:
        deck_counts (list) : list of deck card counts, like [0, 2, 1, 0, 0, 5, 0, ...]
    Returns:
        one_hots (list)    : ~40-element list of card indices in their 'one-hot' set encoding
    """
    deck_idxs = []
    for idx, val in enumerate(deck_counts):
        for c in range(val):
            deck_idxs.append(idx)
    return deck_idxs


def subsample_deck_idxs(deck_idxs, probs):
    """
    Utility function that subsamples entries of a one-hot encoding based on probabilities contained in probs. The
    subsampling scheme follows the method laid out in Mikolov et al. in which indvidual tokens are selected based on
    their frequency in the corpus. Tokens that occur exceedingly frequently as selected with a low probability, whereas
    tokens that occur exceedingly rarely are selected with probability approaching 1.0.

    Practically speaking for us, this makes samples involving basic land cards (i.e. Plains, Island, Swamp, Mountain,
    and Forest) much less frequent.

    Args:
        deck_idxs (list)   : (N,) "Raw" decklist -- a (usually) 40-element list of card indexes
        probs (Tensor)     : (D,) Tensor of probabilities for each token in the 'vocab' (i.e. set)
    """
    subsample_mask = torch.bernoulli(probs[deck_idxs])
    subsample = np.array(deck_idxs)[np.array([binary.item() for binary in subsample_mask.to(dtype=bool)])]

    return torch.Tensor(subsample).to(dtype=torch.int)


def scrape_gih_wr(url):
    """ Quick little utility to scrape the GIH WR statistic from a 17Lands table. """
    import requests
    from bs4 import BeautifulSoup
    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By

    driver = webdriver.Chrome()
    driver.get(url)

    table_present = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "table"))
    )

    page_src = driver.page_source

    driver.quit()

    soup = BeautifulSoup(page_src, 'lxml')
    table = soup.find(class_='scrolling-table')
    header_row = table.find('tr')
    headers = [th.text.strip() for th in header_row.find_all('th')]

    try:
        tgt_column_index = headers.index("GIH WR")
    except ValueError:
        print("Target column not found")
        tgt_column_index = None

    if tgt_column_index is None:
        return None
    else:
        gih_wrs = []

        tbody = table.find('tbody')
        for row in tbody.find_all('tr'):
            card_name = row.find(class_='list_card_name').text

            cols = row.find_all('td')
            gih_span = cols[tgt_column_index].find('span')

            try:
                wr = float(gih_span.text.strip().strip('%')) / 100
            except AttributeError:
                wr = 0.35  # A few cards are so unplayably bad that they don't have enough data and don't have a win %

            gih_wrs.append((card_name, wr))

        print("Break")
        return gih_wrs
