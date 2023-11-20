"""
This file contains helper functions for extracting decklists from 17Lands data. We use the "Game Data" files sourced
from 17Lands. Each row corresponds to 1 game, and hence 1 deck. Note that this means any particular deck will appear
several times in the dataset which shouldn't be *too* problematic when it comes to issues of bias.

One row in the "Game Data" .csv is essentially an N-hot encdoded vector of length S, where S is the number of cards in
the set this game was played in. N is the number of copies of that card that were present in the decklist.
"""

# Imports

import csv
