"""
Set of functions for performing some downstream task evaluations on the various embeddings.
"""
import torch
import torch.nn.functional as F

import numpy as np
import pickle

from itertools import combinations


class Card2VecEmbeddingEval(object):
    """
    Evals related to the card2vec embedding (i.e. the word2vec-like embedding, not the NLP portion of the model)

    Attributes:
        __ args __
        embed_weights (Tensor) : Learned weight Tensor for trained on a particular set
        set_symbol (str)       : 3-letter MtG set-name -- mostly required to sanity check myself
        card_names (tuple)     : tuple of card name index mappings for the set
        device (torch.device)  : device being used for Tensor calculations

        __ attr __
        D (int)       : size of the vocabulary (i.e. how many cards are in the set?)
        embed_size (int) : embedding size

    """
    def __init__(self, embed_weights, set_symbol, card_names, device):
        """ Learned word2vec embedding weights """
        self.embed_weights = embed_weights
        self.set_symbol = set_symbol
        self.card_names = card_names
        self.device = device

        self.D = embed_weights.shape[0]
        self.embed_size = embed_weights.shape[1]

    def eval_distances(self, a, b, calc_euclid=False):
        """ Calculate the distance between two card vectors in the embedding space.
        Optionally calculates the

        Arguments:
            a (int)            : index of the first vector in the vocabulary
            b (int)            : index of the second vector in the vocabulary
            calc_euclid (bool) : calculates euclidean distance if True

        Return:
            (sim, dist) : Tuple of (cosine similarity, euclidean distance) between a and b
                          euclidean distance will be None if calc_euclid unless calc_euclid = True
        """
        _a = F.embedding(torch.tensor(a).to(self.device), weight=self.embed_weights)
        _b = F.embedding(torch.tensor(b).to(self.device), weight=self.embed_weights)

        dist = None
        if calc_euclid:
            dist = torch.norm(_a - _b, p=2)              # Euclidean distance

        sim = F.cosine_similarity(_a, _b, dim=0).item()  # Cosine similarity

        return sim, dist

    def set_pairwise_similarities(self):
        """
        Generates similarity metrics for every card pair in a set.

        Return:
            dists (dict{<tuple>: <float>}) : dict keyed by each 2-combination of indices within the embedding space,
                                             values are the cosine similarities between each pair
        """
        N, _ = self.embed_weights.shape

        dists = {
            pair: self.eval_distances(pair[0], pair[1])
            for pair in list(combinations([i for i in range(N)], 2))
        }

        return dists

    def eval_run_statistics(self, sims):
        """
        Generates useful summary statistics from a training evaluation dict

        Arguments:
            sims (dict) : saved eval dict object
        """
        top5_over_epochs = []     # Top 5 similarity pairs across epochs
        bottom5_over_epochs = []  # Bottom 5 similarity pairs across epochs
        mins_over_epochs = []     # Minimum similarity card pair and value across epochs
        maxs_over_epochs = []     # Maximum similarity card pair and value across epochs
        avg_over_epochs = []      # Averge over all pairwise similarities across epochs
        var_over_epochs = []      # Variance over all pairwise similarities across epochs

        # Number of card pairs to report at the top and bottom of the similarity distribution
        top_N = 5

        for epoch in sims["data"]:
            e_data = np.array(list(epoch.items()))
            cos_sims_argsort = np.argsort(e_data[:, 1, 0])

            # Top 5 most similar pair
            top5 = cos_sims_argsort[-top_N:]
            top5_tups = [tuple(r) for r in e_data[top5][:, 0]]
            top5_over_epochs.append(top5_tups)

            # Top 5 least similar pairs
            bottom5 = cos_sims_argsort[:top_N]
            bottom5_tups = [tuple(r) for r in e_data[bottom5][:, 0]]
            bottom5_over_epochs.append(bottom5_tups)

            # Minimum similarity and pair
            min_pair, min_sim = (tuple(e_data[cos_sims_argsort[:5]][0][0]), e_data[cos_sims_argsort[:top_N]][0][1][0])
            mins_over_epochs.append((min_pair, min_sim))

            # Maximum similarity and pair
            max_pair, max_sim = (tuple(e_data[cos_sims_argsort[-5:]][-1][0]), e_data[cos_sims_argsort[-top_N:]][-1][1][0])
            maxs_over_epochs.append((max_pair, max_sim))

            # Avg similarity for the epoch
            avg_over_epochs.append(np.mean(e_data[:, 1, 0]))

            # Variance of similarities for the epoch
            var_over_epochs.append(np.var(e_data[:, 1, 0]))

        return top5_over_epochs, bottom5_over_epochs, mins_over_epochs, maxs_over_epochs, avg_over_epochs, var_over_epochs
