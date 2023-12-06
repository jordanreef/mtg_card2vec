"""
Set of functions for performing some downstream task evaluations on the various embeddings.
"""
import torch
import torch.nn.functional as F

from itertools import combinations


class Card2VecEmbeddingEval(object):
    """
    Evals related to the card2vec embedding (i.e. the word2vec-like embedding, not the NLP portion of the model)

    Attributes:
        __ args __
        embed_weights (Tensor) : Learned weight Tensor for trained on a particular set
        set_symbol (str)       : 3-letter MtG set-name -- mostly required to sanity check myself
        device (torch.device)  : device being used for Tensor calculations

        __ attr __
        D (int)       : size of the vocabulary (i.e. how many cards are in the set?)
        embed_size (int) : embedding size

    """
    def __init__(self, embed_weights, set_symbol, device):
        """ Learned word2vec embedding weights """
        self.embed_weights = embed_weights
        self.set_symbol = set_symbol
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
