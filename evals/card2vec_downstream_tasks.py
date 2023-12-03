"""
Set of functions for performing some downstream task evaluations on the various embeddings.
"""
import torch
import torch.nn.functional as F


class Card2VecEmbeddingEval(object):
    """
    Evals related to the card2vec embedding (i.e. the word2vec-like embedding, not the NLP portion of the model)
    """
    def __init__(self, embed_weights):
        """ Learned word2vec embedding weights """
        self.embed_weights = embed_weights

    def eval_distances(self, a, b):
        """ Calculate the distance between point a and point b in the embedding space.
        Calculates both the euclidean distance and cosine similarity

        Return:
            (dist, sim) : Tuple of (euclidean distance, cosine similarity) between a and b
        """
        _a = a @ self.embed_weights
        _b = b @ self.embed_weights

        dist = torch.sqrt(torch.sum(torch.pow(_a - _b, 2)))  # Euclidean distance
        sim = F.cosine_similarity(_a, _b, dim=0).item()      # Cosine similarity

        return dist, sim
