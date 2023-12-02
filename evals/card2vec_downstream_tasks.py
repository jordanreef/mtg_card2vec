"""
Set of functions for performing some downstream task evaluations on the various embeddings.
"""
import torch
import torch.nn.functional as F


class Card2VecEmbeddingEval(object):
    """
    Evals related to the card2vec embedding (i.e. the word2vec-like embedding, not the NLP portion of the model)
    """
    def __init__(self):
        pass

    def eval_distances(self, a, b):
        """ Calculate the distance between point a and point b in the embedding space.
        Calculates both the euclidean distance and cosine similarity

        Return:
            (dist, sim) : Tuple of (euclidean distance, cosine similarity) between a and b
        """
        dist = torch.cdist(a, b)
        sim = F.cosine_similarity(a, b)
