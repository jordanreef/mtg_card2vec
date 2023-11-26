"""
word2vec-like embedding portion of the model, responsible for capturing the statistical associations between cards.

For the purposes of generating this embedding, the "context window" for a particular sample is simply a 17Lands draft
deck. Training pairs are subsampled combinations of each card that appear within the draft deck.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class Card2VecFFNN(nn.Module):
    """
    Implementation of the Card2Vec embedding model. Based on the Feedforward Neural Net Language Model (FFNN) defined
    in Mikolov et al. 2013, Efficient Estimation of Word Representations in Vector Space. In the this model, the
    weights of a single hidden layer are trained based on a Softmax regression

    Attributes:
        set_size (int)      : Size of the "vocabulary" (i.e. number of cards in the set being trained on)
        embedding_dim (int) : Number of neurons in the hidden layer of the embedding -- a hyperparameter
    """
    def __init__(self, set_size, embedding_dim):
        super(Card2VecFFNN, self).__init__()
        self.embedding = nn.Embedding(set_size, embedding_dim)
        self.hidden = nn.Linear(embedding_dim, set_size)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, target):
        embed_target = self.embedding(target)
        out = self.hidden(embed_target)
        return out


def train_card2vec_embedding(set_size, embedding_dim,  # vocab size and embedding dim
                             training_corpus,          # training set of training pairs
                             epochs, learning_rate):   # training / optimizer hyperparameters
    """
    Creates an instance of a Card2VecFFN model, loads data from the supplied training_corpus, and learns card embeddings

    Arguments:
        set_size (int)           : size of the training 'vocabulary' (i.e. len of the one-hot encodings)
        embedding_dim (int)      : embedding size, hyperparameter
        training_corpus (Tensor) : (N, 2, D) large Tensor of training samples
        epochs (int)             : number of training epochs, hyperparameter
        learning_rate (float)    : SGD learning rate, hyperparameter

    Return:
        card_embeddings : return embedding weights after training
    """
    pass
