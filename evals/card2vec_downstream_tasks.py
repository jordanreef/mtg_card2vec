"""
Set of functions for performing some downstream task evaluations on the various embeddings.
"""
import torch
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib.pyplot as plt

from itertools import combinations

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

from data_preprocessing.seventeen_lands_preprocessor import NAME_2_ID, ID_2_NAME


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

    def generate_tsne(self, labels, cluster_names, color_mapping, plot_label,    # Basic plot info
                      perplexity=15.0, learning_rate='auto', random_state=1337,  # TSNE hyperparams
                      save=False, save_dir=None):                                # filesystem / saving info
        """
        Generates a TSNE plot based on the embedding weights of this Card2VecEmbeddingEval.

        Arguments:
            labels (list)         : list of mapping integers; should be len(self.embed_weights)
            cluster_names (dict0  : list of {int: str} cluster labels to appear in the plot legend
            color_mapping (dict)  : dict of {int: color_str} mappings
            plot_label (str)      : additional label to be included in the plot title

            perplexity (float)           : relates to number of nearest neighbors -- larger datasets require larger perplexity
            learning_rate (float or str) : t-SNE lr, too high makes data appear as an equidistant 'ball', too low makes
                                           the data appear too densely packed
            random_state (int)           : random seed used in t-SNE eval

            save (bool)           : set to true to save the figure
            save_dir (str)        : relative path prefix to directory in which to save plot
        """
        plt.clf()  # Ensure figure is clear

        labels = np.array(labels)

        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state)
        c2v_tsne = tsne.fit_transform(self.embed_weights.cpu().numpy())

        # Apply cluster colorations, plot points
        plt.figure(figsize=(14, 10))
        for label, color in color_mapping.items():
            indices = labels == label

            if label == 7:  # Underemphasize the 'Other' class
                scatter = plt.scatter(c2v_tsne[indices, 0], c2v_tsne[indices, 1],
                                      marker='.', color=color, s=200, edgecolors='black', linewidths=0.3, alpha=0.5,
                                      label=cluster_names[label])
            else:           # Emphasis all other classes
                scatter = plt.scatter(c2v_tsne[indices, 0], c2v_tsne[indices, 1],
                                      marker='.', color=color, s=400, edgecolors='black', linewidths=0.5, alpha=0.95,
                                      label=cluster_names[label])

        # Backgroudnd color
        plt.gcf().set_facecolor('whitesmoke')
        scatter.axes.set_facecolor('whitesmoke')

        # Axes aren't really meaningful, so remove them
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')

        plt.title(f'card2vec t-SNE -- {plot_label}', fontweight="bold", fontsize=20)
        plt.legend()

        if save:
            plt.savefig(f"{save_dir}/{plot_label}_tsne.png")
        else:
            plt.show()

        plt.close()

    def draft_pick(self, context, choices, winrates, wr_lower=0.2, wr_upper=0.7,
                   eval_type='additive', cluster_algo='centroid', k=None):
        """
        Make a draft pick! Chooses a card from among choices that maximizes the cosine similarity to a centroid
        calculated from all currently selected cards (the 'context'). These similarities are weighted by the overall
        power-level (winrate statistics) of each card.

        The winrate statistics are turned into a normalized range based on the wr_lower and wr_upper params.

        Arguments:
            context (list)  : list of cards (indices in the embedding) representing current card picks
            choices (list)  : list of cards (indices) -- learner chooses the best of these for their deck
            winrates (list) : winrate statistics scraped from 17Lands -- used as a prior weighting for card selection

            wr_lower (float) : Lower bound of the winrate normalization range
            wr_upper (float) : Upper bound of the winrate normalization range

            eval_type (str) : One of { additive, multiplicative, no_embed, no_winrate }
                              Determines the calculation used to finally make the draft pick:
                                - additive       : adds the similarity metric to the winrate metric
                                - multiplicative : multiplies the similarity metric by the winrate metric
                                - no_embed       : picks based on raw winrate statistics
                                - no_winrate     : picks based only on the embedding similarity

            cluster_algo (str) : One of { centroid, k_means, db_scan }
                                 Which algorithm to use when generating clusters. Either naiive centroid, k_means, db_scan

            k (int) : Choice of k (if using k_means cluster_algo)

        Return:
            choice (int) : index of chosen card within the embedding
        """
        if cluster_algo == 'k_means':
            if k is None:
                raise ValueError("Need to specify value for param 'k' when using cluster_algo == 'k_means'")

            # Don't use more clusters than there are context vectors
            if k > len(context):
                k = len(context)

        ctx_idxs = [self.card_names[0][name] for name in context]
        choice_idxs = [self.card_names[0][name] for name in choices]

        sims = None  # similarities determined by clustering algorithm
        if cluster_algo == 'centroid':  # naiive centroid ______________________________________________________________
            # Single centroid of context
            ctx_embeddings = torch.stack([F.embedding(torch.tensor(idx).to(self.device), weight=self.embed_weights)
                                          for idx in ctx_idxs]).to(self.device)
            ctx_centroid = torch.mean(ctx_embeddings, dim=0)

            choice_embeddings = [F.embedding(torch.tensor(idx).to(self.device), weight=self.embed_weights).to(self.device)
                                 for idx in choice_idxs]

            # Get the similarity between the context centroid and each choice -- these are logits
            sims = torch.tensor([F.cosine_similarity(ctx_centroid, ce, dim=0).item()
                                 for ce in choice_embeddings]).to(self.device)

        elif cluster_algo == 'k_means':  # k_means _____________________________________________________________________
            # K-Means on context
            ctx_embeddings = torch.stack([F.embedding(torch.tensor(idx).to(self.device), weight=self.embed_weights)
                                          for idx in ctx_idxs]).cpu().numpy()

            kmeans = KMeans(n_clusters=k, random_state=2024, n_init=20, algorithm='elkan')
            kmeans.fit(ctx_embeddings)

            # Find the largest cluster
            biggest_cluster_idx = np.argmax(np.bincount(kmeans.labels_))
            big_cluster = kmeans.cluster_centers_[biggest_cluster_idx]

            # Get the embeddings for each card in the choices
            choice_embeddings = [F.embedding(torch.tensor(idx).to(self.device), weight=self.embed_weights).cpu().numpy()
                                 for idx in choice_idxs]

            # # For each choice, get its highest similarity to any of the k centroids -- these are logits
            # sims = np.min(pairwise_distances(choice_embeddings, kmeans.cluster_centers_, metric='cosine'), axis=1)

            # Cosine similarity
            sims = [np.dot(ce, big_cluster) / (np.linalg.norm(ce) * np.linalg.norm(big_cluster))
                    for ce in choice_embeddings]
            sims = torch.tensor(sims).to(self.device)  # Pass back to torch

            print("Break")
        else:
            raise ValueError("Invalid cluster_algo passed to draft_pick()")

        # ______________________________________________________________________________________________________________

        # Take a Softmax over the similarities
        sims = F.softmax(sims, dim=0)

        # Preprocess card winrates
        _wrs = [(self.card_names[NAME_2_ID][tup[0]], tup[1]) for tup in winrates]
        _wrs = sorted(_wrs, key=lambda x: x[0])
        _wr_mean = np.mean([tup[1] for tup in _wrs])  # Avg winrate -- used as a default for missing datapoints later
                                                      # This mostly applies to basic lands

        # Some datapoints are missing from winrate data, so we need to fill in the gap (basic lands, mostly)
        wrs = [None for _ in range(len(self.embed_weights))]
        for emb_idx in range(len(self.embed_weights)):
            if emb_idx in [tup[0] for tup in _wrs]:
                wrs[emb_idx] = next((tup[1] for tup in _wrs if tup[0] == emb_idx))
            else:
                wrs[emb_idx] = _wr_mean
        wrs = torch.tensor(wrs).to(self.device)

        # Normalize card winrates to supplied range
        wr_max = torch.max(wrs)
        wr_min = torch.min(wrs)
        wrs = wr_lower + (wrs - wr_min) * (wr_upper - wr_lower) / (wr_max - wr_min)

        if eval_type == "additive":
            normalized_sims = torch.tensor([sims[i] + wrs[choice_idxs[i]] for i in range(len(choices))]).to(self.device)
        elif eval_type == "multiplicative":
            normalized_sims = torch.tensor([sims[i] * wrs[choice_idxs[i]] for i in range(len(choices))]).to(self.device)
        elif eval_type == "no_embed":
            normalized_sims = torch.tensor([wrs[choice_idxs[i]] for i in range(len(choices))]).to(self.device)
        elif eval_type == "no_winrate":
            normalized_sims = sims
        else:
            raise ValueError('Invalid eval_type passed to Card2VecEmbeddingEval.draft_pick()')

        # Return the embedding index of the best pick
        return choice_idxs[torch.argmax(normalized_sims).item()]
