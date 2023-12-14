""" Script for conducting a hyperparameter grid search on word2vec embeddings"""

import torch
import numpy as np

import pickle

from models.card2vec_embedding import train_card2vec_embedding

from data_preprocessing.seventeen_lands_preprocessor import NAME_2_ID
from evals.card2vec_downstream_tasks import Card2VecEmbeddingEval


def card2vec_hyper_parameter_search(set_size, set, set_data, card_names, epochs, device):

    """___ HYPER-PARAMETER GRID ___ """

    # LTR GRID -- 18 runs
    # embed_sizes = [300, 400]
    # lrs = [0.01, 0.001, 0.0001]
    # batch_sizes = [64, 128, 256]

    # WOE GRID -- 4 runs
    embed_sizes = [300, 400]
    lrs = [0.001, 0.0001]
    batch_sizes = [128]

    """ ___________________________ """

    for emb_size in embed_sizes:
        for lr in lrs:
            for bs in batch_sizes:
                vecs = train_card2vec_embedding(set_size=set_size, embedding_dim=emb_size, set=set,
                                                training_corpus=set_data, card_labels=card_names,
                                                epochs=epochs, learning_rate=lr, batch_size=bs, device=device,
                                                evals=True, eval_dir=f"run_data/evals", eval_label=f"{set}_{emb_size}_{lr}_{bs}",
                                                plot=True, plot_dir=f"run_data/losses", plot_label=f"{set}_{emb_size}_{lr}_{bs}")

                torch.save(vecs, f"trained/card2vec_{set}_emb-{emb_size}_epochs-{epochs}_lr-{lr}_bs-{bs}.pt")


def gather_hyperparam_run_metrics(set, card_names, eval_dir, loss_dir, weight_dir,
                                  sims, dissims,
                                  embed_sizes, lrs, batch_sizes):
    """
    Scrapes summary statistics on similarity evaluations and loss curves from a hyperparameter grid search output.
    Output of this function is a clean dense representation of run data that should be easy to generate figures and
    other useful artifacts from.

    Args:
        set (str)             : 3-letter set code (e.g. "WOE")
        card_names (tuple)    : card-name to index mapping tuple generated by seventeen_lands_preprocessor
        eval_dir (str)        : relative path prefix to pairwise evals
        loss_dir (str)        : relative path prefix to loss curves
        weight_dir (str)      : relative path prefix to output weight tensors

        sims (list<tuple>)    : list of card-name tuples in the set, considered 'similar' by a human
        dissims (list<tuple>) : list of card-name tuples in the set, considered 'dissimilar' by a human

        embed_sizes (list)    : list of integer embedding dimensions to iterate over when scraping for files
        lrs (list)            : list of learning rate values to iterate over when scraping for files
        batch_sizes (list)    : list of integer batch sizes to iterate over when scraping for files
    """

    # Get IDs for card name tuples
    sim_ids = []
    for tup in sims:
        sim_ids.append(
            (card_names[NAME_2_ID][tup[0]], card_names[NAME_2_ID][tup[1]])
        )

    # Get IDs for card name tuples
    dissim_ids = []
    for tup in dissims:
        dissim_ids.append(
            (card_names[NAME_2_ID][tup[0]], card_names[NAME_2_ID][tup[1]])
        )

    outs = {}  # Will hold all data across all hyperparam runs

    for emb_size in embed_sizes:
        for lr in lrs:
            for bs in batch_sizes:
                """ ___ ONE HYPERPARAM CONFIG ___ """

                outs_key_str = f"{emb_size}_{lr}_{bs}"  # Will key the 'outs' dict
                outs.update({outs_key_str: {}})

                print(f"Gathering data for run: {outs_key_str}")

                ######################
                #### LOADING DATA ####
                ######################

                # Load pairwise similarities from training runs
                with open(f"{eval_dir}/{set}_{emb_size}_{lr}_{bs}_pairwise_sims.pkl", "rb") as pkl_file:
                    evals = pickle.load(pkl_file)

                # Load loss curves -- normalize
                with open(f"{loss_dir}/{set}_{emb_size}_{lr}_{bs}_loss_data.pkl", "rb") as pkl_file:
                    losses = pickle.load(pkl_file)

                    loss_norm = bs / 64  # Loss curves saved in files need to be normalized by batch size
                    # Batch sizes were 64, 128, and 256 -- so normalize with 64

                    losses["train"] = np.array(losses["train"]) * loss_norm
                    losses["val"] = np.array(losses["val"]) * loss_norm
                    losses["test_loss"] = losses["test_loss"] * loss_norm

                # # Load saved embedding weights
                with open(f"{weight_dir}/card2vec_{set}_emb-{emb_size}_epochs-20_lr-{lr}_bs-{bs}.pt", "rb") as torchfile:
                    weights = torch.load(torchfile)

                ##########################
                #### END LOADING DATA ####
                ##########################

                #####################
                #### LOSS CURVES ####
                #####################

                outs[outs_key_str].update({"loss": {}})
                outs[outs_key_str]["loss"].update({"train": losses["train"]})
                outs[outs_key_str]["loss"].update({"valid": losses["val"]})
                outs[outs_key_str]["loss"].update({"test": losses["test_loss"]})

                #########################
                #### END LOSS CURVES ####
                #########################

                ###########################
                #### CALC EVAL METRICS ####
                ###########################

                outs[outs_key_str].update({"evals": {}})

                # Spin up evaluator
                c2v_eval = Card2VecEmbeddingEval(weights, set_symbol=set, card_names=card_names, device=None)  # torch device not needed
                top5s, bottom5s, mins, maxs, avgs, variances = c2v_eval.eval_run_statistics(evals)  # Summary statistics across epochs

                outs[outs_key_str]["evals"].update({"top5s": top5s})
                outs[outs_key_str]["evals"].update({"bottom5s": bottom5s})
                outs[outs_key_str]["evals"].update({"mins": mins})
                outs[outs_key_str]["evals"].update({"maxs": maxs})
                outs[outs_key_str]["evals"].update({"avgs": avgs})
                outs[outs_key_str]["evals"].update({"variances": variances})

                # Get similarity evaluation data for all bespoke card pairs for this run, across all epochs
                card_pair_evals = {"sims": [], "dissims": []}
                for epoch in evals['data']:
                    # Retrieve similarities for similar pairs of cards for this epoch
                    sim_scores_this_epoch = []
                    for pair in sim_ids:
                        try:
                            sim_scores_this_epoch.append(
                                epoch[pair][0])  # Drop the None (euclid distances were not gathered)
                        except KeyError:
                            # eval datum are indexed by tuple -- so order will matter
                            sim_scores_this_epoch.append(epoch[pair[::-1]][0])  # Reverse the index tuple on KeyError

                    # Retrieve similarities for dissimilar pairs of cards for this epoch
                    dissim_scores_this_epoch = []
                    for pair in dissim_ids:
                        try:
                            dissim_scores_this_epoch.append(
                                epoch[pair][0])  # Drop the None (euclid distances were not gathered)
                        except KeyError:
                            # eval datum are indexed by tuple -- so order will matter
                            dissim_scores_this_epoch.append(epoch[pair[::-1]][0])  # Reverse the index tuple on KeyError

                    card_pair_evals["sims"].append(sim_scores_this_epoch)  # Save extracted similar scores
                    card_pair_evals["dissims"].append(dissim_scores_this_epoch)  # Save extracted dissimilar scores

                outs[outs_key_str]["evals"].update({"card_pair_evals": card_pair_evals})

                ###############################
                #### END CALC EVAL METRICS ####
                ###############################

                """ ___ ONE HYPERPARAM CONFIG END ___ """

    return outs
