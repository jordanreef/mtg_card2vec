""" Script for conducting a hyperparameter grid search on word2vec embeddings"""

import torch

from models.card2vec_embedding import train_card2vec_embedding


def card2vec_hyper_parameter_search(set_size, set, set_data, card_names, epochs, device):

    """___ HYPER-PARAMETER GRID ___ """

    embed_sizes = [200, 250, 300, 350, 400]
    lrs = [0.01, 0.001, 0.0001]
    batch_sizes = [64, 128, 256]

    """ ___________________________ """

    for emb_size in embed_sizes:
        for lr in lrs:
            for bs in batch_sizes:
                vecs = train_card2vec_embedding(set_size=set_size, embedding_dim=emb_size, set=set,
                                                training_corpus=set_data, card_labels=card_names,
                                                epochs=epochs, learning_rate=0.001, batch_size=256, device=device,
                                                evals=True, eval_dir=f"run_data/evals", eval_label=f"{set}_{emb_size}_{lr}_{bs}",
                                                plot=True, plot_dir=f"run_data/losses", plot_label=f"{set}_{emb_size}_{lr}_{bs}")

                torch.save(vecs, f"trained/card2vec_{set}_emb-{emb_size}_epochs-{epochs}_lr-{lr}_bs-{bs}.pt")
