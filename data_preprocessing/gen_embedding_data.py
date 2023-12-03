"""
Datasets sourced from: https://www.17lands.com/public_datasets

Note -- when running locally, be sure to change the 'path_prefix' variable at the start of the script to point to the
local (relative) location of the saved dataset files on your file system.
"""

import pickle
import torch

from seventeen_lands_preprocessor import gen_card_names, gen_training_pairs

if __name__ == "__main__":

    path_prefix = "../../datasets"  # MODIFY

    # Large data sets
    sets = ["DMU", "MOM", "WOE"]
    deck_num = 100000  # 100 K
    sample = 0.001     # Hyperparameter that controls strength of subsampling

    for set in sets:
        print(f"Extracting training pairs for {set}")

        with open(f"{path_prefix}/game_data_public.{set}.PremierDraft.csv", "r") as csv_file:
            card_names = gen_card_names(csv_file)
            csv_file.seek(0)
            training_pairs = gen_training_pairs(csv_file, deck_num, sample)

        # Save tensor
        torch.save(training_pairs, f"datasets/training_pairs_{set}.pt")
        # Save card names
        with open(f"{path_prefix}/card_names_{set}.pkl", "wb") as pkl_file:
            pickle.dump(card_names, pkl_file)

    # Medium data sets
    sets = ["LTR", "BRO", "SNC", "ONE"]
    deck_num = 50000  # 50 K
    sample = 0.001    # Hyperparameter that controls strength of subsampling

    for set in sets:
        print(f"Extracting training pairs for {set}")

        with open(f"{path_prefix}/game_data_public.{set}.PremierDraft.csv", "r") as csv_file:
            card_names = gen_card_names(csv_file)
            csv_file.seek(0)
            training_pairs = gen_training_pairs(csv_file, deck_num, sample)

        # Save tensor
        torch.save(training_pairs, f"datasets/training_pairs_{set}.pt")
        # Save card names
        with open(f"{path_prefix}/card_names_{set}.pkl", "wb") as pkl_file:
            pickle.dump(card_names, pkl_file)

    # Small data sets
    sets = ["NEO", "SIR", "VOW"]
    deck_num = 10000  # 10K
    sample = 0.001    # Hyperparameter that controls strength of subsampling

    for set in sets:
        print(f"Extracting training pairs for {set}")

        with open(f"{path_prefix}/game_data_public.{set}.PremierDraft.csv", "r") as csv_file:
            card_names = gen_card_names(csv_file)
            csv_file.seek(0)
            training_pairs = gen_training_pairs(csv_file, deck_num, sample)

        # Save tensor
        torch.save(training_pairs, f"datasets/training_pairs_{set}.pt")
        # Save card names
        with open(f"{path_prefix}/card_names_{set}.pkl", "wb") as pkl_file:
            pickle.dump(card_names, pkl_file)

