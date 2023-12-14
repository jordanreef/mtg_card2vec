"""
Staging ground for running other stuff!

NOTES TO SELF:

3 Candidate Sets:
    WOE -- Large
    LTR -- Medium
    NEO -- Small
"""
import matplotlib.cbook
import torch
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

from random import shuffle

from evals.card2vec_downstream_tasks import Card2VecEmbeddingEval

from data_preprocessing.seventeen_lands_preprocessor import gen_card_names, gen_training_pairs, scrape_gih_wr

from models.card2vec_embedding import Card2VecFFNN, train_card2vec_embedding

from card2vec_hyperparameter_search import card2vec_hyper_parameter_search, gather_hyperparam_run_metrics

from utils import get_card_colors, color_label_mapping

NAME_2_ID = 0
ID_2_NAME = 1

if __name__ == "__main__":

    path_prefix = "datasets"
    set = "WOE"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Card names
    with open(f"datasets/card_names_{set}.pkl", "rb") as pkl_file:
        card_names = pickle.load(pkl_file)

    # Oracle data
    with open(f"datasets/oracle_cards.pkl", "rb") as pkl_file:
        oracle_cards = pickle.load(pkl_file)

    # Trained c2v embedding
    with open(f"trained/new_card2vec_WOE_emb-300_epochs-20_lr-0.001_bs-128.pt", "rb") as torch_file:
        woe_embedding = torch.load(torch_file)

    # Load winrates
    with open(f"datasets/WOE_winrates.pkl", "rb") as pkl_file:
        woe_winrates = pickle.load(pkl_file)

    # region Evals

    # all_run_out = {"data": None, "info": {}}
    #
    # with open(f"datasets/card_names_{set}.pkl", "rb") as pkl_file:
    #     card_names = pickle.load(pkl_file)
    #
    # all_run_out["info"].update({"card_names": card_names})
    #
    # # Search over saved hyperparameter search eval data
    # # WOE hyperparams
    # embed_sizes = [300]
    # lrs = [0.001]
    # batch_sizes = [128]
    #
    # # LTR hyperparams
    # # embed_sizes = [300, 400]
    # # lrs = [0.01, 0.001, 0.0001]
    # # batch_sizes = [64, 128, 256]
    #
    # all_run_out["info"].update({"embed_sizes": embed_sizes})
    # all_run_out["info"].update({"lrs": lrs})
    # all_run_out["info"].update({"batch_sizes": batch_sizes})
    #
    # # # Similar LTR pairings
    # # ltr_sims = [
    # #     ("Relentless Rohirrim", "Rally at the Hornburg"),            # Same Color, Same Strategy (red aggro)
    # #     ("Relentless Rohirrim", "Rohirrim Lancer"),                  # Same Color, Same Strategy (red aggro)
    # #     ("Dunland Crebain", "The Torment of Gollum"),                # Same Color, Same Strategy (black amass)
    # #     ('Dunland Crebain', "Easterling Vanguard"),                  # Same Color, Same Strategy (black amass)
    # #     ("Arwen's Gift", "Stern Scolding"),                          # Same Color, Same Startegy (blue control)
    # #     ("Gandalf, Friend of the Shire", "Bilbo, Retired Burglar"),  # Different Color, Same Stategy (UR tempt)
    # #     ("Foray of Orcs", "Dunland Crebain"),                        # Different Color, Same Strategy (BR amass)
    # #     ("Rosie Cotton of South Lane", "Peregrin Took"),             # Different Color, Same Strategy (WG food)
    # #     ('LothlÃ³rien Lookout', 'Arwen UndÃ³miel'),                  # Different Color, Same Strategy (UG scry)
    # #     ('ThÃ©oden, King of Rohan', "Rohirrim Lancer"),              # Different Color, Same Strategy (WR humans)
    # #     ("Flowering of the White Tree", "Plains"),                   # Sanity check (heavy white requirement)
    # #     ("Saruman's Trickery", "Island"),                            # Sanity check (heavy blue requirement)
    # #     ("Claim the Precious", "Swamp"),                             # Sanity check (heavy black requirement)
    # #     ("Moria Marauder", "Mountain"),                              # Sanity check (heavy red requirement)
    # #     ("Radagast the Brown", "Forest")                             # Sanity check (heavy green requirement)
    # # ]
    # #
    # # # 10 dissimilar LTR pairings
    # # ltr_dissims = [
    # #     ("Rohirrim Lancer", "Arwen's Gift"),                  # Different Color, Different Strategy
    # #     ("The Torment of Gollum", "Elven Farsight"),          # Different Color, Different Strategy
    # #     ("The Mouth of Sauron", "Frodo Baggins"),             # Different Color, Different Strategy
    # #     ("Old Man Willow", "Bilbo, Retired Burglar"),         # Different Color, Different Strategy
    # #     ("Elrond, Master of Healing", "Lash of the Balrog"),  # Different Color, Different Strategy
    # #     ("Birthday Escape", "Nimrodel Watcher"),              # Same color, Different strategy
    # #     ("Fiery Inscription", "Rohirrim Lancer"),             # Same color, Different strategy
    # #     ("Smite the Deathless", "Mirkwood Bats"),             # Generically good + other
    # #     ("Orcish Bowmasters", "Stew the Coneys"),             # Generically good + other
    # #     ("Horn of Gondor", "Mordor Trebuchet"),               # Generically good + other
    # #     ("Flowering of the White Tree", "Forest"),            # Sanity check (heavy white requirement)
    # #     ("Saruman's Trickery", "Plains"),                     # Sanity check (heavy blue requirement)
    # #     ("Claim the Precious", "Forest"),                     # Sanity check (heavy black requirement)
    # #     ("Moria Marauder", "Plains"),                         # Sanity check (heavy red requirement)
    # #     ("Radagast the Brown", "Swamp")                       # Sanity check (heavy green requirement)
    # # ]
    #
    # woe_sims = [
    #     ("Hopeful Vigil", "Stockpiling Celebrant"),           # White bounce combo
    #     ("Ice Out", "Johann's Stopgap"),                      # Blue control magic
    #     ("Hopeless Nightmare", "Candy Grapple"),              # Black bargain
    #     ("Belligerent of the Ball", "Grand Ball Guest"),      # Red celebration
    #     ("Welcome to Sweettooth", "Tough Cookie"),            # Green food
    #     ("Ash, Party Crasher", "Imodane's Recruiter"),        # RW aggro
    #     ("Greta, Sweettooth Scourge", "Gingerbread Hunter"),  # BG food
    #     ("Feral Encounter", "Forest"),                        # Sanity check -- heavy green
    #     ("Goddric, Cloaked Reveler", "Mountain"),             # Sanity check -- heavy red
    #     ("Ice Out", "Island")                                 # Sanity check -- heavy blue
    # ]
    #
    # woe_dissims = [
    #     ("Ash, Party Crasher", "Obyra, Dreaming Duelist"),             # RW vs UB
    #     ("Greta, Sweettooth Scourge", "Johann, Apprentice Sorcerer"),  # BG vs UR
    #     ("Neva, Stalked by Nightmares", "Ruby, Daring Tracker"),       # WB vs RG
    #     ("Syr Armont, the Redeemer", "Totentanz, Swarm Piper"),        # WG vs BR
    #     ("Harried Spearguard", "Sharae of Numbing Depths"),            # Red aggro vs WU control
    #     ("Spell Stutter", "Armory Mice"),                              # Blue control vs white aggro
    #     ("Virtue of Knowledge", "Grand Ball Guest"),                   # Blue unplayable vs red aggro
    #     ("Feral Encounter", "Island"),                                 # Sanity check -- heavy green vs other basic
    #     ("Goddric, Cloaked Reveler", "Forest"),                        # Sanity check -- heavy red vs other basic
    #     ("Ice Out", "Mountain"),                                       # Sanity check -- heavy blue vs other basic
    # ]
    #
    # all_run_out["info"].update({"sims": woe_sims})
    # all_run_out["info"].update({"dissims": woe_dissims})
    #
    # all_run_out_data = gather_hyperparam_run_metrics(set, card_names, "run_data/evals", "run_data/losses", "trained",
    #                                                  woe_sims, woe_dissims,
    #                                                  embed_sizes, lrs, batch_sizes)
    #
    # all_run_out["data"] = all_run_out_data
    #
    # with open(f"{set}_all_run_out.pkl", "wb") as pkl_file:
    #     pickle.dump(all_run_out, pkl_file)
    #
    # print("Break")

    # endregion

    # region Data Collection

    # print(f"Extracting training pairs for {set}")
    #
    # deck_num = 100000  # 100 K
    # sample = 0.001     # Hyperparameter that controls strength of subsampling
    #
    # with open(f"{path_prefix}/game_data_public.{set}.PremierDraft.csv", "r") as csv_file:
    #     card_names = gen_card_names(csv_file)
    #     csv_file.seek(0)
    #     training_pairs = gen_training_pairs(csv_file, deck_num, sample)
    #
    # # Save tensor
    # torch.save(training_pairs, f"datasets/training_pairs_{set}_test.pt")

    # Save card names
    # with open(f"{path_prefix}/card_names_{set}.pkl", "wb") as pkl_file:
    #     pickle.dump(card_names, pkl_file)

    # gih_wrs = scrape_gih_wr("https://www.17lands.com/card_data?expansion=WOE&format=PremierDraft&start=2023-09-05")

    # with open("datasets/WOE_winrates.pkl", "wb") as pkl_file:
    #     pickle.dump(gih_wrs, pkl_file)

    # with open("datasets/WOE_winrates.pkl", "rb") as pkl_file:
    #     gih_wrs = pickle.load(pkl_file)

    print("Break")

    # endregion

    # region Training Tests
    #
    # # HYPERPARAMETERS
    #
    # emb_size = 300
    # epochs = 20
    # lr = 0.001
    # bs = 128
    #
    # # Load some data
    # set_data = torch.load(f"datasets/training_pairs_{set}_test.pt")
    #
    # with open(f"datasets/card_names_{set}.pkl", "rb") as pkl_file:
    #     card_names = pickle.load(pkl_file)
    #
    # N, _ = set_data.shape   # N = number of training pairs
    # D = len(card_names[0])  # D = number of different cards in the set
    #
    # print(f"Starting training on {set}.")
    #
    # vecs = train_card2vec_embedding(set_size=D, embedding_dim=emb_size, set=set,
    #                                 training_corpus=set_data, card_labels=card_names,
    #                                 epochs=epochs, learning_rate=lr, batch_size=bs, device=device,
    #                                 evals=True, eval_dir=f"run_data/evals", eval_label=f"{set}_test",
    #                                 plot=True, plot_dir=f"run_data/losses", plot_label=f"{set}_test")
    #
    # torch.save(vecs, f"trained/card2vec_{set}_emb-{emb_size}_epochs-{epochs}_lr-{lr}_bs-{bs}.pt")

    # endregion

    # region Hyperparameter Search

    # Load data
    # set_data = torch.load(f"datasets/training_pairs_{set}.pt")
    # with open(f"datasets/card_names_{set}.pkl", "rb") as pkl_file:
    #     card_names = pickle.load(pkl_file)
    #
    # N, _ = set_data.shape   # N = number of training pairs
    # D = len(card_names[0])  # D = number of different cards in the set
    #
    # card2vec_hyper_parameter_search(D, set, set_data, card_names, 20, device)

    # endregion

    # region Hyperparameter Investigation

    # with open(f"{set}_all_run_out.pkl", "rb") as pkl_file:
    #     all_run_out = pickle.load(pkl_file)
    #     data = all_run_out['data']
    #     info = all_run_out['info']
    #
    # min_test_loss = math.inf
    # min_test_loss_key = None
    #
    # highest_sim_mean = -math.inf
    # highest_sim_mean_key = None
    # dissim_mean_of_highest = None
    #
    # lowest_dissim_mean = math.inf
    # lowest_dissim_mean_key = None
    # sim_mean_of_lowest = None
    #
    # greatest_mean_diff = -math.inf
    # greatest_mean_diff_key = None
    # sim_mean_greatest_diff = None
    # dissim_mean_greatest_diff = None
    #
    # for run, run_data in data.items():
    #
    #     if run_data['loss']['test'] < min_test_loss:
    #         min_test_loss = run_data['loss']['test']
    #         min_test_loss_key = run
    #
    #     sim_evals_final = np.array(run_data['evals']['card_pair_evals']['sims'][-1])
    #     dissim_evals_final = np.array(run_data['evals']['card_pair_evals']['dissims'][-1])
    #
    #     sim_mean = np.mean(sim_evals_final)
    #     dissim_mean = np.mean(dissim_evals_final)
    #
    #     if sim_mean > highest_sim_mean:
    #         highest_sim_mean = sim_mean
    #         highest_sim_mean_key = run
    #         dissim_mean_of_highest = dissim_mean
    #
    #     if dissim_mean < lowest_dissim_mean:
    #         lowest_dissim_mean = dissim_mean
    #         lowest_dissim_mean_key = run
    #         sim_mean_of_lowest = sim_mean
    #
    #     if (sim_mean - dissim_mean) > greatest_mean_diff:
    #         greatest_mean_diff = sim_mean - dissim_mean
    #         greatest_mean_diff_key = run
    #         sim_mean_greatest_diff = sim_mean
    #         dissim_mean_greatest_diff = dissim_mean
    #
    # print("Break")
    #
    #
    # c2v_eval = Card2VecEmbeddingEval(woe_embedding, "WOE", card_names, device)
    #
    # # Generate card_colors for TSNE labelling
    # # card_colors = get_card_colors(oracle_cards, card_names)
    # # with open("datasets/card_colors_WOE.pkl", "wb") as pkl_file:
    # #     pickle.dump(card_colors, pkl_file)
    #
    # # # Load card_colors for TSNE labelling
    # with open("datasets/card_colors_WOE.pkl", "rb") as pkl_file:
    #     card_colors = pickle.load(pkl_file)
    #
    # # region t-SNE
    #
    # # # WUBRG, multicolor, colorless
    # cluster_names = {0: "White", 1: "Blue", 2: "Black", 3: "Red", 4: "Green", 5: "Colorless", 6: "Multicolor", 7: "Other"}
    # cluster_colors = {0: 'white', 1: 'royalblue', 2: 'black', 3: 'indianred', 4: 'forestgreen', 5: 'tan', 6: 'goldenrod', 7: 'darkgray'}
    #
    # with open("color_mapping.pkl", "wb") as pkl_file:
    #     pickle.dump(cluster_colors, pkl_file)
    #
    # with open("cluster_names.pkl", "wb") as pkl_file:
    #     pickle.dump(cluster_names, pkl_file)
    # #
    # # # # TSNE PARAMETER SEARCH
    # # # perps = [10.0, 15.0, 20.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
    # # # lrs = ['auto', 10.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0]
    # # #
    # # # for perp in perps:
    # # #     for lr in lrs:
    # # #         plot_label = f"p_{perp}___lr_{lr}"
    # # #         c2v_eval.generate_tsne(color_labels, cluster_names, cluster_colors, plot_label,
    # # #                                perplexity=perp, learning_rate=lr,
    # # #                                save=True, save_dir="artifacts/tsne_tests")
    # #
    # # # All-color t-SNE
    # color_labels = []
    # for k, v in card_names[0].items():
    #     label = color_label_mapping(card_colors[k])
    #     color_labels.append(label)
    #
    # c2v_eval.generate_tsne(color_labels, cluster_names, cluster_colors, "Wilds of Eldraine",
    #                        perplexity=15.0, learning_rate='auto',
    #                        save=True, save_dir="artifacts/tsne_experiments")
    #
    # print("break")
    #
    # def gen_color_labels(eval_subset):
    #     # Generate cluster labels based on card colors
    #     color_labels = []
    #     for k, v in card_names[0].items():
    #         label = color_label_mapping(card_colors[k])
    #         color_labels.append(label)
    #
    #     for card in card_names[0]:
    #         if card not in eval_subset:
    #             color_labels[card_names[0][card]] = 7  # Set cluster label to 'Other'
    #
    #     return color_labels
    #
    # from evals.eval_puzzles import WOE_tsne_clusters
    #
    # for cluster in WOE_tsne_clusters.items():
    #     # print("Break")
    #     color_labels = gen_color_labels(cluster[1])
    #     c2v_eval.generate_tsne(color_labels, cluster_names, cluster_colors, cluster[0],
    #                            perplexity=15.0, learning_rate='auto',
    #                            save=True, save_dir="artifacts/tsne_experiments")
    #
    # # INVESTIGATION NOTES -- WOE
    # # 300_0.001_128 yielded the highest similar mean and the lowest test loss
    # # 400_0.0001_128 yielded the lowest dissimilar mean
    # # 300_0.001_128 yielded the highest difference
    #
    # # INVESTIGATION NOTES -- LTR
    # # 400_0.001_64 yielded highest similar mean
    # # 300_0.0001_128 yieled the lowest dissimilar mean (near zero)
    # # 300_0.0001_128 yielded highest difference -- though still not especially high

    # endregion

    # region Draft Selections

    c2v_eval = Card2VecEmbeddingEval(woe_embedding, "WOE", card_names, device)

    from evals.eval_puzzles import WOE_draft_puzzles

    # Brief search over winrate weighting ranges
    lower_bounds = [0.3, 0.4, 0.5, 0.6]
    upper_bounds = [0.8, 0.9, 1.0, 1.1]
    eval_types = ['additive', 'multiplicative', 'no_embed', 'no_winrate']
    ks = [2, 3, 4, 5]

    results = {}
    for lb in lower_bounds:
        for ub in upper_bounds:
            for et in eval_types:
                for k in ks:
                    correct = 0  # Number of picks selected correctly
                    puzzles_right = []
                    for c, puzzle in enumerate(WOE_draft_puzzles):
                        context, choices, slns = (puzzle['context'], puzzle['choices'], puzzle['solutions'])
                        shuffle(choices)

                        # Make a draft pick
                        pick = c2v_eval.draft_pick(context, choices, woe_winrates, lb, ub,
                                                   et, cluster_algo='k_means', k=k)

                        # pick = c2v_eval.draft_pick(context, choices, woe_winrates, lb, ub, et)

                        pick_name = card_names[ID_2_NAME][pick]

                        # Check if 'correct'
                        if pick_name in slns:
                            correct += 1
                            puzzles_right.append(c)

                    results.update(
                        {(lb, ub, et, k): [float(correct / len(WOE_draft_puzzles)), puzzles_right]}
                    )

    avg_add = np.mean([results[trial][0] for trial in results if trial[2] == "additive"])
    max_add = np.max([results[trial][0] for trial in results if trial[2] == "additive"])

    avg_mul = np.mean([results[trial][0] for trial in results if trial[2] == "multiplicative"])
    max_mul = np.max([results[trial][0] for trial in results if trial[2] == "multiplicative"])

    avg_no_emb = np.mean([results[trial][0] for trial in results if trial[2] == "no_embed"])
    max_no_emb = np.max([results[trial][0] for trial in results if trial[2] == "no_embed"])

    avg_no_wrs = np.mean([results[trial][0] for trial in results if trial[2] == "no_winrate"])
    max_no_wrs = np.max([results[trial][0] for trial in results if trial[2] == "no_winrate"])

    print("Break")

    # endregion

    # region Plotting

    # def plot_sims(sims, dissims, sim_scale, set_name):
    #
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #
    #     ax.set_xlabel('Epochs', fontweight="bold")
    #     ax.set_ylabel('Similarity', fontweight="bold")
    #
    #     ax.set_ylim(sim_scale[0], sim_scale[1])  # Y-scale to use for similarity score
    #
    #     ax.plot(sims, color="darkblue", label="Similar Pairs")
    #     ax.plot(dissims, color="deepskyblue", label="Dissimilar Pairs", linestyle="dashed")
    #
    #     plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    #
    #     ax.legend(loc="upper left")
    #     plt.title(f"'{set_name}' card2vec Similarity Evaluations", fontweight="bold")
    #
    #     # plt.show()
    #     plt.savefig(f"artifacts/{set_name}_card2vec_sim_dissim_pairs.png")
    #
    # def plot_loss_vs_simdif(loss, sim_diffs, sim_scale, set_name):
    #     fig, ax1 = plt.subplots(figsize=(8, 6))
    #
    #     ax1.set_xlabel('Epochs', fontweight="bold")
    #     ax1.set_ylabel('Avg Validation Loss', color="red", fontweight="bold")
    #     ax1.plot(loss, color="red", label='Validation Loss')
    #     ax1.tick_params(axis='y', labelcolor="red")
    #
    #     ax2 = ax1.twinx()
    #
    #     ax2.set_ylabel('Similarity Difference', color="blue", fontweight="bold")
    #
    #     ax2.set_ylim(sim_scale[0], sim_scale[1])  # Y-scale to use for similarity score
    #
    #     ax2.plot(sim_diffs, color="blue", label="Similarity Difference")
    #     ax2.tick_params(axis='y', labelcolor="blue")
    #
    #     plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    #
    #     fig.legend(loc="upper left", bbox_to_anchor=(0.14, 0.875))
    #     plt.title(f"'{set_name}' Loss/Similarity Comparison", fontweight="bold")
    #
    #     # plt.show()
    #     plt.savefig(f"artifacts/{set_name}_card2vec_loss_and_sim.png")
    #
    # def plot_simdiff_comparison(set_a, set_b, sim_scale):
    #     """ Pass tuples of (data, name) for set_a / set_b """
    #     set_a_data, set_a_label = set_a
    #     set_b_data, set_b_label = set_b
    #
    #     fi, ax = plt.subplots(figsize=(8, 6))
    #
    #     ax.set_xlabel('Epochs', fontweight="bold")
    #     ax.set_ylabel('Similarity Difference', fontweight="bold")
    #
    #     ax.set_ylim(sim_scale[0], sim_scale[1])
    #
    #     plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    #
    #     ax.plot(set_a_data, color="blue", label=set_a_label)
    #     ax.plot(set_b_data, color="red", label=set_b_label)
    #
    #     ax.legend(loc="upper left")
    #     plt.title(f"{set_a_label} and {set_b_label} Similarity Comparison", fontweight="bold")
    #
    #     plt.savefig(f"artifacts/{set_a_label}_{set_b_label}_sim_comparison.png")
    #
    # sim_y_scale = (-0.05, 0.5)
    # sim_loss_y_scale = (0.0, 0.2)
    #
    # # (New) WOE Loss vs Similarity Curves
    #
    # with open("new_WOE_all_run_out.pkl", "rb") as pkl_file:
    #     woe_run_data = pickle.load(pkl_file)
    #
    # good_woe_run = woe_run_data['data']['300_0.001_128']
    #
    # new_woe_train_loss = good_woe_run['loss']['train']  # (20,)
    # new_woe_valid_loss = good_woe_run['loss']['valid']  # (20,)
    #
    # new_woe_sim_scores = np.array([np.mean(sim_list) for sim_list in good_woe_run['evals']['card_pair_evals']['sims']])  # (20,)
    # new_woe_dissim_scores = np.array([np.mean(sim_list) for sim_list in good_woe_run['evals']['card_pair_evals']['dissims']])
    # new_woe_sim_dissim_diffs = new_woe_sim_scores - new_woe_dissim_scores
    #
    # plot_sims(new_woe_sim_scores, new_woe_dissim_scores, sim_y_scale, "Wilds of Eldraine")
    # plot_loss_vs_simdif(new_woe_valid_loss, new_woe_sim_dissim_diffs, sim_loss_y_scale, "Wilds of Eldraine")
    #
    # # WOE Loss vs Similarity Curves
    #
    # with open("WOE_all_run_out.pkl", "rb") as pkl_file:
    #     woe_run_data = pickle.load(pkl_file)
    #
    # good_woe_run = woe_run_data['data']['300_0.001_128']
    #
    # woe_train_loss = good_woe_run['loss']['train']  # (20,)
    # woe_valid_loss = good_woe_run['loss']['valid']  # (20,)
    #
    # woe_sim_scores = np.array([np.mean(sim_list) for sim_list in good_woe_run['evals']['card_pair_evals']['sims']])  # (20,)
    # woe_dissim_scores = np.array([np.mean(sim_list) for sim_list in good_woe_run['evals']['card_pair_evals']['dissims']])
    # woe_sim_dissim_diffs = woe_sim_scores - woe_dissim_scores
    #
    # plot_sims(woe_sim_scores, woe_dissim_scores, sim_y_scale, "Wilds of Eldraine (small)")
    # plot_loss_vs_simdif(woe_valid_loss, woe_sim_dissim_diffs, sim_loss_y_scale, "Wilds of Eldraine (small)")
    #
    # # LTR Loss vs Similarity Curves
    #
    # with open("LTR_all_run_out.pkl", "rb") as pkl_file:
    #     ltr_run_data = pickle.load(pkl_file)
    #
    # good_ltr_run = ltr_run_data['data']['300_0.0001_128']
    #
    # ltr_train_loss = good_ltr_run['loss']['train']  # (20,)
    # ltr_valid_loss = good_ltr_run['loss']['valid']  # (20,)
    #
    # ltr_sim_scores = np.array([np.mean(sim_list) for sim_list in good_ltr_run['evals']['card_pair_evals']['sims']])  # (20,)
    # ltr_dissim_scores = np.array([np.mean(sim_list) for sim_list in good_ltr_run['evals']['card_pair_evals']['dissims']])
    # ltr_sim_dissim_diffs = ltr_sim_scores - ltr_dissim_scores
    #
    # plot_sims(ltr_sim_scores, ltr_dissim_scores, sim_y_scale, "Lord of the Rings")
    # plot_loss_vs_simdif(ltr_valid_loss, ltr_sim_dissim_diffs, sim_loss_y_scale, "Lord of the Rings")
    #
    # plot_simdiff_comparison((new_woe_sim_dissim_diffs, "WOE"), (ltr_sim_dissim_diffs, "LTR"), sim_loss_y_scale)
    # plot_simdiff_comparison((new_woe_sim_dissim_diffs, "WOE (large)"), (woe_sim_dissim_diffs, "WOE (small)"), sim_loss_y_scale)

    # endregion

