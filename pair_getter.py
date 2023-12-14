""" Quick little script to get some good eval pairs from WOE """

import pickle

if __name__ == "__main__":

    with open("datasets/card_names_NEO.pkl", "rb") as pkl_file:
        card_names_neo = pickle.load(pkl_file)

    with open("datasets/card_names_WOE.pkl", "rb") as pkl_file:
        card_name_woe = pickle.load(pkl_file)

    # Pairs of cards from WOE that were heavily played together
    similar_pair_names = [
        ("Hopeful Vigil", "Stockpiling Celebrant"),
        ("The Princess Takes Flight", "Stockpiling Celebrant"),
        ("Johann's Stopgap", "Hatching Plans"),
        ("Ice Out", "Hatching Plans"),
        ("Barrow Naughty", "Faerie Dreamthief"),
        ("Faerie Fencing", "Faerie Dreamthief"),
        ("Bespoke Battlegarb", "Edgewall Pack"),
        ("Belligerent of the Ball", "Edgewall Pack"),
        ("Hamlet Glutton", "Hollow Scavenger"),
        ("Welcome to Sweettooth", "Hollow Scavenger")
    ]

    similar_pair_ids = []

    dissimilar_pair_names = [
        ("Ash, Party Crasher", "Greta, Sweettooth Scourge"),
        ("Syr Armont, the Redeemer", "Obyra, Dreaming Duelist"),
        ("Sharae of Numbing Depths", "Ruby, Daring Tracker"),
        ("Johann, Apprentice Sorcerer", "Neva, Stalked by Nightmares"),
        ("Totentanz, Swarm Piper", "Troyan, Gutsy Explorer")
    ]

    dissimilar_pair_ids = []

    for pair in similar_pair_names:
        similar_pair_ids.append((card_names[0][pair[0]], card_names[0][pair[1]]))

    for pair in dissimilar_pair_names:
        dissimilar_pair_ids.append((card_names[0][pair[0]], card_names[0][pair[1]]))

    data = {
        "similar": {"names": similar_pair_names, "ids": similar_pair_ids},
        "dissimilar": {"names": dissimilar_pair_names, "ids": dissimilar_pair_ids}
    }

    with open("mtg_card2vec/evals/WOE_eval_pairs.pkl", "wb") as pkl_file:
        pickle.dump(data, pkl_file)