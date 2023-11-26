from oracle_text_extractor import filter_fields
import pickle

if __name__ == "__main__":
    # Get filtered oracle card objects
    with open("../raw_datasets/oracle-cards-20231119220154.json", "r", encoding="utf8") as json_file:
        oracle_out = filter_fields(json_file)

    # Save oracle objects to file
    with open("../datasets/oracle_cards.pkl", "wb") as pkl_file:
        pickle.dump(oracle_out, pkl_file)

    # Depickle
    # with open("../datasets/oracle_cards.pkl", "rb") as pkl_file:
    #     oracle_out = pickle.load(pkl_file)
