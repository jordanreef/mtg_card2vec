"""
This file contains helper functions for extracting, saving, loading, etc. data sourced from the Scryfall "Oracle Cards"
bulk API object: https://scryfall.com/docs/api/bulk-data

The "Oracle Cards" object contains one entry for each individual "Oracle ID" -- basically this means that each unique
game piece in MTG appears once in the object (i.e. art variants, reprints, etc. do not appear).

Since individual cards can be printed across several sets, the caller should generally select from this data based on an
explicit list of card names. I decided not to include the set of each card object, as filtering based on this can lead
to some unintuitive results (cards only appear in one set -- the set they were most recently printed in).
"""

# Imports
import json


def filter_fields(oracle_file):
    """
    Filter uneccessary fields from the raw Oracle text, and massage the outputs slightly for certain card layouts (MDFCs,
    split cards, etc.)

    Args:
        input (list<dict>) : Raw output from the Bulk API is a list of dict
    Returns:
        output (list<dict>) : Filtered list of card texts
    """
    keys = [
        "name", "mana_cost", "colors", "cmc", "type_line", "oracle_text", "power", "toughness", "loyalty", "defense",
        "layout", "card_faces"
    ]

    all_card_js = json.load(oracle_file)

    out = []
    for card in all_card_js:
        _card = {k: card.get(k, None) for k in keys}

        # Multi-faced cards (i.e. cards with split, flip, transform layout) need some further preprocessing
        if _card["card_faces"] is not None:
            _card["card_faces"] = [{k: face.get(k, None) for k in keys} for face in _card["card_faces"]]

        out.append(_card)

    return out
