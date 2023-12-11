"""
Various small utilities to make training, experimenting, validating, etc. a bit easier
"""


def get_card_colors(oracle_data, card_names):
    """
    Generates a dictionary of {card_name: [card_colors]} from the Scryfall oracle data. 17Lands and Scryfall use
    slightly different naming conventions for their cards so this function ensures a smooth translation

    Arguments:
        oracle_data (list) : Big list of oracle card data
        card_names (tuple) : List of card_names tuple as returned by the 17Lands preprocessor.
    """
    oracle_data = [c for c in oracle_data if c["layout"] != 'art_series']  # Remove problematic scryfall objects

    card_colors = {}
    for card_name in card_names[0]:
        found = False
        for c in oracle_data:
            if card_name == c["name"]:
                card_colors.update({card_name: c["colors"]})
                found = True
                break

        if not found:
            for c in oracle_data:
                if card_name in c["name"]:
                    card_colors.update({card_name: c["colors"]})
                    break

    return card_colors


def color_label_mapping(card_colors):
    """
    Small utility that converts a list of card_colors (as in oracle_data) to an integer label number (to
    be used in plot labelling)
    """
    white = 0
    blue = 1
    black = 2
    red = 3
    green = 4
    colorless = 5
    multicolor = 6
    default = -1

    if len(card_colors) > 1:
        return multicolor
    elif len(card_colors) == 0:
        return colorless
    elif card_colors[0] == 'W':
        return white
    elif card_colors[0] == 'U':
        return blue
    elif card_colors[0] == 'B':
        return black
    elif card_colors[0] == 'R':
        return red
    elif card_colors[0] == 'G':
        return green

    return default
