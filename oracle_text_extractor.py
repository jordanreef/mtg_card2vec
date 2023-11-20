"""
This file contains helper functions for extracting, saving, loading, etc. data sourced from the Scryfall "Oracle Cards"
bulk API object: https://scryfall.com/docs/api/bulk-data

The "Oracle Cards" object contains one entry for each individual "Oracle ID" -- basically this means that each unique
game piece in MTG appears once in the object (i.e. art variants, reprints, etc. do not appear).
"""

# Imports

import json
import csv
