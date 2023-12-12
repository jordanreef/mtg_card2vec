"""
Static data needed for doing some downstream evaluations / puzzles on card2vec and BERT.
"""

# region WOE t-SNE Clusters

staples_white = ["Plains", "Besotted Knight", "Cooped Up", "Hopeful Vigil", "The Princess Takes Flight", "Stockpiling Celebrant",
                 "Spellbook Vendor", "Werefox Bodyguard", "Cursed Courtier", "Kellan's Lightblades", "Archon's Glory"]

staples_blue = ["Island", "Aquatic Alchemist", "Archive Dragon", "Bitter Chill", "Horned Loch-Whale", "Ice Out",
                "Into the Fae Court", "Johann's Stopgap", "Hatching Plans", "Picklock Prankster", "Spell Stutter"]

staples_black = ["Swamp", "Barrow Naughty", "Candy Grapple", "Conceited Witch", "Feed the Cauldron", "Hopeless Nightmare",
                 "High Fae Negotiator", "Mintstrosity", "Rat Out", "Scream Puff", "Sweettooth Witch"]

staples_red = ["Mountain", "Belligerent of the Ball", "Bespoke Battlegarb", "Cut In", "Edgewall Pack", "Flick a Coin",
               "Harried Spearguard", "Monstrous Rage", "Song of Totentanz", "Redcap Thief", "Witchstalker Frenzy"]

staples_green = ["Forest", "Agatha's Champion", "Brave the Wilds", "Curse of the Werefox", "Ferocious Werefox", "Graceful Takedown",
                 "Gruff Triplets", "Hamlet Glutton", "Hollow Scavenger", "Rootrider Faun", "Royal Treatment"]

wr_aggro = ["Armory Mice", "Cheeky House-Mouse", "Hopeful Vigil", "Return Triumphant", "Stockpiling Celebrant",
            "Belligerent of the Ball", "Embereth Veteran", "Harried Spearguard", "Ash, Party Crasher", "Gingerbrute"]

ub_faeries = ["Mocking Sprite", "Snaremaster Sprite", "Spell Stutter", "Talion's Messenger", "Barrow Naughty",
              "Ego Drain", "Faerie Dreamthief", "Faerie Fencing", "Obyra, Dreaming Duelist", "Talion, the Kindly Lord"]

bg_food = ["Back for Seconds", "Candy Grapple", "Mintstrosity", "Sweettooth Witch", "Night of the Sweets' Revenge",
           "Hamlet Glutton", "Hollow Scavenger", "Tough Cookie", "Greta, Sweettooth Scourge", "Restless Cottage"]

wu_freeze = ["Frostbridge Guard", "Plunge into Winter", "Rimefur Reindeer", "Bitter Chill", "Freeze in Place",
             "Icewrought Sentry", "Succumb to the Cold", "Hylda of the Icy Crown", "Sharae of Numbing Depths", "Threadbind Clique"]

WOE_tsne_clusters = {
    'staples_white': staples_white,
    'staples_blue': staples_blue,
    'staples_black': staples_black,
    'staples_red': staples_red,
    'staples_green': staples_green,
    'wr_aggro': wr_aggro,
    'ub_faeries': ub_faeries,
    'bg_food': bg_food,
    'wu_freeze': wu_freeze
}

# endregion

# region WOE Draft Pick Puzzles

# Pick the only choice that is the correct color -- should select "Hopeful Vigil"
woe_easy_pick_1 = {
    'context': ["Besotted Knight", "Cooped Up", "The Princess Takes Flight", "Stockpiling Celebrant",
                 "Spellbook Vendor", "Werefox Bodyguard", "Cursed Courtier", "Kellan's Lightblades", "Archon's Glory"],
    'choices': ["Hopeful Vigil", "Into the Fae Court", "Barrow Naughty", "Frantic Firebolt", "Royal Treatment"],
    'solutions': ["Hopeful Vigil"]
}

# Pick the only choice that is the correct color -- should select "Hamlet Glutton"
woe_easy_pick_2 = {
    'context': ["Agatha's Champion", "Brave the Wilds", "Curse of the Werefox", "Ferocious Werefox", "Graceful Takedown",
                 "Gruff Triplets", "Hollow Scavenger", "Rootrider Faun", "Royal Treatment"],
    'choices': ["Hamlet Glutton", "Ruby, Daring Tracker", "Syr Armont, the Redeemer", "Sweettooth Witch", "Ice Out"],
    'solutions': ["Hamlet Glutton"]
}

# Pick the signpost uncommon for my color pair (WR) -- should select "Ash, Party Crasher"
woe_easy_pick_3 = {
    'context': ["Armory Mice", "Cheeky House-Mouse", "Hopeful Vigil", "Return Triumphant", "Stockpiling Celebrant",
                "Belligerent of the Ball", "Embereth Veteran", "Harried Spearguard", "Gingerbrute"],
    'choices': ["Ash, Party Crasher", "Ice Out", "Mintstrosity", "Up the Beanstalk", "Gingerbrute"],
    'solutions': ["Ash, Party Crasher"]
}

# Pick a card in the correct strategy (WR aggro) -- some choices are in the correct colors but not the right strategy
# should select either "Grand Ball Guest" or "Armory Mice"
woe_medium_pick_1 = {
    'context': ["Cheeky House-Mouse", "Hopeful Vigil", "Return Triumphant", "Stockpiling Celebrant",
                "Belligerent of the Ball", "Embereth Veteran", "Harried Spearguard", "Ash, Party Crasher", "Gingerbrute"],
    'choices': ["Grand Ball Guest", "Armory Mice",
                "Unruly Catapult", "Knight of Doves",
                "Ice Out", "Taken by Nightmares", "Hamlet Glutton", "Feral Encounter"],
    'solutions': ["Grand Ball Guest", "Armory Mice"]
}

# Pick the best card given no strong attachment to any colors / strategies -- should select "Imodane's Recruiter"
woe_medium_pick_2 = {
    'context': ["The Princess Takes Flight", "Bitter Chill", "The Witch's Vanity", "Torch the Tower", "Rootrider Faun"],
    'choices': ["Imodane's Recruiter",
                "Besotted Knight", "Spell Stutter", "Barrow Naughty", "Gnawing Crescendo", "Hamlet Glutton"],
    'solutions': ["Imodane's Recruiter"]
}

# Avoid being confounded by off colors in context, i.e. stick with your best color (Green)
# should select "Agatha's Champion"
woe_medium_pick_3 = {
    'context': ["Welcome to Sweettooth", "Hollow Scavenger", "Hamlet Glutton",
                "Torch the Tower", "Voracious Vermin"],
    'choices': ["Agatha's Champion",
                "Cut In", "Candy Grapple", "Ice Out", "Stockpiling Celebrant", "The Goose Mother", "Restless Bivouac"],
    'solutions': ["Agatha's Champion"]
}

# Pick the best card for in my strategy (WU control) given no clear color signal
# should select 'Expel the Interlopers'
woe_hard_pick_1 = {
    'context': ["Farsight Ritual", "Ice Out", "Johann's Stopgap", "Bitter Chill", "Collector's Vault"],
    'choices': ["Expel the Interlopers",
                "Armory Mice", "Tangled Colony", "Mintstrosity", "Goblin Bombardment", "Harried Spearguard", "Stormkeld Vanguard", "Leaping Ambush", "Troyan, Gutsy Explorer"],
    'solutions': ["Expel the Interlopers"]
}

# Select a crucial strategy piece, avoid good on-color staples
# should select 'Season of Growth'
woe_hard_pick_2 = {
    'context': ["Monstrous Rage", "Cut In", "Witch's Mark", "Curse of the Werefox", "Royal Treatment"],
    'choices': ["Season of Growth",
                "Torch the Tower", "Grand Ball Guest", "Tough Cookie", "Rootrider Faun", "Ruby, Daring Tracker",
                "Regal Bunnicorn", "Sleep-Cursed Faerie", "Candy Grapple"],
    'solutions': ["Season of Growth"]
}

# Pack 2 Pick 1 -- Generally tough pick, relatively large context -- strategy is (BR rat aggro)
# Choice contains some of the best cards in the set that do not share our color, and decent cards that share our color
# (but not our strategy)
# should select one of: {"Totentanz, Swarm Piper", "Torch the Tower", "Candy Grapple"}
woe_hard_pick_3 = {
    'context': ['Redcap Gutter-Dweller', 'Tangled Colony', 'Rat Out', 'Ratcatcher Trainee', "Lord Skitter's Butcher",
                "Voracious Vermin", "Edgewall Pack", "Tattered Ratter", "Harried Spearguard", "Bespoke Battlegarb",
                "Edgewall Inn", "Candy Trail", "Misleading Motes", "Slumbering Keepguard"],
    'choices': ["Totentanz, Swarm Piper", "Torch the Tower", "Candy Grapple",
                "Gruff Triplets", "Greta, Sweettooth Scourge", "Hopeful Vigil", "Archive Dragon",
                "Faerie Dreamthief", "Barrow Naughty", "Grabby Giant", "Hearth Elemental",
                "Gingerbrute", "Prophetic Prism", "Scarecrow Guide", "Crystal Grotto"],
    'solutions': ["Totentanz, Swarm Piper", "Torch the Tower", "Candy Grapple"]
}

WOE_draft_puzzles = [woe_easy_pick_1, woe_easy_pick_2, woe_easy_pick_3,
                     woe_medium_pick_1, woe_medium_pick_2, woe_medium_pick_3,
                     woe_hard_pick_1, woe_hard_pick_2, woe_hard_pick_3]

# endregion

# region LTR Draft Pick Puzzles

ltr_easy_pick_1 = {
    'context': ["Rohirrim Lancer"],
    'choices': ["Relentless Rohirrim", "Revive the Shire"],
    'solutions': ["Relentless Rohirrm"]
}

ltr_easy_pick_2 = {
    'context': ["Dunland Crebain"],
    'choices': ["Easterling Vanguard", "Hithlain Knots"],
    'solutions': ["Easterling Vanguard"]
}

ltr_easy_pick_3 = {
    'context': ["Errand-Rider of Gondor"],
    'choices': ["Flowering of the White Tree", "Orcish Medicine"],
    'solutions': ["Flowering of the White Tree"]
}

ltr_medium_pick_1 = {
    'context': ["Dunland Crebain"],
    'choices': ["Easterling Vanguard", "Ithilien Kingfisher"],
    'solutions': ["Easterling Vanguard"]
}

ltr_medium_pick_2 = {
    'context': ["Peregrine Took"],
    'choices': ["Shire Shirriff", "Mirkwood Spider"],
    'solutions': ["Shire Shirriff"]
}

ltr_medium_pick_3 = {
    'context': ["Smite the Deathless"],
    'choices': ["Ranger's Firebrand", "Stew the Coneys"],
    'solutions': ["Ranger's Firebrand"]
}

ltr_hard_pick_1 = {
    'context': ["Old Man Willow"],
    'choices': ["Entish Restoration", "Voracious Fell Beast"],
    'solutions': ["Entish Restoration"]
}

ltr_hard_pick_2 = {
    'context': ["Rohirrim Lancer"],
    'choices': ["Horn of Gondor", "Foray of Orcs"],
    'solutions': ["Horn of Gondor"]
}

LTR_draft_puzzles = [ltr_easy_pick_1, ltr_easy_pick_2, ltr_easy_pick_3,
                     ltr_medium_pick_1, ltr_medium_pick_2, ltr_medium_pick_3,
                     ltr_hard_pick_1, ltr_hard_pick_2]

# endregion