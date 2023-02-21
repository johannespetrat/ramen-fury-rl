from enum import Enum
import copy
from dataclasses import dataclass
import numpy as np

class Type(Enum):
    no_card = 0
    protein = 1
    vegetarian = 2
    both = 3
    nori = 4
    chili = 5
    fury = 6
    soy_sauce = 7
    beef = 8
    chicken = 9
    shrimp = 10



@dataclass
class Card:
    type: Type
    name: str
    
    @property
    def value(self):
        return self.type.value

    def __eq__(self, other):
        return self.name == other.name

    def is_broth(self):
        return self.value in [6, 7, 8, 9, 10]

    def is_protein(self):
        return self.value in [1, 3]
    
    def is_vegetarian(self):
        return self.value in [2, 3]

    def is_nori(self):
        return self.value == 4

    def is_chili(self):
        return self.value == 5


class Deck:
    def __init__(self):
        self.deck = [Card(Type.protein, "Naruto") for _ in range(4)] + \
                    [Card(Type.protein, "Chashu") for _ in range(4)] + \
                    [Card(Type.protein, "Eggs") for _ in range(4)] + \
                    [Card(Type.both, "Tofu") for _ in range(4)] + \
                    [Card(Type.vegetarian, "Scallions") for _ in range(4)] + \
                    [Card(Type.vegetarian, "Mushrooms") for _ in range(4)] + \
                    [Card(Type.vegetarian, "Corn") for _ in range(4)] + \
                    [Card(Type.nori, "Nori") for _ in range(8)] + \
                    [Card(Type.chili, "Chili") for _ in range(12)] + \
                    [Card(Type.fury, "Fury") for _ in range(4)] + \
                    [Card(Type.soy_sauce, "Soy Sauce") for _ in range(4)] + \
                    [Card(Type.beef, "Beef") for _ in range(4)] + \
                    [Card(Type.chicken, "Chicken") for _ in range(4)] + \
                    [Card(Type.shrimp, "Shrimp") for _ in range(4)]
        np.random.shuffle(self.deck)
        self.discard_pile = []
    
    def draw_card(self, n=1):
        cards = []
        if n > len(self.deck):  # draw cards until the deck runs out and then reuse the discard pile
            diff = n - len(self.deck)
            cards = [self.deck.pop(0) for _ in range(len(self.deck))]
            np.random.shuffle(self.discard_pile)
            self.deck = copy.deepcopy(self.discard_pile)
            self.discard_pile = []

        diff = n
        while diff > 0:
            if self.deck:
                cards.append(self.deck.pop(0))
            diff -= 1

        return cards

    def add_to_discard_pile(self, cards):
        self.discard_pile += cards

    def __len__(self):
        return len(self.deck)