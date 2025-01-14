import gymnasium as gym
import numpy as np
import random
from itertools import permutations
from gymnasium.spaces import Discrete, Tuple

colors = ['Hearts', 'Diamonds', 'Spades', 'Clubs']
face_values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J', 'Q', 'K']
values = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}


def calculateCardValues(cards):
    total_card_value = 0

    for card in cards:
        total_card_value += values[card[1]]

    return total_card_value


def swapCards(cards, selected_card):
    lowest = 0
    index = 0
    card_swapped = False

    if len(cards) < 4:
        cards.append(selected_card)
        return cards, card_swapped

    lowest = values[cards[0][1]]
    i = 0
    for i in range(len(cards)):
        card = cards[i]
        if values[card[1]] < lowest:
            lowest = values[card[1]]
            index = i

    if values[selected_card[1]] > lowest:
        cards[index] = selected_card
        card_swapped = True
    return cards, card_swapped


class AccumulateHighestCardsEnv(gym.Env):
    def __init__(self, has_infinite_deck=False):
        super(AccumulateHighestCardsEnv, self).__init__()
        self.deck = []
        self.discardedCards = []
        self.playerCards = []
        self.shownCard = None
        self.has_infinite_deck = has_infinite_deck

        self.initDeck()  # creating a deck of cards and shuffling it

        # [total_value_of_player_cards = 52 for 4 cards, card_on_top_of_deck]
        self.observation_space = Tuple((Discrete(52), Discrete(13)))
        self.action_space = Discrete(2)  # 3 actions: PICK and DISCARD

    def step(self, action):
        if action:  # PICK action

            previous_player_card_value = calculateCardValues(self.playerCards)
            self.playerCards, card_swapped = swapCards(self.playerCards, self.shownCard)
            updated_player_card_value = calculateCardValues(self.playerCards)

            reward = updated_player_card_value - previous_player_card_value
            terminated = True
            if len(self.deck) > 0:
                terminated = False
                self.shownCard = self.deck.pop(0)  # showing the topmost card of the deck

            next_state = tuple([updated_player_card_value, values[self.shownCard[1]]])
            return next_state, reward, terminated, {}, {"playerCards": self.playerCards}

        else:  # DISCARD action
            terminated = True
            valueOfShownCard = 0

            previous_player_card_value = calculateCardValues(self.playerCards)
            player_cards, card_swapped = swapCards(self.playerCards, self.shownCard)
            updated_player_card_value = calculateCardValues(player_cards)

            reward = previous_player_card_value - updated_player_card_value
            # reward = -valueOfShownCard

            self.discardedCards.append(self.shownCard)
            if len(self.deck) > 0:
                terminated = False
                self.shownCard = self.deck.pop(0)  # showing the topmost card of the deck
                valueOfShownCard = values[self.shownCard[1]]

            next_state = tuple([updated_player_card_value, valueOfShownCard])

        return next_state, reward, terminated, {}, {"playerCards": self.playerCards}


    def reset(self):
        self.discardedCards = []
        self.playerCards = []
        self.initDeck()

        self.shownCard = self.deck.pop(0)

        return [tuple([0, values[self.shownCard[1]]])]

    def shuffleDeck(self):
        self.deck = random.sample(self.deck, k=len(self.deck))

    def render(self, mode='human'):
        result = 1
        # print('render')

    def initDeck(self):
        self.deck = [tuple([color, value]) for value in face_values for color in colors]
        self.shuffleDeck()
