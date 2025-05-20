import gymnasium as gym
import numpy as np
import random
from itertools import permutations
from gymnasium.spaces import Discrete, Tuple

colors = ['Hearts', 'Diamonds', 'Spades', 'Clubs']
face_values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J', 'Q', 'K']
values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


def findHighestAndLowestCardValues(cards, shownCard):
    if not cards:  # Handle empty list of cards
        return 0, -1, 0, -1, -1  # Return default values and invalid indices

    index_lowest = 0
    index_highest = 0
    index_equal = 0
    lowest = values[cards[0][1]]
    highest = values[cards[0][1]]
    shownCardValue = shownCard if type(shownCard) == int else values[shownCard[1]]

    for i in range(len(cards)):
        card = cards[i]
        if values[card[1]] < lowest:
            lowest = values[card[1]]
            index_lowest = i

        if values[card[1]] > highest:
            highest = values[card[1]]
            index_highest = i

        if values[card[1]] == shownCardValue:
            index_equal = i

    return highest, index_highest, lowest, index_lowest, index_equal


def simulateWAR(cards_on_stake, dealer_cards, player_cards, max_war_depth=10, current_depth=0):
    if current_depth > max_war_depth or len(dealer_cards) < 3 or len(player_cards) < 3:
        # Handle edge case: insufficient cards or too many recursive calls
        if len(dealer_cards) < 3:
            player_cards.extend(dealer_cards)
            player_cards.extend(cards_on_stake)
            dealer_cards = []
        elif len(player_cards) < 3:
            dealer_cards.extend(player_cards)
            dealer_cards.extend(cards_on_stake)
            player_cards = []
        return 0, player_cards, dealer_cards

    dealer_card = dealer_cards.pop(np.random.randint(0, len(dealer_cards)))
    player_card = player_cards.pop(np.random.randint(0, len(player_cards)))

    cards_on_stake.extend([dealer_cards.pop(0), dealer_cards.pop(0), dealer_card,
                           player_cards.pop(0), player_cards.pop(0), player_card])

    if values[dealer_card[1]] == values[player_card[1]]:
        return simulateWAR(cards_on_stake, dealer_cards, player_cards, max_war_depth, current_depth + 1)

    if values[dealer_card[1]] > values[player_card[1]]:  # dealer won the WAR
        reward = -5
        dealer_cards.extend(cards_on_stake)
    else:  # player won the WAR
        reward = +5
        player_cards.extend(cards_on_stake)
    return reward, player_cards, dealer_cards


class WarEnv(gym.Env):
    def __init__(self, has_infinite_deck=False):
        super(WarEnv, self).__init__()
        self.deck = []
        self.dealerCards = []
        self.playerCards = []
        self.shownCard = None
        self.has_infinite_deck = has_infinite_deck

        self.initDeck()  # creating a deck of cards and shuffling it
        self.distributeDeck()  # equally distributing the cards between player and dealer

        # [Value of Dealer's card, player card with the highest value, player card with the lowest value, has equal card as dealers]
        self.observation_space = Tuple(
            (Discrete(13, start=2), Discrete(13, start=2), Discrete(13, start=2), Discrete(2)))
        self.action_space = Discrete(3)  # 3 actions: select-highest ; select-lowest ; Select-equal

    def step(self, action):
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action {action}. Allowed actions are 0, 1, and 2.")

        dealer_card_value = values[self.shownCard[1]]
        highest, index_highest, lowest, index_lowest, index_equal = (
            findHighestAndLowestCardValues(self.playerCards, self.shownCard))
        reward = 0

        if action == 0:  # select-highest action

            if dealer_card_value > highest:
                reward = -1
                self.dealerCards.append(self.shownCard)
                self.dealerCards.append(self.playerCards.pop(index_highest))

            elif dealer_card_value < highest:
                reward = 1
                self.playerCards.append(self.shownCard)

            else:
                reward, self.playerCards, self.dealerCards = simulateWAR(
                    [self.shownCard, self.playerCards.pop(index_highest)],
                    self.dealerCards,
                    self.playerCards)

        elif action == 1:  # select-lowest action
            if dealer_card_value > lowest:
                reward = -1
                self.dealerCards.append(self.shownCard)
                self.dealerCards.append(self.playerCards.pop(index_lowest))

            elif dealer_card_value < lowest:
                reward = 1
                self.playerCards.append(self.shownCard)

            else:
                reward, self.playerCards, self.dealerCards = simulateWAR(
                    [self.shownCard, self.playerCards.pop(index_lowest)],
                    self.dealerCards,
                    self.playerCards)

        else:  # select-equal action
            if index_equal > 0:  # simulate WAR
                reward, self.playerCards, self.dealerCards = simulateWAR(
                    [self.shownCard, self.playerCards.pop(index_equal)],
                    self.dealerCards,
                    self.playerCards)
            else:
                reward = -2  # No equal card is present, penalize the invalid action
                # self.dealerCards.append(self.shownCard)  #Let the player know it is an invalid action

        if reward != -2:
            if len(self.dealerCards) > 0:
                self.shownCard = self.dealerCards.pop()
            else:
                self.shownCard = None

        terminated = False
        if len(self.playerCards) == 0 or len(self.dealerCards) == 0:
            terminated = True
            if len(self.dealerCards) == 0:
                reward = 10
            else:
                reward = -10

        highest, _, lowest, _, index_equal = findHighestAndLowestCardValues(self.playerCards, self.shownCard if self.shownCard else 0)
        has_equal = 0
        if index_equal > 0:
            has_equal = 1

        if self.shownCard:
            dealer_card_value = values[self.shownCard[1]]

            next_state = tuple([dealer_card_value, highest, lowest, has_equal])
        else:
            # Provide a valid final state for termination
            dealer_card_value = 0
            # highest, lowest, has_equal = 0, 0, 0
            next_state = tuple([dealer_card_value, highest, lowest, has_equal])

        # if terminated and self.shownCard is not None:
        #     self.playerCards.append(self.shownCard)
        if len(self.playerCards) == 0 and self.shownCard is not None:
            self.dealerCards.append(self.shownCard)
        elif len(self.dealerCards) == 0 and self.shownCard is not None:
            self.playerCards.append(self.shownCard)

        return next_state, reward, terminated, {}, {"playerCards": len(self.playerCards), "dealerCards": len(self.dealerCards), "shownCard": self.shownCard}

    def reset(self):
        self.dealerCards = []
        self.playerCards = []
        self.shownCard = None
        self.initDeck()
        self.distributeDeck()

        self.shownCard = self.dealerCards.pop()
        dealer_card_value = values[self.shownCard[1]]
        highest, _, lowest, _, index_equal = findHighestAndLowestCardValues(self.playerCards, self.shownCard)

        has_equal = 0
        if index_equal > 0:
            has_equal = 1

        return [tuple([dealer_card_value, highest, lowest, has_equal])]

    def shuffleDeck(self):
        self.deck = random.sample(self.deck, k=len(self.deck))

    def render(self, mode='human'):
        pass

    def initDeck(self):
        self.deck = [tuple([color, value]) for value in face_values for color in colors]
        self.shuffleDeck()

    def distributeDeck(self):
        for i in range(len(self.deck)):
            if i % 2 == 0:
                self.playerCards.append(self.deck[i])
            else:
                self.dealerCards.append(self.deck[i])
