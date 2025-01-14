import gymnasium as gym
import numpy as np
import random
from itertools import permutations
from gymnasium.spaces import Discrete, Tuple

colors = ['Hearts', 'Diamonds', 'Spades', 'Clubs']
face_values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J', 'Q', 'K']
values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'A': 1, 'J': 10, 'Q': 10, 'K': 10}


def calculateCardValues(cards):
    # usable_ace = False  # flag that denotes whether a usable ace has been used amongst the cards
    # is_usable_ace_used = True  # flag to check whether usable ace has been utilized already, case when second ace is encountered amongst the cards
    # is_first_ace = True  # flag to detect the occurrence of first ace
    # total_card_value = 0
    #
    # for card in cards:
    #     total_card_value += values[card[1]]  # Add value of that card to the total variable
    #     if card[1] == 'A' and not is_usable_ace_used:
    #         # cards satisfying this condition means there are more than 1 aces in the selected cards,
    #         # therefore, all the upcoming aces will have a face value of 1.
    #         total_card_value -= 10  # since the default face value of ace is 11, therefore subtracting 10 gives 1
    #
    #     if card[1] == 'A' and is_first_ace:
    #         # condition use to detect the occurrence of first ace and mark the flag values respectively
    #         is_first_ace = False
    #         is_usable_ace_used = False
    #         usable_ace = True
    #
    #     if usable_ace and total_card_value > 21:
    #         # condition checks if a usable ace has been used which leads to a BUST, in that case,
    #         # the ace will not be usable and therefore hold the value of 1
    #         usable_ace = False
    #         total_card_value -= 10

    total_card_value = 0
    num_aces = 0

    for card in cards:
        total_card_value += values[card[1]]
        if card[1] == 'A':
            num_aces += 1

    # Upgrade aces from 1 to 11 as long as it doesn't cause a bust
    usable_ace = False
    if num_aces > 0 and total_card_value + 10 <= 21:
        total_card_value += 10
        usable_ace = True

    return total_card_value, int(usable_ace)


class BlackJackEnv(gym.Env):
    def __init__(self, has_infinite_deck=False):
        super(BlackJackEnv, self).__init__()
        self.deck = []
        self.discardedCards = []
        self.playerCards = []
        self.dealerCards = []
        self.has_infinite_deck = has_infinite_deck

        self.initDeck()  # creating a deck of cards and shuffling it

        # [total_value_of_player_cards, total_value_of_dealer_cards, usable_ace (bool)]
        # self.observation_space = gym.spaces.Box(low=np.array([4, 2, 0]), high=np.array([30, 11, 1]), shape=(3,),dtype=np.int64)
        self.observation_space = Tuple((Discrete(32), Discrete(11), Discrete(2)))
        self.action_space = Discrete(2)

    def step(self, action):
        if action:  # HIT action
            self.playerCards.append(self.deck.pop(0))
            player_card_value, usable_ace = calculateCardValues(self.playerCards)
            dealer_card_value = values[self.dealerCards[0][1]]
            next_state = tuple([player_card_value, dealer_card_value, usable_ace])

            reward = 0
            terminated = True
            if player_card_value == 21:
                reward = 1.0  # BLACKJACK, not natural
            elif player_card_value > 21:  # BUST
                reward = -1.0
            else:
                # the game is not over thus not returning any reward.
                reward = 0.0
                terminated = False
            return next_state, reward, terminated, {}, {}

        else:  # STICK action
            # this code runs when the player decides to STICK with the cards in hands
            player_card_value, usable_ace = calculateCardValues(self.playerCards)
            dealer_card_value = values[self.dealerCards[0][1]]
            next_state = tuple([player_card_value, dealer_card_value, usable_ace])

            final_dealer_card_value = dealer_card_value
            # while final_dealer_card_value < player_card_value and final_dealer_card_value < 17:
            while final_dealer_card_value < 17:
                # Changed from 21 to 17, since dealer can't hit if its card values = 17 as per the rules

                # the dealer now decides to pick cards until he is satisfied with its hand
                self.dealerCards.append(self.deck.pop(0))
                dealer_card_value, _ = calculateCardValues(self.dealerCards)
                final_dealer_card_value = dealer_card_value

        # all these conditions will only be executed once player has chosen STICK action
        # since the round is over, therefore we will discard all the cards from the player/dealer's hands
        if self.has_infinite_deck:
            # has_infinite_deck == TRUE, means it an infinite deck, therefore concatenating all used cards in deck
            self.deck.extend(self.playerCards)
            self.deck.extend(self.dealerCards)
            self.shuffleDeck()
        else:
            # has_infinite_deck == FALSE, means a finite deck, thus discarding the used cards.
            self.discardedCards.append([self.playerCards, self.dealerCards])

            if len(self.deck) <= 5:
                self.initDeck()

        self.playerCards = []
        self.dealerCards = []

        reward = 0
        # TIE
        # if player_card_value == final_dealer_card_value:  # rewards the player with a value of 0, when game is a draw
        #     reward = 0.0
        #
        # # WINS
        # elif player_card_value == 21:  # rewards the player with a value of 1.5 when he has a natural blackjack
        #     if len(self.playerCards) == 2:
        #         reward = 1.5
        #     else:
        #         reward = 1
        # elif 21 > player_card_value > final_dealer_card_value and final_dealer_card_value < 21:  # rewards the player with a value of 1 since  player_cards > dealer_cards
        #     reward = 1
        # elif final_dealer_card_value > 21:
        #     reward = 1
        #
        # # LOSSES
        # elif player_card_value > 21:  # penalizes the player with a value of -1 for losing
        #     reward = -1
        # elif player_card_value < final_dealer_card_value:
        #     reward = -1

        if player_card_value > 21:
            reward = -1.0
        elif final_dealer_card_value > 21 or player_card_value > final_dealer_card_value:
            reward = 1.0
        elif player_card_value < final_dealer_card_value:
            reward = -1.0
        else:
            reward = 0.0

        return next_state, reward, True, {}, {'final_dealer_card_value': final_dealer_card_value}
        # else:
        #     print("WHAT STATE IS THIS ?? ", next_state)

    def reset(self):
        self.discardedCards = []
        self.initDeck()

        player_cards = [self.deck.pop(0), self.deck.pop(0)]
        dealer_cards = [self.deck.pop(0)]
        player_card_value, usable_ace = calculateCardValues(player_cards)
        dealer_card_value = values[dealer_cards[0][1]]

        self.playerCards = player_cards
        self.dealerCards = dealer_cards

        return [tuple([player_card_value, dealer_card_value, usable_ace])]

    def shuffleDeck(self):
        self.deck = random.sample(self.deck, k=len(self.deck))

    def render(self, mode='human'):
        result = 1
        # print('render')

    def initDeck(self):
        self.deck = [tuple([color, value]) for value in face_values for color in colors]
        self.shuffleDeck()
