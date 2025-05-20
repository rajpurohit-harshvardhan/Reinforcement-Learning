import random
import gymnasium as gym
import numpy as np

colors = ['Hearts', 'Diamonds', 'Spades', 'Clubs']
face_values = [ '2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J', 'Q', 'K']
deck = [tuple([color, value]) for value in face_values for color in colors]
# print(deck[0], deck[0][1])
# print(deck.pop(0)[0])
# print(deck.pop(0)[1].isnumeric())

# print(random.sample(deck, k=len(deck)))
x = gym.spaces.Box(low=np.array([4,2,0]), high=np.array([30, 11, 1]), shape=(3,), dtype=np.int64)
# print(x)
# print(x.sample())
# print(x.sample())
# print(x.sample())

values = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'A': 11, 'J': 10, 'Q': 10,
          'K': 10}
def calculate_card_values(player_cards, dealer_cards):
    usable_ace = False
    is_usable_ace_used = True
    is_first_ace = True
    player_card_value = 0
    for card in player_cards:
        player_card_value += values[card[1]]

        if card[1] == 'A' and not is_usable_ace_used:
            player_card_value -= 10

        if card[1] == 'A' and is_first_ace:
            is_first_ace = False
            is_usable_ace_used = False
            usable_ace = True

        if usable_ace and player_card_value > 21:
            usable_ace = False
            player_card_value -= 10


    return player_card_value, usable_ace

# print(calculate_card_values([tuple(['HEarts', 'A']), tuple(['Diamond', 'A']), tuple(['Diamond', '2']), tuple(['Diamond', '10']), tuple(['clubs', 'A'])], []))
# print(calculate_card_values([tuple(['HEarts', 'J']), tuple(['Diamond', 'A']), tuple(['Diamond', '2']), tuple(['Diamond', '10']), tuple(['clubs', 'A'])], []))
# print(calculate_card_values([tuple(['HEarts', 'J']), tuple(['Diamond', 'A'])], []))

# a = [1,2,3]
# print(a)
# a.append(4)
# print(a)


card_values = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}
def swapCards(cards, selected_card):
    print(cards, selected_card)
    lowest = 0
    index = 0
    for i in range(len(cards)):
        card = cards[i]
        if card_values[card[1]] < lowest:
            lowest = card_values[card[1]]
            index = i
    cards[index] = selected_card
    print(cards)
    return cards

# swapCards([deck[0], deck[1], deck[2]], tuple(["Clubs", "K"]))
a = 5
# print(type(a), type(a) == int)
# ball_position_y = np.linspace(0, 170, 5)
ball_position_y = np.linspace(-1, 1, 5)
print(ball_position_y)