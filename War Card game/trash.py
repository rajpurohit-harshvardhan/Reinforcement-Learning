from gymnasium.spaces import Discrete, Tuple

a = Discrete(13, start=2)
# print(a.sample())

colors = ['Hearts', 'Diamonds', 'Spades', 'Clubs']
face_values = [ '2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J', 'Q', 'K']
deck = [tuple([color, value]) for value in face_values for color in colors]

def distributeDeck():
    playerCards=[]
    dealerCards=[]
    for i in range(len(deck)):
        if i % 2 == 0:
            playerCards.append(deck[i])
        else:
            dealerCards.append(deck[i])

    print(playerCards, dealerCards)

distributeDeck()