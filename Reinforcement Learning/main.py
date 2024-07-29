import SARSA
import Q_learning
import random

total_episodes = 100000
max_steps = 100

cards_in_deck = [1, 2, 3]
cards_select_limit = 2


def play(env, Q):
    print("PLAYING :::: ")
    # INPUT :: shuffled deck of cards, select limit.
    max_reward = sum(sorted(cards_in_deck)[-cards_select_limit:])
    random.shuffle(cards_in_deck)
    actions = {0: "DISCARD", 1: "PICK", 2: "SWAP"}

    # WORKING ::
    selected_cards = []
    #  - pop out a card from the deck, represent it as a state, EVERY popped out card is the card in hand
    card_in_hand = cards_in_deck.pop()
    current_state = "(" + str(card_in_hand) + "|" + str(",".join(selected_cards)) + ")"
    current_state_index = env.state_to_index[current_state]
    reward = 0

    loop = True
    cards_selected_counter = 0
    while loop:
        Q_list = Q[current_state_index].tolist()
        Q_list_reversed = sorted(Q_list)[::-1]

        for q_value in Q_list_reversed:
            action = Q_list.index(q_value)

            possible_states = env.possible_states_dict[current_state][action]
            if len(possible_states) == 0:
                continue

            state_selected = None
            reward_attained = 0
            for state in possible_states:
                calculated_reward = SARSA.calculate_reward(current_state, state)
                if calculated_reward < 0:  # handles case when reward for given state and action pair is penalising
                    continue

                reward_attained = calculated_reward
                state_selected = state
                env.possible_states_dict[current_state][action].remove(state_selected)
                break

            if not state_selected:  # handles the case when no state ,for given action, is rewarding
                continue

            reward += reward_attained
            print(current_state + " + " + actions[action] + " = " + state_selected)
            current_state = state_selected
            current_state_index = env.state_to_index[state_selected]
            break

        selected_cards = [int(x) for x in
                          current_state[current_state.index("|") + 1:current_state.index(")")] if
                          x.isnumeric()]
        if len(selected_cards) == cards_select_limit and reward == max_reward:
            loop = False
    print("Reward:", reward)


def main():
    # Training the agent using SARSA
    env, Q = SARSA.create_environment(cards_in_deck, cards_select_limit)
    Q =SARSA.train(total_episodes, max_steps, env, Q)
    SARSA.display_Q_table(env, Q)

    # Training the agent using Q-Learning
    # env, Q = Q_learning.create_environment(cards_in_deck, cards_select_limit)
    # Q = Q_learning.train(total_episodes, max_steps, env, Q)
    # Q_learning.display_Q_table(env, Q)

    play(env, Q)

main()