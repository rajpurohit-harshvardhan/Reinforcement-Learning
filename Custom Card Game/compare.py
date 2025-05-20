import SARSA
import Q_learning
import random
import matplotlib.pyplot as plt

total_episodes = 100000
max_steps = 100
number_of_iterations_for_stats = 3

cards_in_deck = [1, 2, 3, 4, 5, 6, 7]
cards_select_limit = 3


def play(env, Q):
    try:
        cards_to_play = cards_in_deck[:]
        actions_taken = []

        print("PLAYING :::: ")
        # INPUT :: shuffled deck of cards, select limit.
        max_reward = sum(sorted(cards_to_play)[-cards_select_limit:])
        random.shuffle(cards_to_play)
        actions = {0: "DISCARD", 1: "PICK", 2: "SWAP"}

        # WORKING ::
        selected_cards = []
        #  - pop out a card from the deck, represent it as a state, EVERY popped out card is the card in hand
        card_in_hand = cards_to_play.pop()
        current_state = "(" + str(card_in_hand) + "|" + str(",".join(selected_cards)) + ")"
        current_state_index = env.state_to_index[current_state]
        reward = 0

        loop = True
        cards_selected_counter = 0
        possible_states_dict = dict(env.possible_states_dict)
        while loop:
            Q_list = Q[current_state_index].tolist()
            Q_list_reversed = sorted(Q_list)[::-1]

            if (len(possible_states_dict[current_state][0]) + len(possible_states_dict[current_state][1]) +
                   len(possible_states_dict[current_state][2])) == 0:
                print(len(selected_cards), reward, max_reward)
                loop = False

            for q_value in Q_list_reversed:
                action = Q_list.index(q_value)

                possible_states = possible_states_dict[current_state][action]
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
                    possible_states_dict[current_state][action].remove(state_selected)
                    break

                if not state_selected:  # handles the case when no state ,for given action, is rewarding
                    continue

                reward += reward_attained
                # print(current_state + " + " + actions[action] + " = " + state_selected)
                current_state = state_selected
                current_state_index = env.state_to_index[state_selected]
                actions_taken.append(actions[action])
                break

            selected_cards = [int(x) for x in
                              current_state[current_state.index("|") + 1:current_state.index(")")] if
                              x.isnumeric()]
            # print("Condition::", len(selected_cards), reward, max_reward, Q[current_state_index].tolist())
            if reward == max_reward:
                loop = False

        print("Reward:", reward)
        print(" --------------------- XX ----------------------")
        return actions_taken
    except Exception as e:
        print("An exception occurred ::", e)


def calculate_stats_and_plot_graph(env, Q, label):
    stats = {
        "DISCARD": 0,
        "PICK": 0,
        "SWAP": 0
    }

    for i in range(number_of_iterations_for_stats):
        actions = play(env, Q)
        stats["DISCARD"] += actions.count("DISCARD")
        stats["PICK"] += actions.count("PICK")
        stats["SWAP"] += actions.count("SWAP")
        # print(stats, i)

    print("Stats calculated :: ", stats)
    lists = sorted(stats.items())  # sorted by key, return a list of tuples

    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    plt.plot(x, y, label=label)
    return plt


def show_graph(graph):
    graph.legend()
    graph.show()


def main():
    # Training the agent using SARSA
    env, Q = SARSA.create_environment(cards_in_deck, cards_select_limit)
    Q = SARSA.train(total_episodes, max_steps, env, Q)
    # SARSA.display_Q_table(env, Q)
    plotted_graph = calculate_stats_and_plot_graph(env, Q, label="sarsa")

    # Training the agent using Q-Learning
    env, Q = Q_learning.create_environment(cards_in_deck, cards_select_limit)
    Q = Q_learning.train(total_episodes, max_steps, env, Q)
    # Q_learning.display_Q_table(env, Q)
    plotted_graph = calculate_stats_and_plot_graph(env, Q, label="q learning")
    show_graph(plotted_graph)


main()
