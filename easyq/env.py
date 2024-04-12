import numpy as np
import gym
from gym import spaces
import random


# Function that plays a single game of Easy21
# Parameters:
# - playerStrategy: A function that takes in the player's sum, the dealer's
#   showing card, and the player's card count signal and returns whether the
#   player should hit or stay
# - playerBetSize_choice: A function that takes in the player's card count
#   signal and the player's bankroll and returns the bet size
# - playerCardCounter: A function that takes in the player's card count signal
#   and the next card drawn and returns the updated card count signal
# - Verbose: Whether to print out the game state
def simulateEasy21_finite_deck(playerStrategy, playerBetSize_choice, playerCardCounter, Verbose=False, num_decks_in_shoe=20, min_decks_to_end=2):
    cards_1_to_10 = np.arange(1, 11)
    deck = np.concatenate((cards_1_to_10, cards_1_to_10, -cards_1_to_10))
    shoe = np.tile(deck, num_decks_in_shoe)
    np.random.shuffle(shoe)

    top_of_shoe_ix = 0
    player_bankroll = 100
    player_cardcount_signal = 1.833333333333333333333

    while len(shoe) - top_of_shoe_ix > min_decks_to_end * len(deck) and player_bankroll > 0:
        initial_round_bankroll = player_bankroll
        
        bet_size = int(playerBetSize_choice(player_cardcount_signal, len(shoe) - top_of_shoe_ix, player_bankroll))
        bet_size = max(min(bet_size, player_bankroll), 0)

        # YIELD: "BET", bet size
        yield ("BET", (player_cardcount_signal, len(shoe) - top_of_shoe_ix, player_bankroll, bet_size))

        if Verbose:
            print(f"{player_bankroll=}, {bet_size=}")

        player_sum = random.randint(1, 10)
        dealer_sum = random.randint(1, 10)

        if Verbose:
            print("==================")
            print(f"Player Starting Sum: {player_sum}")
            print(f"Dealer Starting Sum: {dealer_sum}")

        player_is_active = playerStrategy(player_sum, dealer_sum, player_cardcount_signal)
        player_busted = False

        # YIELD: "BET", player strategy
        yield ("HITSTICK", (player_sum, dealer_sum, player_cardcount_signal, int(player_is_active)))

        if Verbose:
            print("--Player's Turn")
        while player_is_active:
            nextCard = shoe[top_of_shoe_ix]
            top_of_shoe_ix += 1
            player_cardcount_signal = int(playerCardCounter(player_cardcount_signal, len(shoe) - top_of_shoe_ix, nextCard))
            player_sum += nextCard

            player_busted = player_sum < 1 or player_sum > 21
            player_is_active = not player_busted and playerStrategy(player_sum, dealer_sum, player_cardcount_signal)
            player_is_active = player_is_active and top_of_shoe_ix < len(shoe)

            # YIELD: "HITSTICK", state
            yield ("HITSTICK", (player_sum, dealer_sum, player_cardcount_signal, int(player_is_active)))

            if Verbose:
                print(f"{player_sum = }, {player_busted=}, {player_is_active=}")

        dealer_is_active = not player_busted and dealer_sum <= 16
        dealer_is_active = dealer_is_active and top_of_shoe_ix < len(shoe)
        dealer_busted = False

        if Verbose:
            print("--Dealer's Turn")
        while dealer_is_active:
            nextCard = shoe[top_of_shoe_ix]
            top_of_shoe_ix += 1
            player_cardcount_signal = int(playerCardCounter(player_cardcount_signal, len(shoe) - top_of_shoe_ix, nextCard))

            dealer_sum += nextCard
            dealer_busted = dealer_sum < 1 or dealer_sum > 21
            dealer_is_active = not dealer_busted and dealer_sum <= 16
            dealer_is_active = dealer_is_active and top_of_shoe_ix < len(shoe)

            if Verbose:
                print(f"{dealer_sum = }, {dealer_busted=}, {dealer_is_active=}")

        player_wins = dealer_busted or (not player_busted and player_sum > dealer_sum)
        dealer_wins = player_busted or (not dealer_busted and player_sum < dealer_sum)

        if Verbose:
            print(f"{player_wins=}, {dealer_wins=}")

        round_gain = bet_size * player_wins - bet_size * dealer_wins
        player_bankroll += round_gain

        # YIELD: "OUTCOME", player wins, dealer wins
        if initial_round_bankroll == 0:
            percent_change_in_bankroll = 0.0
        else:
            percent_change_in_bankroll = (initial_round_bankroll + round_gain) / initial_round_bankroll
        
        yield ("OUTCOME", (player_wins, round_gain, percent_change_in_bankroll))
