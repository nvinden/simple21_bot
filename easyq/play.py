import torch
import numpy as np

from copy import deepcopy
from collections import deque

from easyq.agent import BetNet, HitNet
from easyq.config import Config
from easyq.util import playerCardCounterUpdate, ExperienceReplay
from easyq.env import simulateEasy21_finite_deck


### Playing Episodes ###

def play_episode(betnet: BetNet, hitnet: HitNet, train_conf: Config, bet_ER : deque, hitstick_ER : ExperienceReplay):
    bet_ER = deepcopy(bet_ER)
    hitstick_ER = deepcopy(hitstick_ER)

    n_full_shoes_played = 0

    round_bet_er = []
    round_hitstick_er = []


    while n_full_shoes_played < train_conf.N_SHOES_PER_EPISODE:
        game_sim = simulateEasy21_finite_deck(hit_net_pick, bet_size_net_pick, playerCardCounterUpdate, Verbose=False, num_decks_in_shoe=20, min_decks_to_end=2)

        for event, outcome in game_sim:
            
            if event == "BET":
                (card_count, cards_left, bank_roll, bet_size) = outcome
                state = betnet.transform_state(card_count, cards_left, bank_roll)
                bet_ER.append((state, bet_size))
            elif event == "HITSTICK":
                player_sum, dealer_sum, player_cardcount_signal, action = outcome
                state = hitnet.transform_state(player_sum, dealer_sum, player_cardcount_signal)
                round_hitstick_er.append((state, action))
            elif event == "OUTCOME":
                _, cash_winnings = outcome

                # Update hitstock_ER with the round experience
                for i in range(len(round_hitstick_er)):
                    state, action = round_hitstick_er[i]
                    next_state = round_hitstick_er[i+1][0] if i+1 < len(round_hitstick_er) else None
                    done = i+1 == len(round_hitstick_er)
                    hitstick_ER.add(state, action, cash_winnings, next_state, done)

                # Update bet_ER with the round experience
                for state, bet_size in round_bet_er:
                    bet_ER.append((state, bet_size, cash_winnings))

                round_hitstick_er = []

    return bet_ER, hitstick_ER

### Action functions ###

def bet_size_net_pick(card_counting_signal_input: float, num_cards_left_in_shoe: int, current_bankroll: int) -> int:
    bet_size = 1
    return bet_size

def hit_net_pick(player_sum, dealer_sum, card_counting_signal) -> bool:
    # Whether to hit or stick given the player sum, dealer sum and card counting signal (True/1 = Hit, False/0 = Stick)
    AlwaysHitAction = np.ones((22, 22))
    AlwaysStayAction = np.zeros((22, 22))
    actions = AlwaysStayAction  # Default to AlwaysStayAction for now
    return actions[dealer_sum, player_sum]
