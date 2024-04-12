import torch
import numpy as np

from copy import deepcopy
from collections import deque

from easyq.agent import BetNet, HitNet
from easyq.config import Config
from easyq.util import playerCardCounterUpdate, ExperienceReplay, get_episode_no, get_epsilon
from easyq.env import simulateEasy21_finite_deck

start_using_betnet = Config.START_USING_BETNET


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
                _, cash_winnings, percent_change_in_bankroll = outcome

                # Update hitstock_ER with the round experience
                for i in range(len(round_hitstick_er)):
                    state, action = round_hitstick_er[i]
                    next_state = round_hitstick_er[i+1][0] if i+1 < len(round_hitstick_er) else np.zeros(shape = state.shape, dtype=np.float32)
                    done = i+1 == len(round_hitstick_er)
                    
                    # TODO: fix the state vector stuff.
                    
                    state_vector = hitnet.transform_state(player_sum, dealer_sum, player_cardcount_signal)
                    
                    # If the player is done, add the final state to the experience replay buffer
                    if done:
                        hitstick_ER.add(state_vector, action, percent_change_in_bankroll, next_state, done)
                    else: 
                        hitstick_ER.add(state_vector, action, 0, next_state, done)


                # Update bet_ER with the round experience
                for state, bet_size in round_bet_er:
                    bet_ER.append((state, bet_size, percent_change_in_bankroll))

                round_bet_er = []
                round_hitstick_er = []
                
        # Update number of full shoes played
        n_full_shoes_played += 1

    return bet_ER, hitstick_ER

### Action functions ###

def bet_size_net_pick(card_counting_signal_input: float, num_cards_left_in_shoe: int, current_bankroll: int) -> int:
    episode_no = get_episode_no()
    
    if episode_no <= 0:
        return generate_random_bet(current_bankroll)
    
    # Do random vs optimal bet size selection here
    epsilon = get_epsilon()
    
    if np.random.rand() < epsilon: # Do best bet size selection
        bet_size = generate_random_bet(current_bankroll)
    else: # Do random bet size selection
        pass
    
    return bet_size

def generate_random_bet(current_bankroll : int) -> int:
    # With probablity 0.5, bet 0
    if np.random.rand() < 0.5:
        return 0
    else: # Randomly get a bet size between 1 and current bankroll
        random_bet = int(np.random.randint(1, current_bankroll + 1))
        return random_bet
    

def hit_net_pick(player_sum, dealer_sum, card_counting_signal) -> bool:
    # Do random vs optimal bet size selection here
    epsilon = get_epsilon()
    
    if np.random.rand() < epsilon: # Do best bet size selection
        action = bool(np.random.choice([0, 1]))
    else: # Do random hit or stick
        pass
        
    
    return action
