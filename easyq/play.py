import torch

# Generates experience replay for a given model. Plays a number of games.
# creates a game tree, expanding using most likely draws. After n_expansions,
# the stay values are calculated using dealer rollouts and the hit values are
# calculated using player rollouts. The state, hit_reward, and stay_reward are
# stored in a list and returned.
# Parameters:
# - model: The model to generate experience replay for
# - n_games: The number of games to play
# - n_expansios: The number of expansions to play per game
# Returns: List of tuples (state, hit_reward, stay_reward)
def generate_experience_replay(model : torch.nn.Module, n_games : int = 30, n_rollouts : int = 500):
    pass