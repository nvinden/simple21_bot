import numpy as np
import gym
from gym import spaces


class Easy21Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Easy21Env, self).__init__()
        # Define action and observation space
        # Example: Actions are: 0 (stick) and 1 (hit)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: player's hand, dealer's first card, and the card counting signal
        self.observation_space = spaces.Tuple((
            spaces.Discrete(22),  # Player's hand: 0-21
            spaces.Discrete(11),  # Dealer's showing card: 1-10
            spaces.Box(low=np.array([-1]), high=np.array([1]))  # Example card counting signal range
        ))

        # Initialize state
        self.state = None

    def step(self, action):
        # Implement the game logic for one step given an action
        # Return: observation (object), reward (float), done (bool), info (dict)
        pass

    def reset(self):
        # Reset the state of the environment to an initial state
        # Return: initial observation
        pass

    def render(self, mode='human', close=False):
        # Render the environment to the screen (optional)
        pass

    def close(self):
        # Any cleanup goes here (optional)
        pass

    
    
    
    
    