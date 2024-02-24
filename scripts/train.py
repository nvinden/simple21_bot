import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from easyq import env, play, agent

def train():
    model = agent.DQN(state_size = env.STATE_SIZE, action_size = 2)
    
    experience_replay = play.generate_experience_replay(model)

if __name__ == "__main__":
    train()