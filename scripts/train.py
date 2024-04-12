import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from easyq import env, play, agent, config, util
from collections import deque

def train():
    train_conf = config.Config()

    hitstock_ER = util.ExperienceReplay(10_000)
    bet_ER = deque(maxlen=10_000)

    betnet = agent.BetNet(train_conf.BET_INPUT_DIM, train_conf.BET_MODEL_DIM, 1)
    hitnet = agent.HitNet(train_conf.HIT_INPUT_DIM, train_conf.HIT_MODEL_DIM)

    for episode in range(train_conf.N_EPISODES):
        bet_er, hitnet_er = play.play_episode(betnet, hitnet, train_conf, bet_ER, hitstock_ER)

    
    

if __name__ == "__main__":
    train()