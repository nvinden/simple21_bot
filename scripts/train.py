import os
import sys
import torch
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from easyq import env, play, agent, config, util
from collections import deque

def train():
    # Config
    train_conf = config.Config()
    
    # Runname and logging. Name + timestamp
    now = datetime.now()
    runname = "easyq_test" + now.strftime("%Y-%m-%d %H:%M:%S")

    # Experience Replay Buffers
    hitstock_ER = util.ExperienceReplay(10_000)
    bet_ER = deque(maxlen=10_000)

    # Models
    betnet = agent.BetNet(train_conf.BET_INPUT_DIM, train_conf.BET_MODEL_DIM, 1)
    hitnet = agent.HitNet(train_conf.HIT_INPUT_DIM, train_conf.HIT_MODEL_DIM)
    
    # Optimizers
    bet_optimizer = torch.optim.Adam(betnet.parameters(), lr=0.001)
    hit_optimizer = torch.optim.Adam(hitnet.parameters(), lr=0.001)
    
    util.update_epsilon(train_conf.INITIAL_EPSILON)

    for episode_no in range(train_conf.N_EPISODES):
        util.update_episode_no(episode_no)
        
        bet_er, hitnet_er = play.play_episode(betnet, hitnet, train_conf, bet_ER, hitstock_ER)
        
        # Update the betnet weights (if we are past the threshold episode number)
        if episode_no >= train_conf.START_USING_BETNET:
            agent.update_bet_net(betnet, bet_er, bet_optimizer)
        
        # Update hitnet weights
        agent.update_hit_net(hitnet, hitnet_er, hit_optimizer)
        
        # Save the models
        util.save_models(betnet, hitnet, runname)
        
        # Change the epsilon rate for the next episode
        curr_epsilon = util.get_epsilon()
        next_epsilon = max(curr_epsilon * train_conf.EPSILON_DECAY, train_conf.FINAL_EPSILON)
        util.update_epsilon(next_epsilon)


    
    

if __name__ == "__main__":
    train()