import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from easyq.util import ExperienceReplay

class BetNet(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim, max_bet=100000):
        super(BetNet, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer_block = nn.TransformerEncoderLayer(d_model=model_dim, nhead=1)
        self.output_layer = nn.Linear(model_dim, output_dim)
        self.max_bet = max_bet

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_block(x)
        x = self.output_layer(x)
        return torch.sigmoid(x) * self.max_bet  # Scale output to [0, max_bet]

    def transform_state(self, card_count, cards_left, bank_roll):
        # Normalize inputs more meaningfully considering the problem context
        # Normalize card_count to range from -1 to 1 (assuming card_count can be both positive and negative)
        normalized_card_count = card_count / 10.0  # Assuming card_count is within [-10, 10]
        # Normalize cards_left as a fraction of the total deck (assuming 312 cards in total, for 6 decks)
        normalized_cards_left = cards_left / 312.0
        # Normalize bank_roll logarithmically to have a non-linear relationship
        # Adding 1 to avoid log(0) and scaling logarithmically against max_bet
        normalized_bank_roll = np.log(bank_roll + 1) / np.log(self.max_bet + 1)
        
        return np.array([normalized_card_count, normalized_cards_left, normalized_bank_roll], dtype=np.float32)

    def action_from_output(self, output):
        # Convert network output to a discrete action
        return output.item()

class HitNet(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim=2, max_bet = 100_000):
        super(HitNet, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer_block = nn.TransformerEncoderLayer(d_model=model_dim, nhead=1)
        self.output_layer = nn.Linear(model_dim, output_dim)
        
        self.max_bet = max_bet

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_block(x)
        x = self.output_layer(x)
        return F.softmax(x, dim=-1)  # Output probabilities for actions

    def transform_state(self, player_sum, dealer_sum, bank_roll):
        # Normalize inputs, assuming bank_roll normalization as in BetNet for consistency
        normalized_bank_roll = np.log(bank_roll + 1) / np.log(self.max_bet + 1)
        return np.array([player_sum/21, dealer_sum/21, normalized_bank_roll], dtype=np.float32)

    def action_from_output(self, output):
        # Choose action based on output probabilities
        return torch.argmax(output).item()  # 0 for stick, 1 for hit


### Gradient Updates ###

def update_bet_net(betnet, bet_ER, bet_optimizer, batch_size=32):
    pass

def update_hit_net(hitnet, hitstick_ER : ExperienceReplay, hit_optimizer, batch_size=32, gamma=0.99, no_updates=100):
    # This will hold the average loss per update for debugging or tracking progress
    total_loss = 0

    for _ in range(no_updates):
        # Check if the experience replay buffer has enough samples
        if len(hitstick_ER) < batch_size:
            return  # Not enough samples to perform a training update

        # Sample a batch from the experience replay
        states, actions, rewards, next_states, dones = hitstick_ER.sample(batch_size)

        # Convert data to torch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        # Compute current Q values: Q(s, a)
        current_q_values = hitnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute next Q values: max_a' Q(s', a')
        with torch.no_grad():
            next_q_values = hitnet(next_states).max(1)[0]
            # Apply masking for terminal states
            next_q_values[dones] = 0.0

        # Compute the target Q values
        target_q_values = rewards + gamma * next_q_values

        # Calculate loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the model
        hit_optimizer.zero_grad()
        loss.backward()
        hit_optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / no_updates
    return average_loss
