import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Normalize inputs
        return torch.tensor([card_count/10, cards_left/312, bank_roll/self.max_bet]).float()

    def action_from_output(self, output):
        # Convert network output to a discrete action
        return output.item()

class HitNet(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim=2):
        super(HitNet, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer_block = nn.TransformerEncoderLayer(d_model=model_dim, nhead=1)
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_block(x)
        x = self.output_layer(x)
        return F.softmax(x, dim=-1)  # Output probabilities for actions

    def transform_state(self, player_sum, dealer_sum, bank_roll):
        # Normalize inputs, assuming bank_roll normalization as in BetNet for consistency
        return torch.tensor([player_sum/21, dealer_sum/21, bank_roll/100000]).float()

    def action_from_output(self, output):
        # Choose action based on output probabilities
        return torch.argmax(output).item()  # 0 for stick, 1 for hit
