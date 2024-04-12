import random
import numpy as np

def playerCardCounterUpdate( card_counting_signal_input : float, num_cards_left_in_shoe : int, observed_card : int) -> float:
    # updates the card Counting signal based on:
    #  card_counting_signal_input : the previous card counting signal (which is a float)
    #  observed_card : the card that was dealt out that we just observed
    #  num_cards_left_in_shoe : how many cards are currently left in the deck

    #if the current card_counting_signal is None, then we are just starting a new episode and must initialize it to some value
    if card_counting_signal_input == None:
        playerCardCounterInitialValue = 1.833333333333333333333
        return playerCardCounterInitialValue
    
    if num_cards_left_in_shoe == 0:
        return 0.0
    
    total_card_value_last_deck = (card_counting_signal_input * num_cards_left_in_shoe + 1)
    output_card_counting_signal = (total_card_value_last_deck - observed_card) / (num_cards_left_in_shoe )

    return output_card_counting_signal

class ExperienceReplay:
    def __init__(self, capacity):
        """
        Initialize the Experience Replay buffer.
        
        Args:
        capacity (int): The maximum number of experiences the buffer can hold.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        
        Args:
        state (np.array): The current state.
        action (int): The action taken.
        reward (float): The reward received after taking the action.
        next_state (np.array): The next state after the action was taken.
        done (bool): Whether the episode has ended.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # Expand the list if not at full capacity
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.
        
        Args:
        batch_size (int): The size of the batch to sample.
        
        Returns:
        tuple: A tuple containing batches of states, actions, rewards, next_states, and dones.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.buffer)