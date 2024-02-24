import numpy as np


DECK_START = "default", # "random" or (num) for a specific # of decks or default which is 10 reds, 20 blacks


# STATE: all rows values are one-hot-encoded
# r0-r10: Black card count 
# r10-r20: Red card count
# r20: player hand count
# r21: dealer hand count
STATE_SIZE = (22, 22)
PLAYER_HAND_IDX = 20
DEALER_HAND_IDX = 21

# Starts a new game at indices.
def get_starting_state(deck_start = "default"):
    assert deck_start in ["random", "continue", "default"]
        
    new_state = np.zeros(STATE_SIZE, dtype = float)
    
    # Randomly select the number of cards in deck. Each card type
    # can be between 0 and 10
    if deck_start == "random":
        pass
    elif deck_start == "continue":
        pass
    elif deck_start == "default":
        # Set 10 red and 20 black cards
        new_state[0:10, 2] = 1.0
        new_state[10:20, 1] = 1.0
    else:
        raise Exception("Deck start not proper")
        
    # Define player and dealer's first card
    new_state = draw_card(new_state, "both", starting_cards=True)
        
    return np.array(new_state)

def draw_card(state, who, starting_cards = False):
    # Draw a card from the deck
    # who: "dealer", "player", "both"
    # starting_cards: True if the card is the first card drawn for the player and dealer
    assert who in ["dealer", "player", "both"], "Who is not dealer, player, or both"
    
    # Draw a card from the deck
    if who == "dealer" or who == "both":
        next_card_idx = _get_next_card_idx(state)
        
        card_magnitude = _card_idx_to_card_magnitude(next_card_idx)
        
        if starting_cards:
            card_magnitude = int(np.abs(card_magnitude))
            state = _state_set_idx_value(state, DEALER_HAND_IDX, card_magnitude)
        else: 
            # Setting the dealer's hand to the correct value
            dealer_hand_value = _state_get_idx_value(state, DEALER_HAND_IDX)
            state = _state_set_idx_value(state, DEALER_HAND_IDX, dealer_hand_value + card_magnitude)
            
        # Setting the deck to the correct value
        drawn_card_deck_count = _state_get_idx_value(state, next_card_idx)
        state = _state_set_idx_value(state, next_card_idx, drawn_card_deck_count - 1)
        
    if who == "player" or who == "both":
        next_card_idx = _get_next_card_idx(state)
        
        card_magnitude = _card_idx_to_card_magnitude(next_card_idx)
        
        if starting_cards:
            card_magnitude = int(np.abs(card_magnitude))
            state = _state_set_idx_value(state, PLAYER_HAND_IDX, card_magnitude)
        else: 
            # Setting the dealer's hand to the correct value
            dealer_hand_value = _state_get_idx_value(state, PLAYER_HAND_IDX)
            state = _state_set_idx_value(state, PLAYER_HAND_IDX, dealer_hand_value + card_magnitude)
        
        # Setting the deck to the correct value
        drawn_card_deck_count = _state_get_idx_value(state, next_card_idx)
        state = _state_set_idx_value(state, next_card_idx, drawn_card_deck_count - 1)
        
    return state
    
def to_string(state):
    out = ""
    
    player_val = _state_get_idx_value(state, PLAYER_HAND_IDX)
    dealer_val = _state_get_idx_value(state, DEALER_HAND_IDX)
    
    out += f"P:{player_val} D:{dealer_val} "
    
    for i in range(0, 10):
        count_val = _state_get_idx_value(state, i)
        out += f"B{i + 1}:{count_val} "
    
    for i in range(10, 20):
        count_val = _state_get_idx_value(state, i)
        out += f"R{i - 9}:{count_val} "

    return out

def n_cards_in_deck(state):
    # Get the number of cards in the deck
    cards_in_deck = np.sum(np.argmax(state[:-2], axis = 1))
    return cards_in_deck

def is_busted(state, who):
    # Check if the player or dealer is busted
    if who == "player":
        return _state_get_idx_value(state, PLAYER_HAND_IDX) == 0
    elif who == "dealer":
        return _state_get_idx_value(state, DEALER_HAND_IDX) == 0
    else:
        raise Exception("Who is not dealer or player")
        

###########
# Private #
###########

# Randomly selects the next card from the deck. Returns the state index of the chosen
# card
def _get_next_card_idx(state):
    # Get the next card from the deck
    
    # Indicies of the deck
    deck = state[:-2]
    
    # Gets the number of each type of card
    card_count = np.argmax(deck, axis = 1)
    
    # Randomly select the next card
    next_random_card_idx = np.random.choice(np.arange(0, 20), p = card_count / np.sum(card_count))
    
    return next_random_card_idx

def _card_idx_to_card_magnitude(idx):
    # Convert the card index to the magnitude of the card
    if 0 <= idx and idx < 10: # Black
        return idx + 1
    elif 10 <= idx and idx < 20: # Red
        return -1 * (idx - 9)
    else:
        raise Exception("Card index is not in the deck")
    
def _state_get_idx_value(state, idx) -> int:
    assert idx >= 0 and idx < STATE_SIZE[0], "Index is out of bounds"
    assert tuple(state.shape) == STATE_SIZE, "State is not the proper shape"
    
    return int(np.argmax(state[idx]))

def _state_set_idx_value(state : np.ndarray, idx : int, value : int) -> np.ndarray:
    assert idx >= 0 and idx < STATE_SIZE[0], "Index is out of bounds"
    assert tuple(state.shape) == STATE_SIZE, "State is not the proper shape"
    assert (value >= 0 and value < STATE_SIZE[1]) or idx in [PLAYER_HAND_IDX, DEALER_HAND_IDX], "Value is out of bounds"
    
    state[idx, :] = 0.0
    
    # Player or dealer busted
    if not (value >= 0 and value < STATE_SIZE[1]) and idx in [PLAYER_HAND_IDX, DEALER_HAND_IDX]:
        state[idx, 0] = 1.0
    else: 
        state[idx, value] = 1.0
    
    return state
    
    
    
    
    