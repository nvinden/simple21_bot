

class Config():
    # Game rules:
    N_EPISODES = 100
    N_SHOES_PER_EPISODE = 1
    
    # Training Epsilon:
    INITIAL_EPSILON = 1.0
    FINAL_EPSILON = 0.1
    EPSILON_DECAY = 1.00#0.999 TEMP
    
    # Episode Number to start training and using BetNet
    # Before that just bet a random amount
    START_USING_BETNET = 10

    # BetNet configuration
    BET_INPUT_DIM = 3
    BET_MODEL_DIM = 64

    # HitNet configuration
    HIT_INPUT_DIM = 3
    HIT_MODEL_DIM = 64