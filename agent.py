import numpy as np
from typing import Tuple, Dict, List
from numpy.typing import NDArray

from environment import Grid

# epsilon-greedy method for action selection is not implemented yet

def train_q_learning_agent(
    env: Grid,
    episodes: int = 2000,
    alpha: float = 0.1,         # Constant step-size
    gamma: float = 0.95,        # Discount factor
    epsilon_start: float = 1.0,  # Starting exploration probability
    epsilon_decay: float = 0.995,# Rate at which epsilon shrinks per episode
    epsilon_min: float = 0.01    # Lower bound for exploration
) -> Tuple[np.ndarray, List[float]]:
    """
    Trains a tabular Q-learning agent on the 5x5 Gridworld.
    
    INPUTS:
    -------
    - env: The initialized Grid class instance.
    - episodes: Total number of games to play to let the agent learn.
    - alpha: How heavily to weight new TD errors vs past knowledge.
    - gamma: How much the agent values future rewards vs immediate ones.
    - epsilon_start/decay/min: Parameters controlling the exploration schedule.
    
    OUTPUTS:
    --------
    Returns a tuple containing:
    1. q_table: A NumPy array of shape (5, 5, 4) containing the learned 
                action-values for every state-action pair.
    2. reward_history: A list of floats containing the total cumulative reward 
                       earned in each consecutive episode (used to plot your 
                       learning curve).
    """
    
    # init Q-table: 5 rows, 5 columns, 4 actions
    q_table = np.zeros((env.height, env.width, len(env.action_space)))
    reward_history = []
    epsilon = epsilon_start

    # table of all N_t(a) count of each actions at the respective index 
    # at the specific state
    action_count = np.zeros(shape=q_table.shape) 

    for episode in range(episodes):
        state = env.reset()
        step = 1   
        total_reward = 0
        done = False

        print("episode:", episode)
        while not done or state != env.end_state:
            row, col = state[0], state[1]

            # choose an action using ucb or epsilon-greedy
            action_t = choose_action("ucb", epsilon, state, q_table, action_count, step, env)
            action_count[row, col, action_t] += 1

            # move by one step with chosen action
            # this already updates internal variable so setting state here does not really matter i think
            state, reward, done = env.step(action_t) # done variable is not used here so _

            # TODO: fix this with off-policy TD update rule. curr only goes to -inf
            q_table[row,col, action_t] += reward

            total_reward += reward
            
            print("step:", step)
            step +=1
        
        reward_history.append(total_reward)

    return q_table, reward_history

def choose_action(
        action_selection_method: str, 
        epsilon: float, 
        state: tuple, 
        Q_t: NDArray[np.float64],
        action_count: NDArray, # match q_table shape, and integers represent counts for each action at their respective index
        step: int,
        env: Grid) -> int:
    # action_selection_method = "ucb" or "epsilon-greedy"

    # choosing action with ucb
    if action_selection_method == "ucb":
        # curr row and col of this state
        row, col = state[0], state[1]

        # list of the action counts for every action at this current state
        n_t = [1e-6 if c == 0 else c for c in action_count[row,col,:]]

        a = np.argmax(Q_t[row, col, :] + 2*(np.sqrt(np.log(step) / n_t))) 
        # using broadcasting on log(step) so that it becomes an array of len 4

    elif action_selection_method == "epsilon":
        if np.random.rand() < epsilon:
            # explore
            a = np.random.choice(env.action_space)
        else:
            # exploit, choose max value in the q_table at this state
            a = np.argmax(Q_t[state[0],state[1], :])

    # write with epsilon-greedy next
    
    return int(a)