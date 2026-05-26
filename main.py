from agent import train_q_learning_agent, choose_action
from environment import Grid

import matplotlib.pyplot as plt

def main():
    env = Grid()

    print("Training agent...")
    q_table, reward_history = train_q_learning_agent(env, episodes=100)
    print("Done!")

    plt.plot(reward_history)
    plt.title("Agent Learning Curve over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.show()




if __name__ == "__main__":
    main()