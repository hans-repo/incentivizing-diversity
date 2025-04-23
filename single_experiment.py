import random
import numpy as np
from agent import *
from measurement_functions import *

# Define the possible states an agent can be in
POSSIBLE_STATES = ['State_A', 'State_B', 'State_C', 'State_D']
NUM_STATES = len(POSSIBLE_STATES)
N_AGENTS = 100  # Number of agents
PER_NODE_REWARD_BASE = 1000
PER_NODE_REWARD = PER_NODE_REWARD_BASE  # Per node reward per epoch in an evenly distributed system
# Define fixed rewards for each state
BASE_REWARDS = PER_NODE_REWARD * N_AGENTS / len(POSSIBLE_STATES)
REWARDS_ADAPTIVE_PARAM = 0.2
BASE_RUN_COST = 1000
BASE_SWITCH_COST = 1000
RUN_COST_CEILING = BASE_RUN_COST + 2*BASE_RUN_COST
state_run_costs = np.linspace(BASE_RUN_COST, RUN_COST_CEILING, NUM_STATES)  # One value per state
# Assign a single value to each state
STATE_RUN_COSTS = {
    state: round(value, 4)  # Round to 4 decimal places for readability
    for state, value in zip(POSSIBLE_STATES, state_run_costs)
}
STATE_SWITCH_COSTS = {state: BASE_SWITCH_COST for state in POSSIBLE_STATES}


def main():
    epochs = 200  # Number of epochs
    state_rewards =      {state: BASE_REWARDS for state in POSSIBLE_STATES}
    agents = generate_agents(N_AGENTS, POSSIBLE_STATES, state_rewards, STATE_RUN_COSTS, STATE_SWITCH_COSTS)
    agents_real_history = []
    agents_declared_history = []
    for epoch in range(epochs):
        state_rewards = update_state_rewards(agents, POSSIBLE_STATES, state_rewards, REWARDS_ADAPTIVE_PARAM)
        state_rewards_last_epoch = distribute_rewards(agents, POSSIBLE_STATES, state_rewards)
        agents_real_history.append([agent.real_state for agent in agents])
        agents_declared_history.append([agent.declared_state for agent in agents])
        for agent in agents:
            agent.decision(state_rewards_last_epoch, malicious=True)
    final_diversity = calculate_diversity(agents_real_history, range(epochs), POSSIBLE_STATES, N_AGENTS)
    declared_diversity = calculate_diversity(agents_declared_history, range(epochs), POSSIBLE_STATES, N_AGENTS)
    declared_largest_state = get_largest_state(agents_declared_history, range(epochs), POSSIBLE_STATES, N_AGENTS)
    print("running costs: ", STATE_RUN_COSTS)
    ideal_diversity = get_ideal_diversity(POSSIBLE_STATES)
    print("ideal diversity: ", ideal_diversity)
    print("declared diversity at last epoch: ", declared_diversity[epochs - 1])
    print("largest declared state share of nodes: ", declared_largest_state[epochs-1])
    print("real diversity at last epoch: ", final_diversity[epochs - 1])
    print("rewards per state at the end: ", state_rewards_last_epoch)
    print("base rewards initally: ", BASE_REWARDS)
    plot_stacked_area(agents_declared_history, epochs, POSSIBLE_STATES)
    plot_stacked_area(agents_real_history, epochs, POSSIBLE_STATES)


if __name__ == "__main__":
    main()