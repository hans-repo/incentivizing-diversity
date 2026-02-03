import random
import numpy as np
from agent import *
from measurement_functions import *

# Define the possible versions an agent can be in
POSSIBLE_versionS = ['version_A', 'version_B', 'version_C', 'version_D']
NUM_versionS = len(POSSIBLE_versionS)
N_AGENTS = 544  # Number of agents
PER_NODE_REWARD_BASE = 1000
PER_NODE_REWARD = PER_NODE_REWARD_BASE  # Per node reward per epoch in an evenly distributed system
# Define fixed rewards for each version
BASE_REWARDS = PER_NODE_REWARD * N_AGENTS / len(POSSIBLE_versionS)
REWARDS_ADAPTIVE_PARAM = 0.2
BASE_RUN_COST = 1000
BASE_SWITCH_COST = 1000
RUN_COST_CEILING = BASE_RUN_COST + 2*BASE_RUN_COST
version_run_costs = np.linspace(BASE_RUN_COST, RUN_COST_CEILING, NUM_versionS)  # One value per version
# Assign a single value to each version
version_RUN_COSTS = {
    version: round(value, 4)  # Round to 4 decimal places for readability
    for version, value in zip(POSSIBLE_versionS, version_run_costs)
}
version_SWITCH_COSTS = {version: BASE_SWITCH_COST for version in POSSIBLE_versionS}


def main():
    epochs = 200  # Number of epochs
    version_rewards =      {version: BASE_REWARDS for version in POSSIBLE_versionS}
    agents = generate_agents(N_AGENTS, POSSIBLE_versionS, version_rewards, version_RUN_COSTS, version_SWITCH_COSTS)
    agents_real_history = []
    agents_declared_history = []
    for epoch in range(epochs):
        version_rewards = update_version_rewards(agents, POSSIBLE_versionS, version_rewards, REWARDS_ADAPTIVE_PARAM)
        version_rewards_last_epoch = distribute_rewards(agents, POSSIBLE_versionS, version_rewards)
        agents_real_history.append([agent.real_version for agent in agents])
        agents_declared_history.append([agent.declared_version for agent in agents])
        for agent in agents:
            agent.decision(version_rewards_last_epoch, malicious=True)
    final_diversity = calculate_diversity(agents_real_history, range(epochs), POSSIBLE_versionS, N_AGENTS)
    declared_diversity = calculate_diversity(agents_declared_history, range(epochs), POSSIBLE_versionS, N_AGENTS)
    declared_largest_version = get_largest_version(agents_declared_history, range(epochs), POSSIBLE_versionS, N_AGENTS)
    print("running costs: ", version_RUN_COSTS)
    ideal_diversity = get_ideal_diversity(POSSIBLE_versionS)
    print("ideal diversity: ", ideal_diversity)
    print("declared diversity at last epoch: ", declared_diversity[epochs - 1])
    print("largest declared version share of nodes: ", declared_largest_version[epochs-1])
    print("real diversity at last epoch: ", final_diversity[epochs - 1])
    print("rewards per version at the end: ", version_rewards_last_epoch)
    print("base rewards initally: ", BASE_REWARDS)
    plot_stacked_area(agents_declared_history, epochs, POSSIBLE_versionS)
    plot_stacked_area(agents_real_history, epochs, POSSIBLE_versionS)


if __name__ == "__main__":
    main()