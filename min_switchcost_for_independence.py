import random
import numpy as np
from agent import *
from measurement_functions import *

# Config to show increase in switch cost offer good independence with up to 50% run cost variation.
num_experiments = 100


def main():
    final_diversity_all_experiments = []
    ideal_diversity_all_experiments = []
    x_axis = []
    for i in range(num_experiments):
        # Define the possible versions an agent can be in
        POSSIBLE_versionS = ['version_A', 'version_B', 'version_C', 'version_D']
        NUM_versionS = len(POSSIBLE_versionS)
        N_AGENTS = 100  # Number of agents
        PER_NODE_REWARD = 100  # Per node reward per epoch in an evenly distributed system
        # Define fixed rewards for each version
        version_REWARDS = PER_NODE_REWARD * N_AGENTS / NUM_versionS
        BASE_RUN_COST = PER_NODE_REWARD / 10
        RUN_COST_CEILING = BASE_RUN_COST + 0.5*BASE_RUN_COST
        BASE_SWITCH_COST = 0 + i*PER_NODE_REWARD / 200

        # version_RUN_COSTS = {version: random.uniform(1.0, RUN_COST_CEILING ) * BASE_RUN_COST for version in POSSIBLE_versionS}
        # Generate evenly spaced values for each version
        version_run_costs = np.linspace(BASE_RUN_COST, RUN_COST_CEILING, NUM_versionS)  # One value per version
        # Assign a single value to each version
        version_RUN_COSTS = {
            version: round(value, 4)  # Round to 4 decimal places for readability
            for version, value in zip(POSSIBLE_versionS, version_run_costs)
        }

        version_SWITCH_COSTS = {version: BASE_SWITCH_COST for version in POSSIBLE_versionS }
        x_axis.append(BASE_SWITCH_COST/PER_NODE_REWARD)


        epochs = 200  # Number of epochs
        agents = generate_agents(N_AGENTS, POSSIBLE_versionS, version_REWARDS, version_RUN_COSTS, version_SWITCH_COSTS)
        agents_real_history = []
        agents_declared_history = []
        for epoch in range(epochs):
            version_rewards_last_epoch = distribute_rewards(agents, POSSIBLE_versionS, version_REWARDS)
            agents_real_history.append([agent.real_version for agent in agents])
            agents_declared_history.append([agent.declared_version for agent in agents])
            for agent in agents:
                agent.decision(version_rewards_last_epoch, malicious=True)
        final_diversity = calculate_diversity(agents_real_history, range(epochs), POSSIBLE_versionS, N_AGENTS)
        final_diversity_all_experiments.append(final_diversity[epochs-1])
        ideal_diversity = get_ideal_diversity(POSSIBLE_versionS)
        ideal_diversity_all_experiments.append(ideal_diversity)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, final_diversity_all_experiments, label="simulated diversity")
    plt.plot(x_axis, ideal_diversity_all_experiments, label="ideal diversity")
    plt.xlabel('config switch cost (fraction of expected per node reward)')
    plt.ylabel('Diversity')
    plt.title('Evolution of Diversity')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    # plt.xticks(x_axis)  # Show ticks at intervals of step_size

    plt.show()


if __name__ == "__main__":
    main()