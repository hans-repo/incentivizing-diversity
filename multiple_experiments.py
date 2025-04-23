import random
import numpy as np
from agent import *
from measurement_functions import *

# Config to show increase in switch cost offer good independence with up to 50% run cost variation.
num_experiments = 1
NUM_ATTRIBUTES = 100

def main():
    final_diversity_all_experiments = [[] for _ in range(NUM_ATTRIBUTES)]
    last_25_percent_diversity = [[] for _ in range(NUM_ATTRIBUTES)]
    flattened_diversity = [[] for _ in range(NUM_ATTRIBUTES)]
    average_last_25_percent = [[] for _ in range(NUM_ATTRIBUTES)]
    ideal_diversity_all_experiments = [[] for _ in range(NUM_ATTRIBUTES)]
    largest_state_all_experiments = [[] for _ in range(NUM_ATTRIBUTES)]

    last_25_percent_largest_state = [[] for _ in range(NUM_ATTRIBUTES)]
    flattened_largest_state = [[] for _ in range(NUM_ATTRIBUTES)]
    average_last_25_percent_largest_state = [[] for _ in range(NUM_ATTRIBUTES)]
    x_axis = []
    for i in range(num_experiments):
        # Define the possible states an agent can be in
        # POSSIBLE_STATES = ['NO_STATE', 'State_A', 'State_B', 'State_C', 'State_D']
        POSSIBLE_STATES = ['NO_STATE', 'State_A', 'State_B', 'State_C', 'State_D', 'State_E', 'State_F', 'State_G']
        NUM_STATES = len(POSSIBLE_STATES)
        N_AGENTS = 100  # Number of agents
        PER_NODE_REWARD_BASE = 1000
        PER_NODE_REWARD = PER_NODE_REWARD_BASE  # Per node reward per epoch in an evenly distributed system
        # Define fixed rewards for each state
        BASE_REWARDS = PER_NODE_REWARD_BASE * N_AGENTS / (NUM_STATES-1)
        REWARDS_ADAPTIVE_PARAM = 1000 + 0*i
        REWARDS_INTEGRAL_PARAM = 100
        REWARDS_DERIVATIVE_PARAM = 100
        # BASE_RUN_COST = PER_NODE_REWARD_BASE / 1
        BASE_RUN_COST = 0 + 100
        RUN_COST_CEILING = BASE_RUN_COST + 2*BASE_RUN_COST
        # BASE_SWITCH_COST = 0 + 1*PER_NODE_REWARD_BASE / 200
        BASE_SWITCH_COST = BASE_RUN_COST
        epochs = 5000 + 0*i  # Number of epochs
        # STATE_RUN_COSTS = {state: random.uniform(1.0, RUN_COST_CEILING ) * BASE_RUN_COST for state in POSSIBLE_STATES}
        # Generate evenly spaced values for each state
        state_run_costs = np.linspace(BASE_RUN_COST, RUN_COST_CEILING, NUM_STATES)  # One value per state
        # Assign a single value to each state
        STATE_RUN_COSTS = {
            state: round(value, 4)  # Round to 4 decimal places for readability
            for state, value in zip(POSSIBLE_STATES, state_run_costs)
        }

        STATE_SWITCH_COSTS = {state: BASE_SWITCH_COST for state in POSSIBLE_STATES }
        # x_axis.append(BASE_SWITCH_COST/PER_NODE_REWARD)
        x_axis.append(i*0.01*100)

        
        state_rewards = [{state: BASE_REWARDS for state in POSSIBLE_STATES} for _ in range(NUM_ATTRIBUTES)]
        accumulated_error = [{state: 0 for state in POSSIBLE_STATES} for _ in range(NUM_ATTRIBUTES)]
        last_error = 0
        agents = generate_agents(N_AGENTS, POSSIBLE_STATES, state_rewards, STATE_RUN_COSTS, STATE_SWITCH_COSTS)
        agents_real_history = [[] for _ in range(NUM_ATTRIBUTES)]
        agents_declared_history = [[] for _ in range(NUM_ATTRIBUTES)]
        for epoch in range(epochs):
            state_rewards, accumulated_error, last_error = update_state_rewards(agents, POSSIBLE_STATES, state_rewards, accumulated_error, last_error, REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM)
            state_rewards_last_epoch = distribute_rewards(agents, POSSIBLE_STATES, state_rewards)
            for k in range(NUM_ATTRIBUTES):
                agents_real_history[k].append([agent.real_state[k] for agent in agents])
                agents_declared_history[k].append([agent.declared_state[k] for agent in agents])
            for agent in agents:
                agent.decision(state_rewards_last_epoch, malicious=False)
        final_diversity = calculate_diversity(agents_declared_history, range(epochs), POSSIBLE_STATES, N_AGENTS)
        # Calculate the number of epochs in the last 25%
        last_25_percent = int(epochs * 0.25)
        # Extract the last 25% of diversity values
        for k in range(len(final_diversity)):
            last_25_percent_diversity[k] = [final_diversity[k][epoch] for epoch in range(epochs - last_25_percent, epochs)]
        # Calculate the average of the last 25% values
        # Flatten the list of lists into a single list
            flattened_diversity[k] = [item for sublist in last_25_percent_diversity[k] for item in sublist]
            average_last_25_percent[k] = sum(flattened_diversity[k]) / len(flattened_diversity[k])
        # Append the average to final_diversity_all_experiments
            final_diversity_all_experiments[k].append(average_last_25_percent[k])
            ideal_diversity = get_ideal_diversity(POSSIBLE_STATES)
            ideal_diversity_all_experiments[k].append(ideal_diversity)
            largest_state = get_largest_state(agents_declared_history, range(epochs), POSSIBLE_STATES, N_AGENTS)
            last_25_percent_largest_state[k] = [largest_state[k][epoch] for epoch in range(epochs - last_25_percent, epochs)]
            flattened_largest_state[k] = [item for sublist in last_25_percent_largest_state[k] for item in sublist]
            average_last_25_percent_largest_state[k] = sum(flattened_largest_state[k]) / len(flattened_largest_state[k])
            largest_state_all_experiments[k].append(average_last_25_percent_largest_state[k])
    if num_experiments > 1:
        plt.figure(figsize=(10, 6))
        for k in range(NUM_ATTRIBUTES):
            label1 = "simulated diversity attribute " + str(k)
            label2 = "ideal diversity " + str(k)
            plt.plot(x_axis, final_diversity_all_experiments[k], label=label1)
            plt.plot(x_axis, ideal_diversity_all_experiments[k], label=label2)
        plt.xlabel('running cost increase with 2x cost gap between high and low')
        plt.ylabel('Diversity')
        plt.ylim([0,ideal_diversity+0.1*ideal_diversity])
        plt.title('Evolution of Diversity')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)

        # plt.xticks(x_axis)  # Show ticks at intervals of step_size

        plt.show()

        plt.figure(figsize=(10, 6))
        for k in range(len(largest_state_all_experiments)):
            label = "largest state in attribute " + str(k)
            plt.plot(x_axis,largest_state_all_experiments[k], label=label)
        plt.xlabel('running cost increase with 2x cost gap between high and low')
        plt.ylabel('Share of nodes on largest state')
        plt.title('Evolution of share of nodes on largest state')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)

        # plt.xticks(x_axis)  # Show ticks at intervals of step_size

        plt.show()
        average_loss = get_avg_loss(NUM_ATTRIBUTES, average_last_25_percent, ideal_diversity_all_experiments)
        print("average diversity loss: ", average_loss)
    else:
        epochs_list = list(range(epochs))
        plt.figure(figsize=(10, 6))
        for k in range(NUM_ATTRIBUTES):
            # Assuming final_diversity[k] is a dictionary where keys are epochs
            # and values are lists containing a single diversity value
            diversity_values = [final_diversity[k][epoch] for epoch in epochs_list]

            plt.plot(epochs_list, diversity_values, label=f"Diversity - Attribute {k}")
        print("rewards per state at the end: ", state_rewards_last_epoch)
        print("base rewards initally: ", BASE_REWARDS)
        print("ideal diversity: ", ideal_diversity)
        plt.show()
        for k in range(NUM_ATTRIBUTES):
        # Assuming final_diversity[k] is a dictionary where keys are epochs
        # and values are lists containing a single diversity value
            agents_history = [agents_real_history[k][epoch] for epoch in epochs_list]
            plot_stacked_area(agents_history, epochs, POSSIBLE_STATES)
        

if __name__ == "__main__":
    main()