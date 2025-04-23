import matplotlib.pyplot as plt
import math

def plot_stacked_area(agents_history, epochs, possible_states):
    state_counts = {state: [] for state in possible_states}
    for epoch in range(epochs):
        for state in possible_states:
            count = sum(1 for agent_state in agents_history[epoch] if agent_state == state)
            state_counts[state].append(count)
    plt.figure(figsize=(10, 6))
    plt.stackplot(range(epochs),
                  [state_counts[state] for state in possible_states],
                  labels=possible_states,
                  alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Number of Agents')
    plt.title('Evolution of Agent States Over Epochs')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Set x-ticks to show only integer values with larger steps
    step_size = epochs // 10  # Adjust this value to change the step size
    plt.xticks(range(0, epochs + 1, step_size))  # Show ticks at intervals of step_size

    plt.show()

def calculate_diversity(agents_real_history, epochs, possible_states, n_agents):
    NUM_ATTRIBUTES = len(agents_real_history)
    diversity_all_epoch = [{epoch: [] for epoch in epochs} for _ in range(NUM_ATTRIBUTES)]
    for k in range(NUM_ATTRIBUTES):
        state_counts = {state: [] for state in possible_states}
        for epoch in epochs:
            diversity_this_epoch = 0
            for state in possible_states:
                count = sum(1 for agent_state in agents_real_history[k][epoch] if agent_state == state)
                state_counts[state].append(count)
                if count > 0:
                    if state != 'NO_STATE':
                        diversity_this_epoch = diversity_this_epoch - count / n_agents * math.log2(count / n_agents)
                else:
                    diversity_this_epoch = diversity_this_epoch + 0
            diversity_all_epoch[k][epoch].append(diversity_this_epoch)
    return diversity_all_epoch



def get_largest_state(agents_history, epochs, possible_states, n_agents):
    NUM_ATTRIBUTES = len(agents_history)
    largest_state_all_epochs = [{epoch: [] for epoch in epochs}  for _ in range(NUM_ATTRIBUTES)]
    for k in range(NUM_ATTRIBUTES):
        state_counts = {state: [] for state in possible_states}
        
        for epoch in epochs:
            largest_state_this_epoch = 0
            for state in possible_states:
                count = sum(1 for agent_state in agents_history[k][epoch] if agent_state == state)
                state_counts[state].append(count)
                if count > largest_state_this_epoch:
                    largest_state_this_epoch = count
            largest_state_all_epochs[k][epoch].append(largest_state_this_epoch/n_agents)
    return largest_state_all_epochs


def get_ideal_diversity(possible_states):
    entropy = 0
    p = 1 / (len(possible_states)-1)
    for state in possible_states:
        if state != 'NO_STATE':
            entropy = entropy - p * math.log2(p)
    return entropy

def get_avg_loss(NUM_ATTRIBUTES, average_last_25_percent, ideal_diversity_all_experiments):
    total = 0
    for k in range(NUM_ATTRIBUTES):
        total = total + abs(ideal_diversity_all_experiments[k][0] - average_last_25_percent[k])/ideal_diversity_all_experiments[k][0]
    average_loss = total/NUM_ATTRIBUTES
    return average_loss