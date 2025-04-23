import random

class Agent:
    def __init__(self, id, possible_states, state_rewards, state_run_costs, state_switch_costs):
        self.id = id
        self.possible_states = possible_states  # Pass possible states
        self.state_rewards = state_rewards  # Pass state rewards
        self.state_run_costs = state_run_costs  # Pass state run costs
        self.state_switch_costs = state_switch_costs  # Pass state switch costs
        NUM_ATTRIBUTES = len(state_rewards)
        self.declared_state = ['State_A' for _ in range(NUM_ATTRIBUTES)]  # Initial declared state
        self.real_state = ['State_A' for _ in range(NUM_ATTRIBUTES)]  # Initial real state
        self.reward = 0  # Track the reward received by the agent
        self.personal_switch_cost = 0
        self.switch_cost = {state: random.uniform(0.0, 1.0) * self.personal_switch_cost for state in possible_states}
        self.epochs_since_last_change = [0 for _ in range(NUM_ATTRIBUTES)]  # Track epochs since last state change
        self.switch_threshold = [random.randint(1, 25) for _ in range(NUM_ATTRIBUTES)]  # Randomly chosen threshold for switching

    def decision(self, state_rewards_last_epoch, malicious=False):
        net_gains = {}
        for k in range(len(self.declared_state)):
            for state in self.possible_states:
                if state == self.real_state[k]:
                    net_gains[state] = state_rewards_last_epoch[k][state] - self.state_run_costs[state]
                else:
                    net_gains[state] = (state_rewards_last_epoch[k][state] -
                                        self.state_run_costs[state] -
                                        (self.switch_cost[state] + self.state_switch_costs[state]))
            best_state = random_max(net_gains, key=net_gains.get)
            if net_gains[best_state] > net_gains[self.declared_state[k]] and self.epochs_since_last_change[k] >= self.switch_threshold[k]:
                # Switch state and reset the counter
                self.real_state[k] = best_state
                self.declared_state[k] = best_state
                self.epochs_since_last_change[k] = 0  # Reset the counter after switching
            elif net_gains[best_state] > net_gains[self.declared_state[k]] and self.epochs_since_last_change[k] < self.switch_threshold[k]:
                # Increment the counter if no switch is made
                self.epochs_since_last_change[k] += 1
            else:
                self.epochs_since_last_change[k] = 0

            if malicious:
                total_cost = {}
                for state in self.possible_states:
                    if state == self.declared_state[k]:
                        total_cost[state] = self.state_run_costs[state]
                    else:
                        total_cost[state] = self.state_run_costs[state] + (self.switch_cost[state] + self.state_switch_costs[state])
                cheapest_state = random_min(total_cost, key=total_cost.get, default=self.declared_state[k])

                self.real_state[k] = cheapest_state


def random_min(iterable, key=None, default=None):
    if key is None:
        key = lambda x: x  # Default key function (identity function)

    # Find the minimum value based on the key function
    min_value = min(iterable, key=key)

    # Collect all elements that have the minimum value (based on the key function)
    min_elements = [x for x in iterable if key(x) == key(min_value)]

    # If the default parameter is in the tied elements, choose it
    if default in min_elements:
        return default
    else:
        # Otherwise, randomly select one of the tied elements
        return random.choice(min_elements)


def random_max(iterable, key=None):
    if key is None:
        key = lambda x: x  # Default key function (identity function)

    # Find the maximum value based on the key function
    max_value = max(iterable, key=key)

    # Collect all elements that have the maximum value (based on the key function)
    max_elements = [x for x in iterable if key(x) == key(max_value)]

    # Randomly select one of the tied elements
    return random.choice(max_elements)


def generate_agents(n, possible_states, state_rewards, state_run_costs, state_switch_costs):
    return [Agent(i, possible_states, state_rewards, state_run_costs, state_switch_costs) for i in range(n)]



def update_state_rewards(agents, possible_states, state_rewards, accumulated_error, last_error, adaptive_param, integral_param, derivative_param):
    for k in range(len(state_rewards)):
        state_counts = {state: 0 for state in possible_states}
        for agent in agents:
            state_counts[agent.declared_state[k]] += 1
        for state in possible_states:
            if state == 'NO_STATE':
                state_rewards[k][state] = 0
            else: 
                state_share = state_counts[state]/len(agents)
                ideal_share = 1/(len(possible_states)-1)
                error = (ideal_share-state_share)
                accumulated_error[k][state] = accumulated_error[k][state] + error
                P_term = adaptive_param*error
                # max_integral = 100
                # I_term = integral_param*max(min(accumulated_error[k][state], max_integral), -max_integral)
                I_term = integral_param*accumulated_error[k][state]
                D_term = derivative_param * (error - last_error)
                state_rewards[k][state] = state_rewards[k][state] + P_term + I_term + D_term
    return state_rewards, accumulated_error, error



def distribute_rewards(agents, possible_states, state_rewards):
    NUM_ATTRIBUTES = len(state_rewards)
    state_rewards_last_epoch =  state_rewards
    for k in range(NUM_ATTRIBUTES):
        state_counts = {state: 0 for state in possible_states}
        for agent in agents:
            state_counts[agent.declared_state[k]] += 1

        for state in possible_states:
            if state_counts[state] > 0:
                state_rewards_last_epoch[k][state] = state_rewards[k][state] / state_counts[state]
            else:
                state_rewards_last_epoch[k][state] = state_rewards[k][state]  # No agents in this state
        for agent in agents:
            agent.reward = state_rewards_last_epoch[k][agent.declared_state[k]]
    return state_rewards_last_epoch