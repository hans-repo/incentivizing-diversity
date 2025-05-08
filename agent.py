import random

class Agent:
    def __init__(self, id, possible_states, state_rewards, state_run_costs, state_switch_costs, switch_frequency_param):
        self.id = id
        self.possible_states = possible_states.copy()  # Make a copy to avoid reference issues
        self.state_rewards = state_rewards  # Pass state rewards
        self.state_run_costs = state_run_costs  # Pass state run costs
        self.state_switch_costs = state_switch_costs  # Pass state switch costs
        NUM_ATTRIBUTES = len(state_rewards)
        
        # Set initial state to first non-NO_STATE state
        initial_state = next((state for state in possible_states if state != 'NO_STATE'), 'State_A')
        self.declared_state = [initial_state for _ in range(NUM_ATTRIBUTES)]  # Initial declared state
        self.real_state = [initial_state for _ in range(NUM_ATTRIBUTES)]  # Initial real state
        
        self.reward = 0  # Track the reward received by the agent
        self.personal_switch_cost = 0
        
        # Maintain switch costs for all possible states, including those that might be added later
        self.switch_cost = {state: random.uniform(0.0, 1.0) * self.personal_switch_cost 
                           for state in state_switch_costs.keys()}
        
        self.epochs_since_last_change = [0 for _ in range(NUM_ATTRIBUTES)]  # Track epochs since last state change
        # self.switch_threshold = [random.randint(1, switch_frequency_param) for _ in range(NUM_ATTRIBUTES)]  # Randomly chosen threshold for switching
        self.switch_threshold = [random.random()*switch_frequency_param for _ in range(NUM_ATTRIBUTES)]
        self.switch_cooldown = [random.randint(1, 10) for _ in range(NUM_ATTRIBUTES)]

    def decision(self, state_rewards_last_epoch, malicious=False):
        for k in range(len(self.declared_state)):
            net_gains = {}
            
            # Calculate net gains for all current possible states
            for state in self.possible_states:
                if state == 'NO_STATE':
                    net_gains[state] = 0
                    
                if state == self.real_state[k]:
                    # Already in this state, just pay run cost
                    net_gains[state] = state_rewards_last_epoch[k][state] - self.state_run_costs[state]
                else:
                    # Need to switch to this state, pay switch cost too
                    switch_cost = self.switch_cost.get(state, 0) + self.state_switch_costs.get(state, 0)
                    net_gains[state] = (state_rewards_last_epoch[k][state] -
                                        self.state_run_costs.get(state, 0) -
                                        switch_cost)
            
                
            best_state = random_max(net_gains, key=net_gains.get)
            current_gain = net_gains.get(self.declared_state[k], float('-inf'))
            
            # # Check if it's worth switching and if we've waited long enough
            if net_gains[best_state] < 0:
                self.real_state[k] = 'NO_STATE'
                self.declared_state[k] = 'NO_STATE'
                self.epochs_since_last_change[k] = 0  # Reset the counter after switching
            elif net_gains[best_state] > current_gain + abs(current_gain)*self.switch_threshold[k] and self.epochs_since_last_change[k] >= self.switch_cooldown[k]:
                # Switch state and reset the counter
                self.real_state[k] = best_state
                self.declared_state[k] = best_state
                self.epochs_since_last_change[k] = 0  # Reset the counter after switching
            elif net_gains[best_state] > current_gain + abs(current_gain)*self.switch_threshold[k] and self.epochs_since_last_change[k] < self.switch_cooldown[k]:
                # Increment the counter if no switch is made
                self.epochs_since_last_change[k] += 1
            else:
                self.epochs_since_last_change[k] = 0

            # # Check if it's worth switching and new gains are higher than current + threshold
            # if net_gains[best_state] < 0:
            #     self.real_state[k] = 'NO_STATE'
            #     self.declared_state[k] = 'NO_STATE'
            #     self.epochs_since_last_change[k] = 0  # Reset the counter after switching
            # if net_gains[best_state] > current_gain + abs(current_gain)*self.switch_threshold[k]:
            #     # Switch state and reset the counter
            #     self.real_state[k] = best_state
            #     self.declared_state[k] = best_state
            #     self.epochs_since_last_change[k] = 0  # Reset the counter after switching
            # else:
            #     # Increment the counter if no switch is made
            #     self.epochs_since_last_change[k] += 1

            if malicious:
                total_cost = {}
                for state in self.possible_states:
                    if state == 'NO_STATE':
                        continue  # Skip NO_STATE for malicious behavior too
                        
                    if state == self.declared_state[k]:
                        total_cost[state] = self.state_run_costs.get(state, 0)
                    else:
                        switch_cost = self.switch_cost.get(state, 0) + self.state_switch_costs.get(state, 0)
                        total_cost[state] = self.state_run_costs.get(state, 0) + switch_cost
                
                # Skip if no valid states to choose from
                if not total_cost:
                    continue
                    
                cheapest_state = random_min(total_cost, key=total_cost.get, default=self.declared_state[k])
                self.real_state[k] = cheapest_state

    # Method to handle when new states are added to the system
    def update_possible_states(self, new_possible_states, new_run_costs, new_switch_costs):
        """
        Update the agent's knowledge when new states are added to the system.
        
        Args:
            new_possible_states: The updated list of possible states
            new_run_costs: Dictionary with updated run costs
            new_switch_costs: Dictionary with updated switch costs
        """
        # Update possible states
        self.possible_states = new_possible_states.copy()
        
        # Update switch costs for any new states
        for state in new_possible_states:
            if state not in self.switch_cost:
                self.switch_cost[state] = random.uniform(0.0, 1.0) * self.personal_switch_cost


def random_min(iterable, key=None, default=None):
    if not iterable:
        return default
        
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
    if not iterable:
        return None
        
    if key is None:
        key = lambda x: x  # Default key function (identity function)

    # Find the maximum value based on the key function
    max_value = max(iterable, key=key)

    # Collect all elements that have the maximum value (based on the key function)
    max_elements = [x for x in iterable if key(x) == key(max_value)]

    # Randomly select one of the tied elements
    return random.choice(max_elements)


def generate_agents(n, possible_states, state_rewards, state_run_costs, state_switch_costs, switch_frequency_param):
    return [Agent(i, possible_states, state_rewards, state_run_costs, state_switch_costs, switch_frequency_param) for i in range(n)]


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
                ideal_share = 1/(len(possible_states)-1)  # -1 for NO_STATE
                error = (ideal_share-state_share)
                accumulated_error[k][state] = accumulated_error[k][state] + error
                P_term = adaptive_param*error
                I_term = integral_param*accumulated_error[k][state]
                D_term = derivative_param * (error - last_error[k][state])
                state_rewards[k][state] = P_term + I_term + D_term
                last_error[k][state] = error
    return state_rewards, accumulated_error, last_error


def distribute_rewards(agents, possible_states, state_rewards):
    NUM_ATTRIBUTES = len(state_rewards)
    state_rewards_last_epoch = state_rewards.copy()  # Make a copy to avoid reference issues
    
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