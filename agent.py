import random

class Agent:
    def __init__(self, id, possible_versions, version_rewards, version_run_costs, version_switch_costs, switch_frequency_param):
        self.id = id
        self.possible_versions = possible_versions.copy()  # Make a copy to avoid reference issues
        self.version_rewards = version_rewards  # Pass version rewards
        self.version_run_costs = version_run_costs  # Pass version run costs
        self.version_switch_costs = version_switch_costs  # Pass version switch costs
        NUM_ATTRIBUTES = len(version_rewards)
        
        # Set initial version to first non-NO_version version
        initial_version = next((version for version in possible_versions if version != 'NO_version'), 'version_A')
        self.declared_version = [initial_version for _ in range(NUM_ATTRIBUTES)]  # Initial declared version
        self.real_version = [initial_version for _ in range(NUM_ATTRIBUTES)]  # Initial real version
        
        self.reward = 0  # Track the reward received by the agent
        self.personal_switch_cost = 0
        
        # Maintain switch costs for all possible versions, including those that might be added later
        self.switch_cost = {version: random.uniform(0.0, 1.0) * self.personal_switch_cost 
                           for version in version_switch_costs.keys()}
        
        self.epochs_since_last_change = [0 for _ in range(NUM_ATTRIBUTES)]  # Track epochs since last version change
        # self.switch_threshold = [random.randint(1, switch_frequency_param) for _ in range(NUM_ATTRIBUTES)]  # Randomly chosen threshold for switching
        self.switch_threshold = [random.random()*switch_frequency_param for _ in range(NUM_ATTRIBUTES)]
        self.switch_cooldown = [random.randint(1, 10) for _ in range(NUM_ATTRIBUTES)]

    def decision(self, version_rewards_last_epoch, malicious=False):
        for k in range(len(self.declared_version)):
            net_gains = {}
            
            # Calculate net gains for all current possible versions
            for version in self.possible_versions:
                if version == 'NO_version':
                    net_gains[version] = 0
                    
                if version == self.real_version[k]:
                    # Already in this version, just pay run cost
                    net_gains[version] = version_rewards_last_epoch[k][version] - self.version_run_costs[version]
                else:
                    # Need to switch to this version, pay switch cost too
                    switch_cost = self.switch_cost.get(version, 0) + self.version_switch_costs.get(version, 0)
                    net_gains[version] = (version_rewards_last_epoch[k][version] -
                                        self.version_run_costs.get(version, 0) -
                                        switch_cost)
            
                
            best_version = random_max(net_gains, key=net_gains.get)
            current_gain = net_gains.get(self.declared_version[k], float('-inf'))
            
            # # Check if it's worth switching and if we've waited long enough
            
            if net_gains[best_version] < 0:
                self.real_version[k] = 'NO_version'
                self.declared_version[k] = 'NO_version'
                self.epochs_since_last_change[k] = 0  # Reset the counter after switching
            elif net_gains[best_version] > current_gain + abs(current_gain)*self.switch_threshold[k] and self.epochs_since_last_change[k] >= self.switch_cooldown[k]:
                # Switch version and reset the counter
                self.real_version[k] = best_version
                self.declared_version[k] = best_version
                self.epochs_since_last_change[k] = 0  # Reset the counter after switching
            elif net_gains[best_version] > current_gain + abs(current_gain)*self.switch_threshold[k] and self.epochs_since_last_change[k] < self.switch_cooldown[k]:
                # Increment the counter if no switch is made
                self.epochs_since_last_change[k] += 1
            elif net_gains[self.real_version[k]]<0:
                self.real_version[k] = 'NO_version'
                self.declared_version[k] = 'NO_version'
                self.epochs_since_last_change[k] = 0
            else:
                self.epochs_since_last_change[k] = 0

            if malicious:
                total_cost = {}
                for version in self.possible_versions:
                    if version == 'NO_version':
                        continue  # Skip NO_version for malicious behavior too
                        
                    if version == self.declared_version[k]:
                        total_cost[version] = self.version_run_costs.get(version, 0)
                    else:
                        switch_cost = self.switch_cost.get(version, 0) + self.version_switch_costs.get(version, 0)
                        total_cost[version] = self.version_run_costs.get(version, 0) + switch_cost
                
                # Skip if no valid versions to choose from
                if not total_cost:
                    continue
                    
                cheapest_version = random_min(total_cost, key=total_cost.get, default=self.declared_version[k])
                self.real_version[k] = cheapest_version

    # Method to handle when new versions are added to the system
    def update_possible_versions(self, new_possible_versions, new_run_costs, new_switch_costs):
        """
        Update the agent's knowledge when new versions are added to the system.
        
        Args:
            new_possible_versions: The updated list of possible versions
            new_run_costs: Dictionary with updated run costs
            new_switch_costs: Dictionary with updated switch costs
        """
        # Update possible versions
        self.possible_versions = new_possible_versions.copy()
        
        # Update switch costs for any new versions
        for version in new_possible_versions:
            if version not in self.switch_cost:
                self.switch_cost[version] = random.uniform(0.0, 1.0) * self.personal_switch_cost


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


def generate_agents(n, possible_versions, version_rewards, version_run_costs, version_switch_costs, switch_frequency_param):
    return [Agent(i, possible_versions, version_rewards, version_run_costs, version_switch_costs, switch_frequency_param) for i in range(n)]


def update_version_rewards(agents, current_versions, version_rewards, accumulated_error, last_error, adaptive_param, integral_param, derivative_param):
    """
    Legacy PID update function, kept for backward compatibility.
    This is now refactored in reinforcement_learning.py in the PIDController class.
    """
    for k in range(len(version_rewards)):
        version_counts = {version: 0 for version in current_versions}
        for agent in agents:
            version_counts[agent.declared_version[k]] += 1
        for version in current_versions:
            if version == 'NO_version':
                version_rewards[k][version] = 0
            else: 
                version_share = version_counts[version]/len(agents)
                ideal_share = 1/(len(current_versions)-1)  # -1 for NO_version
                error = (ideal_share-version_share)
                accumulated_error[k][version] = accumulated_error[k][version] + error
                P_term = adaptive_param*error
                I_term = integral_param*accumulated_error[k][version]
                D_term = derivative_param * (error - last_error[k][version])
                version_rewards[k][version] = P_term + I_term + D_term
                last_error[k][version] = error
    return version_rewards, accumulated_error, last_error


def distribute_rewards(agents, current_versions, version_rewards):
    NUM_ATTRIBUTES = len(version_rewards)
    version_rewards_last_epoch = version_rewards.copy()  # Make a copy to avoid reference issues
    
    for k in range(NUM_ATTRIBUTES):
        version_counts = {version: 0 for version in current_versions}
        for agent in agents:
            version_counts[agent.declared_version[k]] += 1

        for version in current_versions:
            if version_counts[version] > 0:
                version_rewards_last_epoch[k][version] = version_rewards[k][version] / version_counts[version]
            else:
                version_rewards_last_epoch[k][version] = version_rewards[k][version]  # No agents in this version
                
        for agent in agents:
            agent.reward = version_rewards_last_epoch[k][agent.declared_version[k]]
            
    return version_rewards_last_epoch