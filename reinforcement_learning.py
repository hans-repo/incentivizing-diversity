import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

class RewardEnvironment:
    """
    Environment for the RL agent that provides state rewards based on agent distributions.
    This environment has a dynamic state space that can change during the simulation.
    Modified for improved stability and reward shaping.
    """
    def __init__(self, possible_states, n_agents):
        self.possible_states = possible_states.copy()
        self.n_agents = n_agents
        self.valid_states = [s for s in possible_states if s != 'NO_STATE']
        self.num_valid_states = len(self.valid_states)
        
        # State is the distribution of agents across the states
        self.state = None
        self.prev_state = None  # Track previous state for reward shaping
        
        # Ideal distribution is even across all valid states
        self.ideal_distribution = self._calculate_ideal_distribution()
        
        # For measuring entropy directly
        self.entropy_history = []
        self.ideal_entropy = self._calculate_ideal_entropy()
        
        # Smoothing factors
        self.reward_smoothing = 0.8  # Exponential moving average factor
        self.smoothed_reward = 0  # Initialize smoothed reward
        
    def _calculate_ideal_distribution(self):
        """Calculate the ideal (even) distribution across all valid states"""
        ideal = {}
        for state in self.possible_states:
            if state == 'NO_STATE':
                ideal[state] = 0
            else:
                ideal[state] = 1 / self.num_valid_states
        return ideal
    
    def _calculate_ideal_entropy(self):
        """Calculate ideal entropy (Shannon entropy) for even distribution"""
        # Only count non-NO_STATE states for entropy calculation
        num_valid = len(self.valid_states)
        if num_valid <= 1:
            return 0.0
        
        p = 1.0 / num_valid  # Equal probability for each state
        return -num_valid * p * np.log2(p)  # Shannon entropy
        
    def _calculate_entropy(self, distribution):
        """Calculate Shannon entropy of current distribution"""
        entropy = 0
        valid_distribution = {s: distribution[s] for s in self.valid_states}
        total = sum(valid_distribution.values())
        
        if total == 0:
            return 0
            
        # Normalize to create a proper probability distribution
        for state in self.valid_states:
            p = valid_distribution[state] / total
            if p > 0:  # Avoid log(0)
                entropy -= p * np.log2(p)
                
        return entropy
    
    def update_states(self, new_states):
        """Update the possible states (e.g., when a new state is added)"""
        self.possible_states = new_states.copy()
        self.valid_states = [s for s in new_states if s != 'NO_STATE']
        self.num_valid_states = len(self.valid_states)
        self.ideal_distribution = self._calculate_ideal_distribution()
        self.ideal_entropy = self._calculate_ideal_entropy()
        
        # Reset state tracking when environment changes
        self.prev_state = self.state
        self.state = None
    
    def observe_state(self, agents):
        """
        Observe the current distribution of agents across states
        
        Args:
            agents: List of Agent objects
            
        Returns:
            Observed state as a normalized distribution dictionary
        """
        state_counts = {state: 0 for state in self.possible_states}
        for agent in agents:
            state_counts[agent.declared_state[0]] += 1  # Using only first attribute for simplicity
        
        # Convert to distribution
        distribution = {}
        for state in self.possible_states:
            distribution[state] = state_counts[state] / self.n_agents
        
        # Store previous state before updating
        self.prev_state = self.state    
        self.state = distribution
        return distribution
    
    def calculate_reward(self, distribution=None):
        """
        Calculate reward based on how close the distribution is to ideal
        With improved reward shaping for stability
        
        Args:
            distribution: Current distribution of agents (if None, use self.state)
            
        Returns:
            reward: Scalar value indicating quality of distribution
            state_rewards: Dictionary of rewards for each state
        """
        if distribution is None:
            distribution = self.state
            
        if distribution is None:
            raise ValueError("No state distribution available")
        
        # Calculate Shannon entropy of current distribution
        current_entropy = self._calculate_entropy(distribution)
        self.entropy_history.append(current_entropy)
        
        # Calculate reward based on entropy (closer to ideal_entropy is better)
        # Normalize to [0, 1] scale
        if self.ideal_entropy > 0:
            entropy_ratio = current_entropy / self.ideal_entropy
            entropy_ratio = min(entropy_ratio, 1.0)  # Cap at 1.0 (ideal)
        else:
            entropy_ratio = 0
            
        # Primary reward based on entropy ratio
        base_reward = entropy_ratio
        
        # Add reward shaping for stability by measuring improvement
        improvement_reward = 0
        if self.prev_state is not None:
            prev_entropy = self._calculate_entropy(self.prev_state)
            improvement = current_entropy - prev_entropy
            
            # Only reward significant improvements to reduce noise
            if improvement > 0.01:
                improvement_reward = 0.1 * np.tanh(improvement * 5)  # Bounded improvement reward
                
        # Combine base reward and improvement reward
        combined_reward = 0.8 * base_reward + 0.2 * improvement_reward
        
        # Apply temporal smoothing for stability
        if self.reward_smoothing > 0:
            if self.smoothed_reward == 0:  # First update
                self.smoothed_reward = combined_reward
            else:
                self.smoothed_reward = self.reward_smoothing * self.smoothed_reward + \
                                      (1 - self.reward_smoothing) * combined_reward
            final_reward = self.smoothed_reward
        else:
            final_reward = combined_reward
            
        # Calculate rewards for each state based on divergence from ideal
        state_rewards = {}
        base_reward_scale = 1000 * self.n_agents  # Base reward size
        
        for state in self.possible_states:
            if state == 'NO_STATE':
                state_rewards[state] = 0
            else:
                # Calculate state-specific reward with more stability
                # - Positive reward for underrepresented states
                # - Negative reward for overrepresented states
                error = self.ideal_distribution[state] - distribution[state]
                
                # Add exponential scaling to heavily penalize large imbalances
                error_sign = np.sign(error)
                error_magnitude = abs(error)
                
                # Scale error using a function that grows faster for large errors
                # This helps prevent state domination
                scaled_error = error_sign * (error_magnitude + 0.1 * error_magnitude**2)
                
                state_rewards[state] = base_reward_scale * scaled_error
                
        return final_reward, state_rewards


class DQNAgent:
    """
    Deep Q-Network agent for learning optimal reward allocation
    Modified for stability with target network, experience replay, 
    and gradient clipping
    """
    def __init__(self, state_size, action_size, learning_rate=0.0005, 
                 discount_factor=0.99, exploration_rate=1.0, 
                 exploration_decay=0.997, min_exploration=0.05,
                 target_update_frequency=100):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)  # Larger replay buffer
        self.learning_rate = learning_rate  # Lower learning rate for stability
        self.gamma = discount_factor  # Higher discount factor for more long-term focus
        self.epsilon = exploration_rate  # Exploration rate
        self.epsilon_decay = exploration_decay  # Slower decay
        self.epsilon_min = min_exploration  # Higher min exploration
        
        # Target network update frequency (in steps)
        self.target_update_freq = target_update_frequency
        self.update_counter = 0
        
        # Create the main model and target model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_model()  # Initialize target model weights
        
        # Add training metrics
        self.loss_history = []
        self.avg_q_values = []
        
    def _build_model(self):
        """Build a simple neural network model using numpy with xavier initialization"""
        # Deeper network for more representation power
        hidden_layer1 = 32
        hidden_layer2 = 32
        
        # Xavier/Glorot initialization for better convergence
        W1 = np.random.randn(self.state_size, hidden_layer1) * np.sqrt(2.0 / (self.state_size + hidden_layer1))
        b1 = np.zeros(hidden_layer1)
        
        W2 = np.random.randn(hidden_layer1, hidden_layer2) * np.sqrt(2.0 / (hidden_layer1 + hidden_layer2))
        b2 = np.zeros(hidden_layer2)
        
        W3 = np.random.randn(hidden_layer2, self.action_size) * np.sqrt(2.0 / (hidden_layer2 + self.action_size))
        b3 = np.zeros(self.action_size)
        
        # Learning rate for SGD
        self.lr = self.learning_rate
        
        return {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2,
            'W3': W3,
            'b3': b3
        }
    
    def _leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU for better gradient flow"""
        return np.maximum(alpha * x, x)
    
    def _leaky_relu_derivative(self, x, alpha=0.01):
        """Derivative of leaky ReLU"""
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx
    
    def _forward(self, state, model=None):
        """Forward pass through the network"""
        if model is None:
            model = self.model
            
        # Store activations for backprop if using main model
        if model == self.model:
            # First layer
            self.z1 = np.dot(state, model['W1']) + model['b1']
            self.a1 = self._leaky_relu(self.z1)
            
            # Second layer
            self.z2 = np.dot(self.a1, model['W2']) + model['b2']
            self.a2 = self._leaky_relu(self.z2)
            
            # Output layer
            self.z3 = np.dot(self.a2, model['W3']) + model['b3']
            
            return self.z3  # Q-values
        else:
            # When using target network, we don't need to store activations
            a1 = self._leaky_relu(np.dot(state, model['W1']) + model['b1'])
            a2 = self._leaky_relu(np.dot(a1, model['W2']) + model['b2'])
            return np.dot(a2, model['W3']) + model['b3']
    
    def _backward(self, state, target):
        """Backward pass with gradient clipping for stability"""
        # Compute gradients
        dz3 = self.z3 - target
        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0)
        
        da2 = np.dot(dz3, self.model['W3'].T)
        dz2 = da2 * self._leaky_relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)
        
        da1 = np.dot(dz2, self.model['W2'].T)
        dz1 = da1 * self._leaky_relu_derivative(self.z1)
        dW1 = np.dot(state.T, dz1)
        db1 = np.sum(dz1, axis=0)
        
        # Calculate loss for tracking
        loss = np.mean(np.square(dz3))
        self.loss_history.append(loss)
        
        # Gradient clipping to prevent exploding gradients
        max_grad_norm = 1.0
        for grad in [dW1, db1, dW2, db2, dW3, db3]:
            norm = np.sqrt(np.sum(np.square(grad)))
            if norm > max_grad_norm:
                grad *= max_grad_norm / norm
        
        # Update weights with clipped gradients
        self.model['W3'] -= self.lr * dW3
        self.model['b3'] -= self.lr * db3
        self.model['W2'] -= self.lr * dW2
        self.model['b2'] -= self.lr * db2
        self.model['W1'] -= self.lr * dW1
        self.model['b1'] -= self.lr * db1
        
        # Return loss for monitoring
        return loss
    
    def _update_target_model(self):
        """Copy main model weights to target model"""
        for key in self.model:
            self.target_model[key] = self.model[key].copy()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        # Clip rewards for stability
        clipped_reward = np.clip(reward, -1, 1)
        self.memory.append((state, action, clipped_reward, next_state, done))
    
    def act(self, state):
        """Act based on the current state with epsilon-greedy policy and some state normalization"""
        # Normalize state for better stability
        normalized_state = state / (np.max(np.abs(state)) + 1e-10)
        
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        q_values = self._forward(normalized_state)
        
        # Track average Q-values for monitoring learning progress
        self.avg_q_values.append(np.mean(q_values))
        
        return np.argmax(q_values[0])
    
    def replay(self, batch_size):
        """Train on random batch from memory using target network for stability"""
        if len(self.memory) < batch_size:
            return 0  # Return 0 loss if not enough samples
        
        # Sample random minibatch
        minibatch = random.sample(self.memory, batch_size)
        
        total_loss = 0
        
        for state, action, reward, next_state, done in minibatch:
            # Normalize states
            normalized_state = state / (np.max(np.abs(state)) + 1e-10)
            normalized_next_state = next_state / (np.max(np.abs(next_state)) + 1e-10)
            
            # Double DQN: use main network to select action, target network to evaluate it
            next_action = np.argmax(self._forward(normalized_next_state)[0])
            
            # Get target Q value using the target network
            target = reward
            if not done:
                target_q = self._forward(normalized_next_state, self.target_model)[0][next_action]
                target += self.gamma * target_q
            
            # Get current Q values and update the target for the selected action
            target_f = self._forward(normalized_state)
            original_val = target_f[0][action]
            target_f[0][action] = target
            
            # Use Huber loss (smoother than MSE) via backpropagation
            loss = self._backward(normalized_state, target_f)
            total_loss += loss
            
            # Update target network periodically
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self._update_target_model()
                print(f"Target network updated. Current ε: {self.epsilon:.4f}")
            
        # More conservative epsilon decay based on learning progress
        if self.epsilon > self.epsilon_min:
            # Adaptive decay: slower when loss is high (unstable), faster when loss is low (stable)
            decay_rate = self.epsilon_decay * (1.0 + 0.1 * np.exp(-total_loss/batch_size))
            self.epsilon = max(self.epsilon_min, self.epsilon * decay_rate)
            
        return total_loss / batch_size  # Return average loss


class RewardLearner:
    """
    RL agent that learns to allocate rewards to different states to achieve
    optimal diversity in the multi-agent system.
    Modified to control each state individually similar to PID in agent.py.
    """
    def __init__(self, possible_states, n_agents, reward_scale=1000):
        self.env = RewardEnvironment(possible_states, n_agents)
        self.possible_states = possible_states.copy()
        self.n_agents = n_agents
        self.valid_states = [s for s in possible_states if s != 'NO_STATE']
        
        # Create a separate RL agent for each valid state
        # Each agent will have a simpler action space (adjusting rewards up/down)
        self.state_agents = {}
        for state in self.valid_states:
            # For each state, the agent receives: 
            # 1. Current distribution across all states (state_size)
            # 2. Error for this specific state from ideal
            # 3. Previous reward value for this state
            state_input_size = len(possible_states) + 2
            
            # Simple action space: adjust reward up or down by varying amounts
            # Actions: strong decrease, medium decrease, slight decrease, no change,
            #          slight increase, medium increase, strong increase
            state_action_size = 7
            
            self.state_agents[state] = DQNAgent(
                state_size=state_input_size, 
                action_size=state_action_size,
                learning_rate=0.0003,  # Lower learning rate for stability
                discount_factor=0.99,
                exploration_rate=1.0,
                exploration_decay=0.998,
                min_exploration=0.05
            )
            
        # Reward scale (similar to base rewards in the PID version)
        self.reward_scale = reward_scale * n_agents
        
        # Mapping from actions to reward adjustments (percentage change)
        self.reward_adjustments = np.array([
            -0.5,   # Strong decrease (-50%)
            -0.2,   # Medium decrease (-20%)
            -0.05,  # Slight decrease (-5%)
            0.0,    # No change
            0.05,   # Slight increase (+5%)
            0.2,    # Medium increase (+20%)
            0.5     # Strong increase (+50%)
        ])
        
        # Keep track of current rewards for each state
        self.current_rewards = {state: 0.0 for state in possible_states}
        
        # PID-like error tracking
        self.accumulated_error = {state: 0.0 for state in possible_states}
        self.last_error = {state: 0.0 for state in possible_states}
        
        # For monitoring learning progress
        self.total_adjustments = 0
        self.training_losses = []
        
    def update_possible_states(self, new_states):
        """Update when new states are added"""
        old_states = self.possible_states.copy()
        self.env.update_states(new_states)
        
        # Update internal state tracking
        self.possible_states = new_states.copy()
        self.valid_states = [s for s in new_states if s != 'NO_STATE']
        
        # Add new states to reward and error tracking
        for state in new_states:
            if state not in self.current_rewards:
                self.current_rewards[state] = 0.0
                self.accumulated_error[state] = 0.0
                self.last_error[state] = 0.0
        
        # Create agents for new states
        new_valid_states = [s for s in new_states if s not in old_states and s != 'NO_STATE']
        for state in new_valid_states:
            state_input_size = len(new_states) + 2
            state_action_size = 7
            
            print(f"Creating new RL agent for state: {state}")
            self.state_agents[state] = DQNAgent(
                state_size=state_input_size, 
                action_size=state_action_size,
                learning_rate=0.0003,
                discount_factor=0.99,
                exploration_rate=0.5,  # Start with moderate exploration
                exploration_decay=0.998,
                min_exploration=0.05
            )
            
        # Preserve existing agents but update their state input size if needed
        if len(old_states) != len(new_states):
            for state in self.valid_states:
                if state in self.state_agents and state not in new_valid_states:
                    # Existing agent needs to be updated for new state size
                    # In a real implementation, we'd preserve the weights and update the network architecture
                    # Here we'll just note that this would be needed
                    print(f"Note: Agent for {state} would need architecture update for new state size")
                    # In practice we'd need to handle this properly
    
    def _state_to_agent_input(self, state_dict, target_state):
        """
        Convert distribution dictionary to input vector for a specific state's agent
        
        Args:
            state_dict: Current distribution across all states
            target_state: The specific state this input is for
            
        Returns:
            Input vector including distribution, error, and current reward
        """
        # Get current error from ideal
        error = self.env.ideal_distribution[target_state] - state_dict[target_state]
        
        # Create input vector: all state distributions + error + current reward
        distribution_values = [state_dict[state] for state in self.env.possible_states]
        
        # Normalized current reward (-1 to 1 range)
        normalized_reward = np.tanh(self.current_rewards[target_state] / self.reward_scale)
        
        # Combine into input vector
        input_vector = distribution_values + [error, normalized_reward]
        return np.array([input_vector])
    
    def update_state_rewards(self, agents):
        """
        Calculate state rewards based on current agent distribution.
        Similar to update_state_rewards in agent.py but using RL.
        
        Args:
            agents: List of agents
            
        Returns:
            Updated state rewards, accumulated error, and last error
        """
        # Get current distribution
        current_distribution = self.env.observe_state(agents)
        
        rewards_adjustment_info = {}  # For logging
        
        # Update rewards for each valid state using its dedicated agent
        for state in self.valid_states:
            # Skip NO_STATE
            if state == 'NO_STATE':
                self.current_rewards[state] = 0
                continue
                
            # Calculate current error (deviation from ideal)
            ideal_share = self.env.ideal_distribution[state]
            current_share = current_distribution[state]
            error = ideal_share - current_share
            
            # Update error tracking (similar to PID controller)
            self.accumulated_error[state] += error
            
            # Prepare input for this state's agent
            agent_input = self._state_to_agent_input(current_distribution, state)
            
            # Get action from agent (which adjustment to make)
            agent = self.state_agents[state]
            action = agent.act(agent_input)
            
            # Apply the selected adjustment to current reward
            adjustment = self.reward_adjustments[action]
            
            # Base factor related to state deviation (like P term in PID)
            base_factor = abs(error) * self.reward_scale
            
            # Integrate the adjustment with current reward (with sign of error)
            reward_delta = adjustment * base_factor * np.sign(error)
            
            # Update current reward
            old_reward = self.current_rewards[state]
            self.current_rewards[state] += reward_delta
            
            # Apply stabilization: limit maximum change and use absolute min/max
            # Prevent wild fluctuations in reward
            max_change = 0.2 * abs(old_reward) + 0.1 * self.reward_scale
            if abs(reward_delta) > max_change:
                reward_delta = np.sign(reward_delta) * max_change
                self.current_rewards[state] = old_reward + reward_delta
            
            # Apply bounds to prevent extreme rewards
            max_reward = 10 * self.reward_scale
            self.current_rewards[state] = np.clip(self.current_rewards[state], -max_reward, max_reward)
            
            # Store info for training
            rewards_adjustment_info[state] = {
                'error': error,
                'action': action,
                'adjustment': adjustment,
                'reward_delta': reward_delta,
                'new_reward': self.current_rewards[state]
            }
            
            # Calculate environment reward for this state (to train the agent)
            # Reward is better when error is reduced
            if state in self.last_error:
                error_change = abs(self.last_error[state]) - abs(error)
                # Positive reward for error reduction, negative for increase
                env_reward = np.tanh(error_change * 10)
                
                # If error is very small, give bonus reward
                if abs(error) < 0.05:
                    env_reward += 0.5
                
                # Remember state for next update
                next_state_input = self._state_to_agent_input(current_distribution, state)
                
                # Store experience in agent's memory
                agent.remember(agent_input, action, env_reward, next_state_input, False)
                
                # Train on batches periodically
                self.total_adjustments += 1
                if self.total_adjustments % 10 == 0:  # Train every 10 adjustments
                    loss = agent.replay(min(32, len(agent.memory)))
                    if loss > 0:
                        self.training_losses.append(loss)
            
            # Update last error for next iteration
            self.last_error[state] = error
            
        # For debugging
        if len(self.training_losses) > 0 and len(self.training_losses) % 100 == 0:
            avg_loss = sum(self.training_losses[-100:]) / 100
            print(f"Average loss (last 100): {avg_loss:.6f}")
            
        # Return the current rewards in the proper format for the simulation
        # This already tracks accumulated errors internally similar to agent.py
        return self.current_rewards, self.accumulated_error, self.last_error
    
    def format_state_rewards(self, raw_rewards, num_attributes=1):
        """
        Format state rewards to match the expected format in the simulation
        
        Args:
            raw_rewards: Dictionary of rewards from RL agent
            num_attributes: Number of attributes in the system
            
        Returns:
            List of dictionaries in the format expected by the simulation
        """
        formatted_rewards = [{state: raw_rewards[state] for state in raw_rewards} 
                             for _ in range(num_attributes)]
        return formatted_rewards
    
    def get_exploration_rates(self):
        """Get current exploration rates for all agents"""
        return {state: agent.epsilon for state, agent in self.state_agents.items()}


def distribute_rewards_rl(agents, possible_states, state_rewards):
    """
    Distribute rewards to agents based on their states.
    This mimics the distribute_rewards function in agent.py but works with RL rewards.
    
    Args:
        agents: List of Agent objects
        possible_states: List of possible states
        state_rewards: List of dictionaries of rewards for each attribute
        
    Returns:
        Updated state_rewards_last_epoch
    """
    NUM_ATTRIBUTES = len(state_rewards)
    state_rewards_last_epoch = []
    
    # Create a deep copy to avoid modifying original
    for k in range(NUM_ATTRIBUTES):
        state_rewards_last_epoch.append({state: state_rewards[k][state] for state in state_rewards[k]})
    
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


def plot_exploration_rate(exploration_rates, epochs):
    """
    Plot the exploration rate over time
    
    Args:
        exploration_rates: List of epsilon values
        epochs: Number of epochs
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), exploration_rates, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Exploration Rate (ε)')
    plt.title('Exploration Rate Decay Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    return plt


def plot_rl_rewards(rewards_history, epochs):
    """
    Plot the RL rewards over time
    
    Args:
        rewards_history: List of rewards
        epochs: Number of epochs
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), rewards_history, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('RL Agent Reward')
    plt.title('RL Agent Reward Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt