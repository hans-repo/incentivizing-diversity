import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

# Define experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer to store and sample experiences"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.state_dim = None  # Track state dimension
        self.action_dim = None  # Track action dimension
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        # Check and update dimensions on first addition
        if len(self.buffer) == 0:
            self.state_dim = len(state)
            if isinstance(action, (np.ndarray, list)):
                self.action_dim = len(action)
            else:
                self.action_dim = 1  # Scalar action
            
        # Skip adding experiences if dimensions don't match
        if len(state) != self.state_dim:
            print(f"Skipping experience with mismatched state dimension: got {len(state)}, expected {self.state_dim}")
            return
            
        if isinstance(action, (np.ndarray, list)) and len(action) != self.action_dim:
            print(f"Skipping experience with mismatched action dimension: got {len(action)}, expected {self.action_dim}")
            return
            
        # Make a copy of state and next_state to ensure they don't get modified
        state_copy = state.copy() if isinstance(state, (list, np.ndarray)) else state
        next_state_copy = next_state.copy() if isinstance(next_state, (list, np.ndarray)) else next_state
        
        # Make a copy of action to ensure it doesn't get modified
        if isinstance(action, (list, np.ndarray)):
            action_copy = action.copy()
        else:
            action_copy = action
            
        experience = Experience(state_copy, action_copy, reward, next_state_copy, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Randomly sample batch_size experiences from buffer"""
        if len(self.buffer) < batch_size:
            # Return None if not enough samples
            return None
            
        try:
            experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
            
            # Convert experiences to numpy arrays first for safer handling
            states = np.array([list(e.state) for e in experiences])  # Ensure state is listified
            
            # Handle actions - make sure they're all the same type (list or scalar)
            if isinstance(experiences[0].action, (np.ndarray, list)):
                actions = np.array([list(e.action) for e in experiences])
            else:
                # For scalar actions
                actions = np.array([[e.action] for e in experiences])
                
            rewards = np.array([[e.reward] for e in experiences])
            next_states = np.array([list(e.next_state) for e in experiences])  # Ensure state is listified
            dones = np.array([[e.done] for e in experiences])
            
            # Convert to tensors
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.FloatTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)
            next_states_tensor = torch.FloatTensor(next_states)
            dones_tensor = torch.FloatTensor(dones)
            
            return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
            
        except (ValueError, RuntimeError) as e:
            print(f"Error sampling from replay buffer: {e}")
            import traceback
            traceback.print_exc()
            # Clear buffer if we encounter errors
            self.clear()
            return None
    
    def __len__(self):
        return len(self.buffer)
        
    def clear(self):
        """Clear the replay buffer"""
        self.buffer.clear()
        self.state_dim = None
        self.action_dim = None


class DQNModel(nn.Module):
    """Deep Q-Network model"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize network with smaller weights for more gradual changes
        self._initialize_weights()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Use tanh to constrain output between -1 and 1 for smoother actions
        return torch.tanh(self.fc3(x))
    
    def _initialize_weights(self):
        """Initialize weights with smaller values for more cautious initial behavior"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Lower gain for smaller values
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class RewardController:
    """Base class for reward controllers"""
    
    def __init__(self, states, attribute_idx=0):
        self.states = states
        self.attribute_idx = attribute_idx
    
    def update_rewards(self, agents, current_states, state_rewards, *args, **kwargs):
        """Update rewards based on current state distribution"""
        raise NotImplementedError


class PIDController(RewardController):
    """PID controller for reward adjustment"""
    
    def __init__(self, states, p_param, i_param, d_param, attribute_idx=0):
        super(PIDController, self).__init__(states, attribute_idx)
        self.p_param = p_param
        self.i_param = i_param
        self.d_param = d_param
        self.accumulated_error = {state: 0 for state in states}
        self.last_error = {state: 0 for state in states}
    
    def update_rewards(self, agents, current_states, state_rewards, *args, **kwargs):
        """Update rewards using PID control"""
        state_counts = {state: 0 for state in current_states}
        for agent in agents:
            state_counts[agent.declared_state[self.attribute_idx]] += 1
        
        for state in current_states:
            if state == 'NO_STATE':
                state_rewards[self.attribute_idx][state] = 0
            else:
                state_share = state_counts[state] / len(agents)
                ideal_share = 1 / (len(current_states) - 1)  # -1 for NO_STATE
                error = (ideal_share - state_share)
                
                # Update accumulated error if state exists in it, otherwise initialize it
                if state in self.accumulated_error:
                    self.accumulated_error[state] += error
                else:
                    self.accumulated_error[state] = error
                    
                # Get last error, default to 0 if not found
                last_err = self.last_error.get(state, 0)
                
                # PID terms
                p_term = self.p_param * error
                i_term = self.i_param * self.accumulated_error[state]
                d_term = self.d_param * (error - last_err)
                
                # Update rewards
                state_rewards[self.attribute_idx][state] = p_term + i_term + d_term
                
                # Store current error as last error for next iteration
                self.last_error[state] = error
        
        return state_rewards


class RLPIDController(RewardController):
    """RL-based PID parameter tuner - adaptively learns reward scale and PID parameters"""
    
    def __init__(self, states, n_agents, initial_p=0, initial_i=0, initial_d=0,
                 epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, gamma=0.99, learning_rate=0.001, batch_size=32, 
                 update_frequency=10, target_update_frequency=100, attribute_idx=0,
                 action_scale=0.1, reward_efficiency_weight=0.3,
                 p_scale_factor=None, i_scale_factor=None, d_scale_factor=None):
        super(RLPIDController, self).__init__(states, attribute_idx)
        
        # RL parameters
        self.n_agents = n_agents
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma  # Discount factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        self.action_scale = action_scale  # Control the magnitude of PID parameter adjustments
        
        # Weight for balancing diversity vs efficiency in reward function
        # Higher values give more weight to minimizing total rewards
        self.reward_efficiency_weight = reward_efficiency_weight
        
        # PID parameters (initialized to zeros or passed values)
        self.p_param = initial_p
        self.i_param = initial_i
        self.d_param = initial_d
        self.prev_pid_params = (initial_p, initial_i, initial_d)
        
        # Dynamic reward scale that the RL controller will learn
        self.reward_scale = 100.0  # Initial small estimate that will be adjusted
        
        # Scale factors for P, I, D relative to reward_scale (now configurable)
        self.p_scale_factor = p_scale_factor if p_scale_factor is not None else 10.0
        self.i_scale_factor = i_scale_factor if i_scale_factor is not None else 1.0
        self.d_scale_factor = d_scale_factor if d_scale_factor is not None else 1.0
        
        # Initial adaptive bounds that will adjust during training
        self.max_p = self.reward_scale * self.p_scale_factor
        self.max_i = self.reward_scale * self.i_scale_factor
        self.max_d = self.reward_scale * self.d_scale_factor
        
        # PID state tracking
        self.accumulated_error = {state: 0 for state in states}
        self.last_error = {state: 0 for state in states}
        
        # Anti-windup for I term - prevent integral accumulation from getting too large
        self.i_term_max = self.reward_scale * 5  # Initial value, will be adjusted
        
        # Track current agent distribution for state representation
        self.agent_counts = {state: 0 for state in states}
        
        # Define state and action dimensions
        # State includes: agent distribution + current PID parameters + current reward scale
        self.state_dim = len(states) + 4  # +3 for current P, I, D values, +1 for reward scale
        
        # Actions: adjustments to P, I, D, and reward_scale
        self.action_dim = 4  # One action dimension for each PID parameter + reward scale
        
        # Initialize models
        self.policy_net = DQNModel(self.state_dim, self.action_dim)
        self.target_net = DQNModel(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Track episodes and steps
        self.steps_done = 0
        self.epoch = 0
        
        # Store previous state
        self.prev_state = None
        self.prev_action = None
        
        # Store state mapping (index to state name)
        self.state_to_idx = {state: i for i, state in enumerate(states)}
        self.idx_to_state = {i: state for i, state in enumerate(self.states)}
        
        # Previous diversity for reward calculation
        self.prev_diversity = 0
        self.ideal_diversity = 0
        
        # Track total rewards for efficiency calculation
        self.prev_total_rewards = 0
        self.total_rewards_history = deque(maxlen=10)
        
        # Track max observed rewards to adaptively set bounds
        self.max_observed_reward = self.reward_scale
        self.reward_history = []
        
        # Performance tracking
        self.last_diversity_ratios = deque(maxlen=10)  # Track recent diversity performance
        self.best_diversity_ratio = 0  # Best diversity ratio achieved
        self.best_pid_params = (initial_p, initial_i, initial_d)  # Best parameters found
        self.best_reward_scale = self.reward_scale  # Best reward scale found
    
    def update_adaptive_bounds(self, current_rewards):
        """Update adaptive bounds for PID parameters based on observed rewards"""
        # Extract the maximum absolute reward value in the current state
        current_max_reward = max([abs(r) for r in current_rewards.values() if r != 0], default=0)
        
        # No artificial caps on rewards
        self.max_observed_reward = max(self.max_observed_reward, current_max_reward)
            
        # Keep a short history of recent max rewards
        self.reward_history.append(current_max_reward)
        if len(self.reward_history) > 10:  # Keep only recent history
            self.reward_history.pop(0)
            
        # Calculate average of recent max rewards for stability
        avg_max_reward = sum(self.reward_history) / max(1, len(self.reward_history))
        
        # Dynamically adjust max bounds based on observed rewards and current reward_scale
        self.max_p = max(self.max_p, avg_max_reward * self.p_scale_factor)
        self.max_i = max(self.max_i, avg_max_reward * self.i_scale_factor)
        self.max_d = max(self.max_d, avg_max_reward * self.d_scale_factor)
        
        # Update I-term max based on current reward scale
        self.i_term_max = self.reward_scale * 5
        
        # Log if bounds have been updated significantly
        if self.steps_done % 100 == 0:
            print(f"Adaptive bounds: max_p={self.max_p:.1f}, max_i={self.max_i:.1f}, max_d={self.max_d:.1f}, reward_scale={self.reward_scale:.1f}")
    
    def get_state_representation(self, agents, current_states):
        """Create state representation: [agent distribution, P, I, D, reward_scale]"""
        # Get agent distribution
        state_counts = {state: 0 for state in current_states}
        for agent in agents:
            state_counts[agent.declared_state[self.attribute_idx]] += 1
        
        # Convert to distribution (percentage in each state)
        state_vector = np.zeros(len(self.state_to_idx))
        
        # Ensure we only include states that we know about in our state representation
        for state, count in state_counts.items():
            if state in self.state_to_idx:  # Only include known states
                state_vector[self.state_to_idx[state]] = count / self.n_agents
        
        # Use logarithmic normalization for PID parameters to handle a wide range of values
        # Add a small constant to avoid log(0)
        norm_p = np.log1p(self.p_param) / np.log1p(self.max_p) if self.max_p > 0 else 0
        norm_i = np.log1p(self.i_param) / np.log1p(self.max_i) if self.max_i > 0 else 0
        norm_d = np.log1p(self.d_param) / np.log1p(self.max_d) if self.max_d > 0 else 0
        
        # Normalize reward scale (logarithmically)
        norm_reward_scale = np.log1p(self.reward_scale) / np.log1p(self.max_observed_reward * 10) if self.max_observed_reward > 0 else 0.5
        
        # Clip to [0, 1] range for safety
        norm_p = np.clip(norm_p, 0, 1)
        norm_i = np.clip(norm_i, 0, 1)
        norm_d = np.clip(norm_d, 0, 1)
        norm_reward_scale = np.clip(norm_reward_scale, 0, 1)
        
        # Combine state vector with normalized parameters
        full_state = np.append(state_vector, [norm_p, norm_i, norm_d, norm_reward_scale])
        
        # Verify that the state vector has the expected dimension
        if len(full_state) != self.state_dim:
            print(f"Warning: State dimension mismatch in get_state_representation. Got {len(full_state)}, expected {self.state_dim}")
            print(f"State vector length: {len(state_vector)}, state_to_idx length: {len(self.state_to_idx)}")
            # Adjust the state vector to match expected dimension
            if len(full_state) < self.state_dim:
                # Pad with zeros if too short
                full_state = np.pad(full_state, (0, self.state_dim - len(full_state)), 'constant')
            else:
                # Truncate if too long
                full_state = full_state[:self.state_dim]
        
        return full_state.tolist()  # Convert to list for consistent serialization
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy with conservative bounds"""
        # Dramatically reduce exploration when diversity is already good
        effective_epsilon = self.epsilon
        if random.random() < effective_epsilon:
            # Exploration: choose random action but with limited range
            bound = 0.3  # Limit random exploration to smaller values (-0.3 to 0.3)
            return np.array([random.uniform(-bound, bound) for _ in range(self.action_dim)])
        else:
            # Exploitation: choose best action according to policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                # Output already constrained by tanh in the network
                return self.policy_net(state_tensor).squeeze().numpy()
    
    def update_model(self):
        """Update the policy network using a batch of experiences"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            # If sampling failed, skip update
            return
            
        states, actions, rewards, next_states, dones = batch
        
        try:
            # Compute current Q values
            current_q_values = self.policy_net(states)
            
            # Compute next Q values using target network
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute loss for the entire action space
            loss = F.mse_loss(current_q_values, target_q_values.repeat(1, current_q_values.shape[1]))
            
            # Update policy network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        except Exception as e:
            print(f"Error in update_model: {e}")
            import traceback
            traceback.print_exc()
            # Clear the replay buffer if we encounter errors during training
            self.replay_buffer.clear()
    
    # Here are the truly adaptive changes needed in the RLPIDController class
    # This approach avoids all hardcoded values for reward scaling and penalties

    # 1. Relative penalty calculation based on reward ratio only
    # Corrected reward function that properly incentivizes minimizing total rewards
    def calculate_reward(self, current_diversity, ideal_diversity, current_total_rewards):
        """Calculate reward with explicit stopping signal at optimal diversity"""
        # Normalize diversity to [0, 1] range
        diversity_ratio = current_diversity / ideal_diversity if ideal_diversity > 0 else 0
        
        # Target diversity threshold - using exactly 1.0 for "perfect" diversity
        target_diversity = 1.0
        near_optimal_threshold = 0.98  # Consider diversity near-optimal above this value
        
        # 1. BASE REWARD - reward for diversity progress, caps at target
        base_reward = min(diversity_ratio, target_diversity)
        
        # # 2. PARAMETER CHANGE DETECTION
        # # Store previous parameters if not already tracked
        # if not hasattr(self, 'prev_pid_params'):
        #     self.prev_pid_params = (self.p_param, self.i_param, self.d_param)
        
        # # Calculate parameter changes
        # p_change = abs(self.p_param - self.prev_pid_params[0])
        # i_change = abs(self.i_param - self.prev_pid_params[1])
        # d_change = abs(self.d_param - self.prev_pid_params[2])
        # total_change = p_change + i_change + d_change
        
        # # 3. STABILITY PENALTY - penalize any parameter changes when diversity is near-optimal
        # stability_penalty = 0
        # if diversity_ratio >= near_optimal_threshold:
        #     # Stronger penalty as diversity gets closer to perfect
        #     penalty_factor = (diversity_ratio - near_optimal_threshold) / (target_diversity - near_optimal_threshold)
        #     # Apply penalty proportional to the magnitude of parameter changes
        #     stability_penalty = -0.5 * penalty_factor * total_change

        # Update previous parameters for next step
        self.prev_pid_params = (self.p_param, self.i_param, self.d_param)

        # # High rewards penalty
        # total_pid = self.p_param + self.i_param + self.d_param
        # efficiency_penalty = (total_pid**diversity_ratio)
        
        # Combine rewards and penalties
        total_reward = (base_reward )
        
        # print("RL total reward", total_reward)
        return total_reward
    # 2. Fully relative reward scale adjustment
    def apply_actions_to_pid_parameters(self, actions):
        """Apply actions to adjust PID parameters and reward scale"""
        # Extract actions
        p_adjustment, i_adjustment, d_adjustment, scale_adjustment = actions
        
        # Use the action scale parameter as a base adjustment factor
        # This avoids hardcoded adjustment values
        base_adjustment = self.action_scale
        
        # Calculate parameter changes relative to current values
        # This allows the system to adapt to any scale without hardcoded thresholds
        p_change = p_adjustment * base_adjustment 
        i_change = i_adjustment * base_adjustment 
        d_change = d_adjustment * base_adjustment 
        
        # Calculate reward scale change relative to current scale
        # Adding a small constant prevents getting stuck at zero
        reward_scale_change = scale_adjustment * base_adjustment
        
        # Get current parameters
        old_p, old_i, old_d = self.p_param, self.i_param, self.d_param
        old_reward_scale = self.reward_scale
        
        # Apply changes with constraints to keep values non-negative
        new_p = max(0, old_p + p_change)
        new_i = max(0, old_i + i_change)
        new_d = max(0, old_d + d_change)
        
        # Apply reward scale change - keep positive
        new_reward_scale = max(0.1, old_reward_scale + reward_scale_change)
        
        # Update parameters
        self.p_param, self.i_param, self.d_param = new_p, new_i, new_d
        self.reward_scale = new_reward_scale
        
        # Log significant changes
        significant_change = (
            abs(p_change) > 0.05 * (old_p + 1.0) or 
            abs(i_change) > 0.05 * (old_i + 1.0) or
            abs(d_change) > 0.05 * (old_d + 1.0) or
            abs(reward_scale_change) > 0.05 * (old_reward_scale + 1.0)
        )
        
        if significant_change or self.steps_done % 100 == 0:
            print(f"Parameters updated: P: {old_p:.1f} -> {new_p:.1f}, "
                f"I: {old_i:.1f} -> {new_i:.1f}, D: {old_d:.1f} -> {new_d:.1f}, "
                f"Reward Scale: {old_reward_scale:.1f} -> {new_reward_scale:.1f}")

    # 3. Adaptive bounds based on observed rewards
    def update_adaptive_bounds(self, current_rewards):
        """Update adaptive bounds for PID parameters based on observed rewards"""
        # Extract the maximum absolute reward value in the current state
        current_max_reward = max([abs(r) for r in current_rewards.values() if r != 0], default=0)
        
        # Track the max observed reward - this adapts to the system's natural scale
        self.max_observed_reward = max(self.max_observed_reward, current_max_reward)
            
        # Keep a short history of recent max rewards
        self.reward_history.append(current_max_reward)
        if len(self.reward_history) > 10:  # Keep only recent history
            self.reward_history.pop(0)
            
        # Calculate average of recent max rewards for stability
        avg_max_reward = sum(self.reward_history) / max(1, len(self.reward_history))
        
        # Update bounds based on observed rewards
        # This lets the system discover appropriate bounds without hardcoded values
        self.max_p = max(self.max_p, avg_max_reward * self.p_scale_factor)
        self.max_i = max(self.max_i, avg_max_reward * self.i_scale_factor)
        self.max_d = max(self.max_d, avg_max_reward * self.d_scale_factor)
        
        # Update I-term max based on current observed rewards
        # This allows the I term to adapt to the system's natural scale
        self.i_term_max = max(1.0, self.max_observed_reward * 2.0)
        
        # Log the bounds periodically
        if self.steps_done % 100 == 0:
            print(f"Adaptive bounds: max_p={self.max_p:.1f}, max_i={self.max_i:.1f}, "
                f"max_d={self.max_d:.1f}, max_observed_reward={self.max_observed_reward:.1f}")
    
    def update_rewards_using_pid(self, agents, current_states, state_rewards):
        """Apply PID control to update rewards based on current parameters"""
        state_counts = {state: 0 for state in current_states}
        for agent in agents:
            state_counts[agent.declared_state[self.attribute_idx]] += 1
        
        # Reset accumulated error if it gets too large (anti-windup)
        for state, error in self.accumulated_error.items():
            if abs(error) > self.i_term_max / max(0.001, self.i_param):
                self.accumulated_error[state] = np.sign(error) * self.i_term_max / max(0.001, self.i_param)
        
        # Track total allocated rewards
        total_rewards = 0
        
        # Check if we're in a state transition period
        in_transition = hasattr(self, 'in_transition') and self.in_transition
        
        # Update transition progress if we're in transition mode
        if in_transition:
            self.transition_progress += 1.0 / self.transition_epochs
            if self.transition_progress >= 1.0:
                # Transition complete
                self.in_transition = False
                self.transition_progress = 1.0
        
        for state in current_states:
            if state == 'NO_STATE':
                state_rewards[self.attribute_idx][state] = 0
            else:
                # Calculate regular PID control values
                state_share = state_counts[state] / len(agents)
                ideal_share = 1 / (len(current_states) - 1)  # -1 for NO_STATE
                error = (ideal_share - state_share)
                
                # Update accumulated error if state exists in it, otherwise initialize it
                if state in self.accumulated_error:
                    self.accumulated_error[state] += error
                else:
                    self.accumulated_error[state] = error
                    
                # Get last error, default to 0 if not found
                last_err = self.last_error.get(state, 0)
                
                # Apply anti-windup for I term - still needed for numerical stability
                if abs(self.accumulated_error[state]) > self.i_term_max / max(0.001, self.i_param):
                    self.accumulated_error[state] = np.sign(self.accumulated_error[state]) * self.i_term_max / max(0.001, self.i_param)
                
                # PID terms
                p_term = self.p_param * error
                i_term = self.i_param * self.accumulated_error[state]
                d_term = self.d_param * (error - last_err)
                
                # Calculate new reward based on PID control
                new_reward = p_term + i_term + d_term
                
                # Special handling during state transition
                if in_transition and hasattr(self, 'prev_rewards'):
                    # If this is an existing state with previous rewards
                    if state in self.prev_rewards:
                        # Blend old and new rewards based on transition progress
                        # Start with mostly old rewards, gradually shift to new calculation
                        blend_factor = self.transition_progress
                        prev_reward = self.prev_rewards[state]
                        
                        # Smooth transition from old to new reward
                        blended_reward = (1 - blend_factor) * prev_reward + blend_factor * new_reward
                        state_rewards[self.attribute_idx][state] = blended_reward
                        
                        # Debug logging for major changes
                        if abs(prev_reward - new_reward) > 100 and self.steps_done % 5 == 0:
                            print(f"State {state}: Blending {prev_reward:.1f}->{new_reward:.1f}, using {blended_reward:.1f}")
                    else:
                        # For the new state, use calculated reward
                        state_rewards[self.attribute_idx][state] = new_reward
                else:
                    # Standard update outside of transition
                    state_rewards[self.attribute_idx][state] = new_reward
                
                # Track total rewards allocated (absolute value)
                total_rewards += abs(state_rewards[self.attribute_idx][state])
                
                # Store current error as last error for next iteration
                self.last_error[state] = error
        
        # Store total rewards for efficiency calculation
        
        self.prev_total_rewards = total_rewards
        
        # Update adaptive bounds based on observed rewards
        self.update_adaptive_bounds(state_rewards[self.attribute_idx])
        
        return state_rewards, total_rewards
        
    def update_rewards(self, agents, current_states, state_rewards, current_diversity=None, 
                    ideal_diversity=None, epoch=None, *args, **kwargs):
        """Update rewards using RL-tuned PID control"""
        # Increment epoch counter
        self.epoch = epoch if epoch is not None else self.epoch + 1
        
        # Check if we need to update our state and action dimensions
        states_changed = False
        if len(current_states) != len(self.states) or set(current_states) != set(self.states):
            print(f"State set changed: {len(self.states)} -> {len(current_states)}")
            print(f"Old states: {self.states}")
            print(f"New states: {current_states}")
            
            # Store old dimensions
            old_state_dim = self.state_dim
            
            # Update state set and dimensions
            self.states = current_states.copy()
            self.state_dim = len(self.states) + 4  # +3 for PID params, +1 for reward_scale
            
            # Update state mapping
            self.state_to_idx = {state: i for i, state in enumerate(self.states)}
            self.idx_to_state = {i: state for i, state in enumerate(self.states)}
            
            # Initialize accumulated error and last error for new states
            for state in current_states:
                if state not in self.accumulated_error:
                    self.accumulated_error[state] = 0
                    self.last_error[state] = 0
            
            states_changed = True
            
            # If state dimensions changed, preserve network weights where possible
            if old_state_dim != self.state_dim:
                print(f"Adapting RL model for new dimensions. State dim: {old_state_dim} -> {self.state_dim}")
                
                # Create new networks with updated dimensions
                new_policy_net = DQNModel(self.state_dim, self.action_dim)
                new_target_net = DQNModel(self.state_dim, self.action_dim)
                
                # Transfer existing weights for the common dimensions
                with torch.no_grad():
                    # Get old weights
                    old_policy_weights = self.policy_net.fc1.weight.data
                    old_policy_bias = self.policy_net.fc1.bias.data
                    old_target_weights = self.target_net.fc1.weight.data
                    old_target_bias = self.target_net.fc1.bias.data
                    
                    # Determine the minimum dimension to copy
                    common_dim = min(old_state_dim, self.state_dim)
                    
                    # Copy weights for the first layer up to the common dimensions
                    new_policy_net.fc1.weight.data[:, :common_dim] = old_policy_weights[:, :common_dim]
                    new_policy_net.fc1.bias.data = old_policy_bias  # Bias dimension doesn't change
                    
                    new_target_net.fc1.weight.data[:, :common_dim] = old_target_weights[:, :common_dim]
                    new_target_net.fc1.bias.data = old_target_bias  # Bias dimension doesn't change
                    
                    # Copy remaining layers completely (dimensions unchanged)
                    new_policy_net.fc2.weight.data = self.policy_net.fc2.weight.data
                    new_policy_net.fc2.bias.data = self.policy_net.fc2.bias.data
                    new_policy_net.fc3.weight.data = self.policy_net.fc3.weight.data
                    new_policy_net.fc3.bias.data = self.policy_net.fc3.bias.data
                    
                    new_target_net.fc2.weight.data = self.target_net.fc2.weight.data
                    new_target_net.fc2.bias.data = self.target_net.fc2.bias.data
                    new_target_net.fc3.weight.data = self.target_net.fc3.weight.data
                    new_target_net.fc3.bias.data = self.target_net.fc3.bias.data
                
                # Update networks
                self.policy_net = new_policy_net
                self.target_net = new_target_net
                self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
                
                # Handle the replay buffer with different dimensions
                if len(self.replay_buffer.buffer) > 0:
                    print("Adapting replay buffer to new dimensions...")
                    adapted_buffer = deque(maxlen=self.replay_buffer.buffer.maxlen)
                    
                    for exp in self.replay_buffer.buffer:
                        # Handle both expansion and reduction in dimensions
                        if self.state_dim > old_state_dim:
                            # Expanding: Add zeros for new dimensions
                            extended_state = list(exp.state) + [0.0] * (self.state_dim - old_state_dim)
                            extended_next_state = list(exp.next_state) + [0.0] * (self.state_dim - old_state_dim)
                        else:
                            # Reducing: Truncate to fewer dimensions
                            extended_state = list(exp.state)[:self.state_dim]
                            extended_next_state = list(exp.next_state)[:self.state_dim]
                        
                        # Create new experience with adjusted states
                        new_exp = Experience(extended_state, exp.action, exp.reward, extended_next_state, exp.done)
                        adapted_buffer.append(new_exp)
                    
                    # Replace the buffer with our adapted version
                    self.replay_buffer.buffer = adapted_buffer
                    self.replay_buffer.state_dim = self.state_dim
                    print(f"Adapted {len(adapted_buffer)} experiences to new dimensions")
                
                # Update previous state with appropriate dimensions if it exists
                if self.prev_state is not None:
                    if self.state_dim > old_state_dim:
                        # Expanding: Add zeros for new dimensions
                        self.prev_state = list(self.prev_state) + [0.0] * (self.state_dim - old_state_dim)
                    else:
                        # Reducing: Truncate to fewer dimensions
                        self.prev_state = list(self.prev_state)[:self.state_dim]
        
        try:
            # Update agent counts for this epoch
            agent_counts = {state: 0 for state in current_states}
            for agent in agents:
                agent_state = agent.declared_state[self.attribute_idx]
                agent_counts[agent_state] = agent_counts.get(agent_state, 0) + 1
            self.agent_counts = agent_counts
            
            # Get current state representation
            current_state = self.get_state_representation(agents, current_states)
            
            # Calculate current diversity ratio for stability check
            diversity_ratio = 0
            if current_diversity is not None and ideal_diversity is not None and ideal_diversity > 0:
                diversity_ratio = current_diversity / ideal_diversity
            
            # Skip RL parameter updates during transition period, just apply current PID
            if hasattr(self, 'in_transition') and self.in_transition:
                # Log transition progress periodically
                if self.steps_done % 10 == 0:
                    print(f"In transition period: {self.transition_progress*100:.1f}% complete")
                
                # Apply PID control without changing parameters
                state_rewards, total_rewards = self.update_rewards_using_pid(agents, current_states, state_rewards)
                self.prev_total_rewards = total_rewards
                
                # Update transition progress
                self.transition_progress += 1.0 / self.transition_epochs
                if self.transition_progress >= 1.0:
                    # Transition complete
                    self.in_transition = False
                    self.transition_progress = 1.0
                    print("Transition period complete, resuming RL parameter updates")
                
                # Update previous state for next iteration (but not action since we didn't select one)
                if len(current_state) == self.state_dim:
                    self.prev_state = current_state 
                self.prev_diversity = current_diversity if current_diversity is not None else self.prev_diversity
                
                # Increment step counter
                self.steps_done += 1
                
                return state_rewards
            # CHECK FOR STABILITY CONDITION - If diversity ratio is above threshold,
            # skip RL updates and just apply the existing PID parameters
            if diversity_ratio > 0.995:
                # Only apply PID control with current parameters
                if self.steps_done % 100 == 0:
                    print(f"Diversity ratio {diversity_ratio:.3f} > 0.99 - Keeping parameters stable")
                    print(f"Current parameters: P: {self.p_param:.1f}, I: {self.i_param:.1f}, D: {self.d_param:.1f}, "
                        f"Reward Scale: {self.reward_scale:.1f}")
                
                # Skip RL learning when diversity is good, just apply PID control
                state_rewards, total_rewards = self.update_rewards_using_pid(agents, current_states, state_rewards)
                self.prev_total_rewards = total_rewards
                
                # Update previous state for next iteration (but not action since we didn't select one)
                if len(current_state) == self.state_dim:
                    self.prev_state = current_state 
                self.prev_diversity = current_diversity if current_diversity is not None else self.prev_diversity
                
                # Increment step counter
                self.steps_done += 1
                
                return state_rewards
            
            # If this is the first call or states changed, just store the initial state
            if self.prev_state is None or states_changed:
                self.prev_state = current_state
                self.prev_diversity = current_diversity if current_diversity is not None else 0
                # For the first step, use a default action (zero adjustment)
                actions = np.zeros(self.action_dim)
            else:
                # Select action based on current state
                actions = self.select_action(current_state)
                
                # Apply the selected actions to adjust parameters
                self.apply_actions_to_pid_parameters(actions)
                
                # Calculate reward if diversity metrics are provided
                if current_diversity is not None and ideal_diversity is not None:
                    # Apply PID control first to get total rewards
                    state_rewards, total_rewards = self.update_rewards_using_pid(agents, current_states, state_rewards)
                    
                    # Calculate reward based on both diversity and efficiency
                    reward = self.calculate_reward(current_diversity, ideal_diversity, total_rewards)
                    # pass ideal diversity to RLPIDController
                    self.ideal_diversity = ideal_diversity
                    # Calculate diversity ratio here
                    diversity_ratio = current_diversity / ideal_diversity if ideal_diversity > 0 else 0
                    
                    # Store experience in replay buffer - only if dimensions match to avoid warnings
                    done = False  # Not episodic in traditional sense
                    if self.prev_action is not None and len(self.prev_action) == self.action_dim:
                        if len(self.prev_state) == self.state_dim and len(current_state) == self.state_dim:
                            self.replay_buffer.add(self.prev_state, self.prev_action, reward, current_state, done)
                    
                    # Update model periodically
                    if self.steps_done % self.update_frequency == 0 and len(self.replay_buffer) >= self.batch_size:
                        self.update_model()
                    
                    # Update target network periodically
                    if self.steps_done % self.target_update_frequency == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                    
                    # Decay epsilon
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                    
                    # Update stored total rewards for next iteration
                    self.prev_total_rewards = total_rewards
                    
                    # Store best parameters if diversity is better than previous best
                    if diversity_ratio > self.best_diversity_ratio:
                        self.best_diversity_ratio = diversity_ratio
                        self.best_pid_params = (self.p_param, self.i_param, self.d_param)
                        self.best_reward_scale = self.reward_scale
                        print(f"New best parameters! P={self.p_param:.1f}, I={self.i_param:.1f}, D={self.d_param:.1f}, "
                            f"Scale={self.reward_scale:.1f} (diversity ratio: {diversity_ratio:.3f})")
                else:
                    # No diversity metrics available, just apply PID control
                    state_rewards, total_rewards = self.update_rewards_using_pid(agents, current_states, state_rewards)
                    self.prev_total_rewards = total_rewards
            
            # Print PID parameters and performance metrics periodically
            if self.steps_done % 20 == 0:
                p_total = self.p_param + self.i_param + self.d_param
                if p_total > 0:
                    p_dist = f"P: {100*self.p_param/p_total:.1f}%, I: {100*self.i_param/p_total:.1f}%, D: {100*self.d_param/p_total:.1f}%"
                else:
                    p_dist = "All parameters are zero"
                    
                print(f"PID params: P={self.p_param:.1f}, I={self.i_param:.1f}, D={self.d_param:.1f}, Scale={self.reward_scale:.1f} | {p_dist}")
            
            # Update previous state and action for next iteration
            # Only update prev_state if dimensions are compatible
            if len(current_state) == self.state_dim:
                self.prev_state = current_state 
            self.prev_action = actions
            self.prev_diversity = current_diversity if current_diversity is not None else self.prev_diversity
            
            # Increment step counter
            self.steps_done += 1
            
            return state_rewards
            
        except Exception as e:
            print(f"Error in update_rewards: {e}")
            import traceback
            traceback.print_exc()
            # If there's an error, return unchanged state rewards
            return state_rewards