import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

# Define experience tuple structure
Experience = namedtuple('Experience', ['version', 'action', 'reward', 'next_version', 'done'])

class ReplayBuffer:
    """Experience replay buffer to store and sample experiences"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.version_dim = None  # Track version dimension
        self.action_dim = None  # Track action dimension
    
    def add(self, version, action, reward, next_version, done):
        """Add experience to buffer"""
        # Check and update dimensions on first addition
        if len(self.buffer) == 0:
            self.version_dim = len(version)
            if isinstance(action, (np.ndarray, list)):
                self.action_dim = len(action)
            else:
                self.action_dim = 1  # Scalar action
            
        # Skip adding experiences if dimensions don't match
        if len(version) != self.version_dim:
            print(f"Skipping experience with mismatched version dimension: got {len(version)}, expected {self.version_dim}")
            return
            
        if isinstance(action, (np.ndarray, list)) and len(action) != self.action_dim:
            print(f"Skipping experience with mismatched action dimension: got {len(action)}, expected {self.action_dim}")
            return
            
        # Make a copy of version and next_version to ensure they don't get modified
        version_copy = version.copy() if isinstance(version, (list, np.ndarray)) else version
        next_version_copy = next_version.copy() if isinstance(next_version, (list, np.ndarray)) else next_version
        
        # Make a copy of action to ensure it doesn't get modified
        if isinstance(action, (list, np.ndarray)):
            action_copy = action.copy()
        else:
            action_copy = action
            
        experience = Experience(version_copy, action_copy, reward, next_version_copy, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Randomly sample batch_size experiences from buffer"""
        if len(self.buffer) < batch_size:
            # Return None if not enough samples
            return None
            
        try:
            experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
            
            # Convert experiences to numpy arrays first for safer handling
            versions = np.array([list(e.version) for e in experiences])  # Ensure version is listified
            
            # Handle actions - make sure they're all the same type (list or scalar)
            if isinstance(experiences[0].action, (np.ndarray, list)):
                actions = np.array([list(e.action) for e in experiences])
            else:
                # For scalar actions
                actions = np.array([[e.action] for e in experiences])
                
            rewards = np.array([[e.reward] for e in experiences])
            next_versions = np.array([list(e.next_version) for e in experiences])  # Ensure version is listified
            dones = np.array([[e.done] for e in experiences])
            
            # Convert to tensors
            versions_tensor = torch.FloatTensor(versions)
            actions_tensor = torch.FloatTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)
            next_versions_tensor = torch.FloatTensor(next_versions)
            dones_tensor = torch.FloatTensor(dones)
            
            return versions_tensor, actions_tensor, rewards_tensor, next_versions_tensor, dones_tensor
            
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
        self.version_dim = None
        self.action_dim = None


class DQNModel(nn.Module):
    """Deep Q-Network model"""
    
    def __init__(self, version_dim, action_dim, hidden_dim=64):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(version_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize network with smaller weights for more gradual changes
        self._initialize_weights()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Use tanh to constrain output between -1 and 1 for smoother actions
        return torch.tanh(self.fc3(x))
    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     return self.fc3(x)  
    
    def _initialize_weights(self):
        """Initialize weights with smaller values for more cautious initial behavior"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Lower gain for smaller values
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class RewardController:
    """Base class for reward controllers"""
    
    def __init__(self, versions, attribute_idx=0):
        self.versions = versions
        self.attribute_idx = attribute_idx
    
    def update_rewards(self, agents, current_versions, version_rewards, *args, **kwargs):
        """Update rewards based on current version distribution"""
        raise NotImplementedError


class PIDController(RewardController):
    """PID controller for reward adjustment"""
    
    def __init__(self, versions, p_param, i_param, d_param, attribute_idx=0):
        super(PIDController, self).__init__(versions, attribute_idx)
        self.p_param = p_param
        self.i_param = i_param
        self.d_param = d_param
        self.accumulated_error = {version: 0 for version in versions}
        self.last_error = {version: 0 for version in versions}
    
    def update_rewards(self, agents, current_versions, version_rewards, *args, **kwargs):
        """Update rewards using PID control"""
        version_counts = {version: 0 for version in current_versions}
        for agent in agents:
            version_counts[agent.declared_version[self.attribute_idx]] += 1
        
        for version in current_versions:
            if version == 'NO_version':
                version_rewards[self.attribute_idx][version] = 0
            else:
                version_share = version_counts[version] / len(agents)
                ideal_share = 1 / (len(current_versions) - 1)  # -1 for NO_version
                error = (ideal_share - version_share)
                
                # Update accumulated error if version exists in it, otherwise initialize it
                if version in self.accumulated_error:
                    self.accumulated_error[version] += error
                else:
                    self.accumulated_error[version] = error
                    
                # Get last error, default to 0 if not found
                last_err = self.last_error.get(version, 0)
                
                # PID terms
                p_term = self.p_param * error
                i_term = self.i_param * self.accumulated_error[version]
                d_term = self.d_param * (error - last_err)
                
                # Update rewards
                version_rewards[self.attribute_idx][version] = p_term + i_term + d_term
            
                # Store current error as last error for next iteration
                self.last_error[version] = error
        
        return version_rewards


class RLPIDController(RewardController):
    """RL-based PID parameter tuner - adaptively learns reward scale and PID parameters"""
    
    def __init__(self, versions, n_agents, initial_p=0, initial_i=0, initial_d=0,
                 epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, gamma=0.99, learning_rate=0.001, batch_size=32, 
                 update_frequency=10, target_update_frequency=100, attribute_idx=0,
                 action_scale=0.1,
                 p_scale_factor=None, i_scale_factor=None, d_scale_factor=None):
        super(RLPIDController, self).__init__(versions, attribute_idx)
        
        # RL parameters
        self.n_agents = n_agents
        self.initial_epsilon = epsilon
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma  # Discount factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        self.action_scale = action_scale  # Control the magnitude of PID parameter adjustments
        
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
        
        # PID version tracking
        self.accumulated_error = {version: 0 for version in versions}
        self.last_error = {version: 0 for version in versions}
        
        # Anti-windup for I term - prevent integral accumulation from getting too large
        self.i_term_max = self.reward_scale * 5  # Initial value, will be adjusted
        
        # Track current agent distribution for version representation
        self.agent_counts = {version: 0 for version in versions}
        
        # Define version and action dimensions
        # version includes: agent distribution + current PID parameters + current reward scale
        self.version_dim = len(versions) + 4  # +3 for current P, I, D values, +1 for reward scale
        
        # Actions: adjustments to P, I, D, and reward_scale
        self.action_dim = 4  # One action dimension for each PID parameter + reward scale
        
        # Initialize models
        self.policy_net = DQNModel(self.version_dim, self.action_dim)
        self.target_net = DQNModel(self.version_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Track episodes and steps
        self.steps_done = 0
        self.epoch = 0
        
        # Store previous version
        self.prev_version = None
        self.prev_action = None
        
        # Store version mapping (index to version name)
        self.version_to_idx = {version: i for i, version in enumerate(versions)}
        self.idx_to_version = {i: version for i, version in enumerate(self.versions)}
        
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
        # Extract the maximum absolute reward value in the current version
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
        
    def get_version_representation(self, agents, current_versions):
        """Create version representation: [agent distribution, P, I, D, reward_scale]"""
        # Get agent distribution
        version_counts = {version: 0 for version in current_versions}
        for agent in agents:
            version_counts[agent.declared_version[self.attribute_idx]] += 1
        
        # Update version mappings if they don't match current versions
        if set(current_versions) != set(self.version_to_idx.keys()):
            print(f"Updating version mappings: {len(self.version_to_idx)} -> {len(current_versions)} versions")
            self.version_to_idx = {version: i for i, version in enumerate(current_versions)}
            self.idx_to_version = {i: version for i, version in enumerate(current_versions)}
        
        # Create version vector with correct size based on current versions
        version_vector = np.zeros(len(current_versions))
        
        # Fill in the version distribution
        for version, count in version_counts.items():
            if version in self.version_to_idx:
                idx = self.version_to_idx[version]
                # Safety check to prevent index out of bounds
                if idx < len(version_vector):
                    version_vector[idx] = count / self.n_agents
                else:
                    print(f"Warning: version {version} has index {idx} but vector size is {len(version_vector)}")
        
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
        
        # Combine version vector with normalized parameters
        full_version = np.append(version_vector, [norm_p, norm_i, norm_d, norm_reward_scale])
        
        # Update expected version dimension based on current versions
        expected_version_dim = len(current_versions) + 4  # +3 for PID params, +1 for reward_scale
        
        # If version dimension changed, update it
        if self.version_dim != expected_version_dim:
            print(f"version dimension changed: {self.version_dim} -> {expected_version_dim}")
            self.version_dim = expected_version_dim
        
        # Verify that the version vector has the expected dimension
        if len(full_version) != self.version_dim:
            print(f"Warning: version dimension mismatch in get_version_representation. Got {len(full_version)}, expected {self.version_dim}")
            print(f"Current versions: {len(current_versions)}, version vector length: {len(version_vector)}")
            
            # Adjust the version vector to match expected dimension
            if len(full_version) < self.version_dim:
                # Pad with zeros if too short
                full_version = np.pad(full_version, (0, self.version_dim - len(full_version)), 'constant')
            else:
                # Truncate if too long
                full_version = full_version[:self.version_dim]
        
        return full_version.tolist()  # Convert to list for consistent serialization

    
    def select_action(self, version):
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
                version_tensor = torch.FloatTensor(version).unsqueeze(0)
                # Output already constrained by tanh in the network
                return self.policy_net(version_tensor).squeeze().numpy()
    
    def update_model(self):
        """Update the policy network using a batch of experiences"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            # If sampling failed, skip update
            return
            
        versions, actions, rewards, next_versions, dones = batch
        
        try:
            # Compute current Q values
            current_q_values = self.policy_net(versions)
            
            # Compute next Q values using target network
            with torch.no_grad():
                next_q_values = self.target_net(next_versions).max(1)[0].unsqueeze(1)
            
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
        
        if self.steps_done % 1000 == 0:
            print(f"Parameters updated: P: {old_p:.1f} -> {new_p:.1f}, "
                f"I: {old_i:.1f} -> {new_i:.1f}, D: {old_d:.1f} -> {new_d:.1f}, "
                f"Reward Scale: {old_reward_scale:.1f} -> {new_reward_scale:.1f}")

    # 3. Adaptive bounds based on observed rewards
    def update_adaptive_bounds(self, current_rewards):
        """Update adaptive bounds for PID parameters based on observed rewards"""
        # Extract the maximum absolute reward value in the current version
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
        if self.steps_done % 1000 == 0:
            print(f"Adaptive bounds: max_p={self.max_p:.1f}, max_i={self.max_i:.1f}, "
                f"max_d={self.max_d:.1f}, max_observed_reward={self.max_observed_reward:.1f}")
    
    def update_rewards_using_pid(self, agents, current_versions, version_rewards):
        """Apply PID control to update rewards based on current parameters"""
        version_counts = {version: 0 for version in current_versions}
        for agent in agents:
            version_counts[agent.declared_version[self.attribute_idx]] += 1
        
        # Reset accumulated error if it gets too large (anti-windup)
        for version, error in self.accumulated_error.items():
            if abs(error) > self.i_term_max / max(0.001, self.i_param):
                self.accumulated_error[version] = np.sign(error) * self.i_term_max / max(0.001, self.i_param)
        
        # Track total allocated rewards
        total_rewards = 0
        
        # Check if we're in a version transition period
        in_transition = hasattr(self, 'in_transition') and self.in_transition
        
        # Update transition progress if we're in transition mode
        if in_transition:
            self.transition_progress += 1.0 / self.transition_epochs
            if self.transition_progress >= 1.0:
                # Transition complete
                self.in_transition = False
                self.transition_progress = 1.0
        
        for version in current_versions:
            if version == 'NO_version':
                version_rewards[self.attribute_idx][version] = 0
            else:
                # Calculate regular PID control values
                version_share = version_counts[version] / len(agents)
                ideal_share = 1 / (len(current_versions) - 1)  # -1 for NO_version
                error = (ideal_share - version_share)
                
                # Update accumulated error if version exists in it, otherwise initialize it
                if version in self.accumulated_error:
                    self.accumulated_error[version] += error
                else:
                    self.accumulated_error[version] = error
                    
                # Get last error, default to 0 if not found
                last_err = self.last_error.get(version, 0)
                
                # Apply anti-windup for I term - still needed for numerical stability
                if abs(self.accumulated_error[version]) > self.i_term_max / max(0.001, self.i_param):
                    self.accumulated_error[version] = np.sign(self.accumulated_error[version]) * self.i_term_max / max(0.001, self.i_param)
                
                # PID terms
                p_term = self.p_param * error
                i_term = self.i_param * self.accumulated_error[version]
                d_term = self.d_param * (error - last_err)
                
                # Calculate new reward based on PID control
                new_reward = p_term + i_term + d_term
                
                # if self.steps_done % 500 == 0:  # Log every 5 steps during transition
                #     print(f"DEBUG {version}: ideal_share={ideal_share:.3f}, version_share={version_share:.3f}, error={error:.3f}")
                #     print(f"  PID params: P={self.p_param:.1f}, I={self.i_param:.1f}, D={self.d_param:.1f}")
                #     print(f"  Accumulated_error={self.accumulated_error[version]:.3f}, last_err={last_err:.3f}")
                #     print(f"  Terms: P={p_term:.1f}, I={i_term:.1f}, D={d_term:.1f}, Total={new_reward:.1f}")
                #     print(f"  Reward scale: {self.reward_scale:.1f}")
                
                version_rewards[self.attribute_idx][version] = new_reward
                
                # Track total rewards allocated (absolute value)
                total_rewards += abs(version_rewards[self.attribute_idx][version])
                
                # Store current error as last error for next iteration
                self.last_error[version] = error
        
        # Store total rewards for efficiency calculation
        
        self.prev_total_rewards = total_rewards
        
        # Update adaptive bounds based on observed rewards
        self.update_adaptive_bounds(version_rewards[self.attribute_idx])
        
        return version_rewards, total_rewards
        
    def update_rewards(self, agents, current_versions, version_rewards, current_diversity=None, 
                    ideal_diversity=None, epoch=None, *args, **kwargs):
        """Update rewards using RL-tuned PID control"""
        # Increment epoch counter
        self.epoch = epoch if epoch is not None else self.epoch + 1
        
        # Check if we need to update our version and action dimensions
        versions_changed = False
        if len(current_versions) != len(self.versions) or set(current_versions) != set(self.versions):
            print(f"version set changed: {len(self.versions)} -> {len(current_versions)}")
            print(f"Old versions: {self.versions}")
            print(f"New versions: {current_versions}")
            
            # Store old dimensions
            old_version_dim = self.version_dim
            old_versions = self.versions.copy()
            
            if hasattr(self, 'good_reward_scale'):
                preserved_reward_scale = self.good_reward_scale
                print(f"Using good reward scale from stable period: {preserved_reward_scale:.1f}")
            else:
                preserved_reward_scale = self.reward_scale
                print(f"No good scale stored, using current: {preserved_reward_scale:.1f}")

            old_epsilon = self.epsilon
            self.epsilon = self.initial_epsilon
            print(f"New version detected - resetting exploration rate: {old_epsilon:.3f} -> {self.epsilon:.3f}")


            # Update version set and dimensions
            self.versions = list(current_versions)  # Convert to list to ensure consistent ordering
            self.version_dim = len(self.versions) + 4  # +3 for PID params, +1 for reward_scale
            
            # Update version mapping - IMPORTANT: Use consistent ordering
            self.version_to_idx = {version: i for i, version in enumerate(self.versions)}
            self.idx_to_version = {i: version for i, version in enumerate(self.versions)}
            
            # Initialize accumulated error and last error for new versions
            for version in current_versions:
                if version not in self.accumulated_error:
                    self.accumulated_error[version] = 0
                if version not in self.last_error:
                    self.last_error[version] = 0
            
            # Scale accumulated errors for the new ideal distribution
            old_ideal_share = 1 / (len(old_versions) - 1) if len(old_versions) > 1 else 0  # -1 for NO_version
            new_ideal_share = 1 / (len(current_versions) - 1) if len(current_versions) > 1 else 0  # -1 for NO_version

            if old_ideal_share > 0 and new_ideal_share > 0:
                error_scale_factor = old_ideal_share / new_ideal_share
                print(f"Scaling accumulated errors by factor: {error_scale_factor:.3f} (old ideal: {old_ideal_share:.3f} -> new ideal: {new_ideal_share:.3f})")
                
                # Scale existing accumulated errors for existing versions
                for version in current_versions:
                    if version in self.accumulated_error and version != 'NO_version':
                        old_error = self.accumulated_error[version]
                        self.accumulated_error[version] *= error_scale_factor
                        if abs(old_error) > 0.001:  # Only log significant errors
                            print(f"Scaled accumulated error for {version}: {old_error:.3f} -> {self.accumulated_error[version]:.3f}")

            versions_changed = True
            
            # If version dimensions changed, preserve network weights where possible
            if old_version_dim != self.version_dim:
                print(f"Adapting RL model for new dimensions. version dim: {old_version_dim} -> {self.version_dim}")
                
                # Create new networks with updated dimensions
                new_policy_net = DQNModel(self.version_dim, self.action_dim)
                new_target_net = DQNModel(self.version_dim, self.action_dim)
                
                # Transfer existing weights for the common dimensions
                with torch.no_grad():
                    # Get old weights
                    old_policy_weights = self.policy_net.fc1.weight.data
                    old_policy_bias = self.policy_net.fc1.bias.data
                    old_target_weights = self.target_net.fc1.weight.data
                    old_target_bias = self.target_net.fc1.bias.data
                    
                    # Determine the minimum dimension to copy
                    # For versions, we need to be more careful about which versions to preserve
                    old_version_count = len(old_versions)
                    new_version_count = len(self.versions)
                    common_version_count = min(old_version_count, new_version_count)
                    
                    # Copy weights for the common version dimensions (first part of input)
                    new_policy_net.fc1.weight.data[:, :common_version_count] = old_policy_weights[:, :common_version_count]
                    new_target_net.fc1.weight.data[:, :common_version_count] = old_target_weights[:, :common_version_count]
                    
                    # Copy the PID parameter weights (last 4 dimensions)
                    # These should always be in the same position relative to the end
                    pid_param_start_old = old_version_count
                    pid_param_start_new = new_version_count
                    
                    if old_version_dim >= 4 and self.version_dim >= 4:
                        # Copy PID parameter weights
                        new_policy_net.fc1.weight.data[:, pid_param_start_new:pid_param_start_new+4] = \
                            old_policy_weights[:, pid_param_start_old:pid_param_start_old+4]
                        new_target_net.fc1.weight.data[:, pid_param_start_new:pid_param_start_new+4] = \
                            old_target_weights[:, pid_param_start_old:pid_param_start_old+4]
                    
                    # Copy bias (doesn't depend on input dimension)
                    new_policy_net.fc1.bias.data = old_policy_bias
                    new_target_net.fc1.bias.data = old_target_bias
                    
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
                        try:
                            # Create new version representations based on the dimension change
                            old_version_part = list(exp.version)[:old_version_count]  # Old version distribution
                            old_pid_part = list(exp.version)[old_version_count:old_version_count+4]  # PID params
                            
                            old_next_version_part = list(exp.next_version)[:old_version_count]
                            old_next_pid_part = list(exp.next_version)[old_version_count:old_version_count+4]
                            
                            # Create new version representation
                            if new_version_count > old_version_count:
                                # Adding versions - pad version distribution with zeros
                                new_version_part = old_version_part + [0.0] * (new_version_count - old_version_count)
                                new_next_version_part = old_next_version_part + [0.0] * (new_version_count - old_version_count)
                            else:
                                # Removing versions - truncate version distribution
                                new_version_part = old_version_part[:new_version_count]
                                new_next_version_part = old_next_version_part[:new_version_count]
                            
                            # Combine with PID parameters
                            extended_version = new_version_part + old_pid_part
                            extended_next_version = new_next_version_part + old_next_pid_part
                            
                            # Create new experience with adjusted versions
                            new_exp = Experience(extended_version, exp.action, exp.reward, extended_next_version, exp.done)
                            adapted_buffer.append(new_exp)
                            
                        except (IndexError, ValueError) as e:
                            print(f"Skipping experience due to adaptation error: {e}")
                            continue
                    
                    # Replace the buffer with our adapted version
                    self.replay_buffer.buffer = adapted_buffer
                    self.replay_buffer.version_dim = self.version_dim
                    print(f"Adapted {len(adapted_buffer)} experiences to new dimensions")
                
                # Update previous version with appropriate dimensions if it exists
                if self.prev_version is not None and len(self.prev_version) > 0:
                    try:
                        # Simple and robust adaptation approach
                        if len(self.prev_version) < self.version_dim:
                            # Pad with zeros if version dimension increased
                            self.prev_version = list(self.prev_version) + [0.0] * (self.version_dim - len(self.prev_version))
                        elif len(self.prev_version) > self.version_dim:
                            # Truncate if version dimension decreased  
                            self.prev_version = self.prev_version[:self.version_dim]
                        
                        # Ensure it's a list for consistency
                        self.prev_version = list(self.prev_version)
                        print(f"Adapted prev_version to new dimensions: {len(self.prev_version)}")
                        
                    except Exception as e:
                        print(f"Error adapting prev_version: {e}, but continuing with adapted version")
                        # Don't reset to None - try to create a reasonable default version
                        if self.version_dim > 0:
                            # Create default version with current agent distribution + current PID params
                            try:
                                current_version = self.get_version_representation(agents, current_versions)
                                self.prev_version = current_version
                                print(f"Created new prev_version from current version: {len(self.prev_version)}")
                            except:
                                # Last resort - create zero version
                                self.prev_version = [0.0] * self.version_dim
                                print(f"Created zero prev_version: {len(self.prev_version)}")
                else:
                    print("prev_version was None, will be set from current version in next iteration")
            self.reward_scale = preserved_reward_scale
            print(f"Restored reward scale after network adaptation: {self.reward_scale:.1f}")
            
            self.in_transition = True
            self.transition_epochs = 500  # Number of epochs to protect reward scale
            self.transition_progress = 0
            self.stable_reward_scale = preserved_reward_scale  # Store for protection
            print(f"Entering transition protection period: {self.transition_epochs} epochs")

        try:
            # Update agent counts for this epoch
            agent_counts = {version: 0 for version in current_versions}
            for agent in agents:
                agent_version = agent.declared_version[self.attribute_idx]
                agent_counts[agent_version] = agent_counts.get(agent_version, 0) + 1
            self.agent_counts = agent_counts
            
            # Get current version representation
            current_version = self.get_version_representation(agents, current_versions)
            
            # Calculate current diversity ratio for stability check
            diversity_ratio = 0
            if current_diversity is not None and ideal_diversity is not None and ideal_diversity > 0:
                diversity_ratio = current_diversity / ideal_diversity
            
            # Skip RL parameter updates during transition period, just apply current PID
            if hasattr(self, 'in_transition') and self.in_transition:
                # Log transition progress periodically
                if self.steps_done % 10 == 0:
                    print(f"In transition period: {self.transition_progress*100:.1f}% complete")
                if hasattr(self, 'stable_reward_scale'):
                    self.reward_scale = self.stable_reward_scale

                # Apply PID control without changing parameters
                version_rewards, total_rewards = self.update_rewards_using_pid(agents, current_versions, version_rewards)
                self.prev_total_rewards = total_rewards
                
                # Update transition progress
                self.transition_progress += 1.0 / self.transition_epochs
                if self.transition_progress >= 1.0:
                    # Transition complete
                    self.in_transition = False
                    self.transition_progress = 1.0
                    print("Transition period complete, resuming RL parameter updates")
                    if hasattr(self, 'stable_reward_scale'):
                        delattr(self, 'stable_reward_scale')
                
                # Update previous version for next iteration (but not action since we didn't select one)
                if len(current_version) == self.version_dim:
                    self.prev_version = current_version 
                self.prev_diversity = current_diversity if current_diversity is not None else self.prev_diversity
                
                # Increment step counter
                self.steps_done += 1
                
                return version_rewards
            
            # CHECK FOR STABILITY CONDITION - If diversity ratio is above threshold,
            # skip RL updates and just apply the existing PID parameters
            if diversity_ratio > 0.995:

                # Store good reward scale when system is stable with good diversity
                if not hasattr(self, 'good_reward_scale') or self.reward_scale > getattr(self, 'good_reward_scale', 0):
                    self.good_reward_scale = self.reward_scale
                    if self.steps_done % 100 == 0:
                        print(f"Storing good reward scale: {self.reward_scale:.1f} (diversity: {diversity_ratio:.3f})")
                
                # Only apply PID control with current parameters
                if self.steps_done % 100 == 0:
                    print(f"Diversity ratio {diversity_ratio:.3f} > 0.99 - Keeping parameters stable")
                    print(f"Current parameters: P: {self.p_param:.1f}, I: {self.i_param:.1f}, D: {self.d_param:.1f}, "
                        f"Reward Scale: {self.reward_scale:.1f}")
                
                # Skip RL learning when diversity is good, just apply PID control
                version_rewards, total_rewards = self.update_rewards_using_pid(agents, current_versions, version_rewards)
                self.prev_total_rewards = total_rewards
                
                # Update previous version for next iteration (but not action since we didn't select one)
                if len(current_version) == self.version_dim:
                    self.prev_version = current_version 
                self.prev_diversity = current_diversity if current_diversity is not None else self.prev_diversity
                
                # Increment step counter
                self.steps_done += 1
                
                return version_rewards
            
            # If this is the first call or versions changed, just store the initial version
            if self.prev_version is None or versions_changed:
                self.prev_version = current_version
                self.prev_diversity = current_diversity if current_diversity is not None else 0
                # For the first step, use a default action (zero adjustment)
                actions = np.zeros(self.action_dim)
            else:
                # Select action based on current version
                actions = self.select_action(current_version)

                if hasattr(self, 'in_transition') and self.in_transition:
                # Zero out the reward scale adjustment (last action)
                    if len(actions) >= 4:
                        actions[3] = 0  # Don't change reward scale during transition
                    
                # Apply the selected actions to adjust parameters
                self.apply_actions_to_pid_parameters(actions)
                
                # Calculate reward if diversity metrics are provided
                if current_diversity is not None and ideal_diversity is not None:
                    # Apply PID control first to get total rewards
                    version_rewards, total_rewards = self.update_rewards_using_pid(agents, current_versions, version_rewards)
                    
                    # Calculate reward based on diversity
                    reward = self.calculate_reward(current_diversity, ideal_diversity, total_rewards)
                    # pass ideal diversity to RLPIDController
                    self.ideal_diversity = ideal_diversity
                    # Calculate diversity ratio here
                    diversity_ratio = current_diversity / ideal_diversity if ideal_diversity > 0 else 0
                    
                    # Store experience in replay buffer - only if dimensions match to avoid warnings
                    done = False  # Not episodic in traditional sense
                    if self.prev_action is not None and len(self.prev_action) == self.action_dim:
                        if len(self.prev_version) == self.version_dim and len(current_version) == self.version_dim:
                            self.replay_buffer.add(self.prev_version, self.prev_action, reward, current_version, done)
                    
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
                else:
                    # No diversity metrics available, just apply PID control
                    version_rewards, total_rewards = self.update_rewards_using_pid(agents, current_versions, version_rewards)
                    self.prev_total_rewards = total_rewards
            
            # Print PID parameters and performance metrics periodically
            if self.steps_done % 1000 == 0:
                p_total = self.p_param + self.i_param + self.d_param
                if p_total > 0:
                    p_dist = f"P: {100*self.p_param/p_total:.1f}%, I: {100*self.i_param/p_total:.1f}%, D: {100*self.d_param/p_total:.1f}%"
                else:
                    p_dist = "All parameters are zero"
                    
                print(f"PID params: P={self.p_param:.1f}, I={self.i_param:.1f}, D={self.d_param:.1f}, Scale={self.reward_scale:.1f} | {p_dist}")
            
            # Update previous version and action for next iteration
            # Only update prev_version if dimensions are compatible
            if len(current_version) == self.version_dim:
                self.prev_version = current_version 
            self.prev_action = actions
            self.prev_diversity = current_diversity if current_diversity is not None else self.prev_diversity
            
            # Increment step counter
            self.steps_done += 1
            
            return version_rewards
            
        except Exception as e:
            print(f"Error in update_rewards: {e}")
            import traceback
            traceback.print_exc()
            # If there's an error, return unchanged version rewards
            return version_rewards
            
            # If this is the first call or versions changed, just store the initial version
            if self.prev_version is None or versions_changed:
                self.prev_version = current_version
                self.prev_diversity = current_diversity if current_diversity is not None else 0
                # For the first step, use a default action (zero adjustment)
                actions = np.zeros(self.action_dim)
            else:
                # Select action based on current version
                actions = self.select_action(current_version)
                
                # Apply the selected actions to adjust parameters
                self.apply_actions_to_pid_parameters(actions)
                
                # Calculate reward if diversity metrics are provided
                if current_diversity is not None and ideal_diversity is not None:
                    # Apply PID control first to get total rewards
                    version_rewards, total_rewards = self.update_rewards_using_pid(agents, current_versions, version_rewards)
                    
                    # Calculate reward based on diversity
                    reward = self.calculate_reward(current_diversity, ideal_diversity, total_rewards)
                    # pass ideal diversity to RLPIDController
                    self.ideal_diversity = ideal_diversity
                    # Calculate diversity ratio here
                    diversity_ratio = current_diversity / ideal_diversity if ideal_diversity > 0 else 0
                    
                    # Store experience in replay buffer - only if dimensions match to avoid warnings
                    done = False  # Not episodic in traditional sense
                    if self.prev_action is not None and len(self.prev_action) == self.action_dim:
                        if len(self.prev_version) == self.version_dim and len(current_version) == self.version_dim:
                            self.replay_buffer.add(self.prev_version, self.prev_action, reward, current_version, done)
                    
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
                    version_rewards, total_rewards = self.update_rewards_using_pid(agents, current_versions, version_rewards)
                    self.prev_total_rewards = total_rewards
            
            # Print PID parameters and performance metrics periodically
            if self.steps_done % 20 == 0:
                p_total = self.p_param + self.i_param + self.d_param
                if p_total > 0:
                    p_dist = f"P: {100*self.p_param/p_total:.1f}%, I: {100*self.i_param/p_total:.1f}%, D: {100*self.d_param/p_total:.1f}%"
                else:
                    p_dist = "All parameters are zero"
                    
                print(f"PID params: P={self.p_param:.1f}, I={self.i_param:.1f}, D={self.d_param:.1f}, Scale={self.reward_scale:.1f} | {p_dist}")
            
            # Update previous version and action for next iteration
            # Only update prev_version if dimensions are compatible
            if len(current_version) == self.version_dim:
                self.prev_version = current_version 
            self.prev_action = actions
            self.prev_diversity = current_diversity if current_diversity is not None else self.prev_diversity
            
            # Increment step counter
            self.steps_done += 1
            
            return version_rewards
            
        except Exception as e:
            print(f"Error in update_rewards: {e}")
            import traceback
            traceback.print_exc()
            # If there's an error, return unchanged version rewards
            return version_rewards