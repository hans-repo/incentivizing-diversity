import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from reinforcement_learning import ReplayBuffer, Experience, RewardController

class DirectRewardDQNModel(nn.Module):
    """DQN model for direct reward control without PID"""
    
    def __init__(self, version_dim, num_versions, hidden_dim=128):
        super(DirectRewardDQNModel, self).__init__()
        # version dim includes agent distribution and current rewards
        self.fc1 = nn.Linear(version_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer outputs reward adjustments for each version
        self.fc4 = nn.Linear(hidden_dim, num_versions)
        
        self._initialize_weights()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Use tanh to constrain adjustments to [-1, 1]
        return torch.tanh(self.fc4(x))

    def _initialize_weights(self):
        """Initialize weights with smaller values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class RLDirectRewardController(RewardController):
    """RL controller that directly sets rewards for each version without PID"""
    
    def __init__(self, versions, n_agents, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, gamma=0.99, learning_rate=0.001, batch_size=32, 
                 update_frequency=10, target_update_frequency=100, attribute_idx=0,
                 action_scale=0.1, initial_max_reward=100.0):
        super(RLDirectRewardController, self).__init__(versions, attribute_idx)
        
        # RL parameters
        self.n_agents = n_agents
        self.initial_epsilon = epsilon  # Store initial epsilon for reset
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        self.action_scale = action_scale
        

        # Direct reward control parameters
        # version includes: agent distribution + current rewards per version + max_reward
        self.version_dim = len(versions) * 2 + 1  # Agent distribution + current rewards + max_reward
        
        # Actions: direct reward adjustments for each version (excluding NO_version) + max_reward adjustment
        self.num_controllable_versions = len([s for s in versions if s != 'NO_version'])
        self.action_dim = self.num_controllable_versions + 1  # +1 for max_reward adjustment
        
        # Initialize models for direct reward control
        self.policy_net = DirectRewardDQNModel(self.version_dim, self.action_dim)
        self.target_net = DirectRewardDQNModel(self.version_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Track current rewards for each version - initialize with small values
        self.current_rewards = {version: initial_max_reward/10 if version != 'NO_version' else 0.0 for version in versions}
        
        # Max reward bound (now learnable!)
        self.max_reward = initial_max_reward
        self.initial_max_reward = initial_max_reward  # Store for reset scenarios
        self.min_max_reward = initial_max_reward * 0.1  # Minimum bound to prevent collapse
        
        # Remove hardcoded reward_history since max_reward is now learnable
        # self.reward_history = deque(maxlen=50)  # No longer needed
        
        # Track current agent distribution
        self.agent_counts = {version: 0 for version in versions}
        
        # Track episodes and steps
        self.steps_done = 0
        self.epoch = 0
        
        # Store previous version
        self.prev_version = None
        self.prev_action = None
        
        # Store version mapping
        self.version_to_idx = {version: i for i, version in enumerate(versions)}
        self.idx_to_version = {i: version for i, version in enumerate(self.versions)}
        
        # Previous diversity for reward calculation
        self.prev_diversity = 0
        self.ideal_diversity = 0
        
        # Track total rewards for efficiency calculation
        self.prev_total_rewards = 0
        self.total_rewards_history = deque(maxlen=10)
        
        # Performance tracking
        self.last_diversity_ratios = deque(maxlen=10)
        self.best_diversity_ratio = 0
        self.best_rewards = {}
        self.best_max_reward = initial_max_reward  # Track best max_reward too
    
    def get_version_representation(self, agents, current_versions):
        """Create version representation: [agent distribution, current rewards, max_reward]"""
        # Get agent distribution
        version_counts = {version: 0 for version in current_versions}
        for agent in agents:
            version_counts[agent.declared_version[self.attribute_idx]] += 1
        
        # Update version mappings if they don't match current versions
        if set(current_versions) != set(self.version_to_idx.keys()):
            print(f"Updating version mappings: {len(self.version_to_idx)} -> {len(current_versions)} versions")
            self.version_to_idx = {version: i for i, version in enumerate(current_versions)}
            self.idx_to_version = {i: version for i, version in enumerate(current_versions)}
        
        # Create version vector with agent distribution
        version_vector = np.zeros(len(current_versions))
        
        for version, count in version_counts.items():
            if version in self.version_to_idx:
                idx = self.version_to_idx[version]
                if idx < len(version_vector):
                    version_vector[idx] = count / self.n_agents
        
        # Append normalized current rewards
        reward_vector = np.zeros(len(current_versions))
        for version in current_versions:
            if version in self.version_to_idx:
                idx = self.version_to_idx[version]
                if idx < len(reward_vector):
                    # Normalize rewards to [-1, 1] range using learnable max_reward
                    if self.max_reward > 0:
                        reward_vector[idx] = np.clip(self.current_rewards.get(version, 0) / self.max_reward, -1, 1)
        
        # Append normalized max_reward (normalize using initial_max_reward as reference)
        normalized_max_reward = self.max_reward / (self.initial_max_reward * 10)  # Allow 10x growth from initial
        normalized_max_reward = np.clip(normalized_max_reward, 0, 1)
        
        # Combine all parts
        full_version = np.concatenate([version_vector, reward_vector, [normalized_max_reward]])
        
        # Update expected version dimension
        expected_version_dim = len(current_versions) * 2 + 1  # agent dist + rewards + max_reward
        
        if self.version_dim != expected_version_dim:
            print(f"version dimension changed: {self.version_dim} -> {expected_version_dim}")
            self.version_dim = expected_version_dim
        
        # Ensure correct dimension
        if len(full_version) != self.version_dim:
            if len(full_version) < self.version_dim:
                full_version = np.pad(full_version, (0, self.version_dim - len(full_version)), 'constant')
            else:
                full_version = full_version[:self.version_dim]
        
        return full_version.tolist()
    
    def select_action(self, version):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Exploration: random reward adjustments + max_reward adjustment
            actions = []
            # Random adjustments for each controllable version
            for _ in range(self.num_controllable_versions):
                actions.append(random.uniform(-1, 1))
            # Random adjustment for max_reward
            actions.append(random.uniform(-1, 1))
            return np.array(actions)
        else:
            # Exploitation: choose best action according to policy network
            with torch.no_grad():
                version_tensor = torch.FloatTensor(version).unsqueeze(0)
                return self.policy_net(version_tensor).squeeze().numpy()
    
    def apply_actions_to_rewards(self, actions, current_versions):
        """Apply actions to directly control rewards and max_reward"""
        controllable_versions = [s for s in current_versions if s != 'NO_version']
        
        # Split actions into reward adjustments and max_reward adjustment
        reward_adjustments = actions[:-1]  # All but last action
        max_reward_adjustment = actions[-1]  # Last action
        
        # FIX: Ensure reward adjustments match the number of controllable versions
        if len(reward_adjustments) != len(controllable_versions):
            print(f"WARNING: Action dimension mismatch. Expected {len(controllable_versions)}, got {len(reward_adjustments)}")
            # Pad with zeros or truncate as needed
            if len(reward_adjustments) < len(controllable_versions):
                reward_adjustments = np.pad(reward_adjustments, (0, len(controllable_versions) - len(reward_adjustments)), 'constant', constant_values=0)
            else:
                reward_adjustments = reward_adjustments[:len(controllable_versions)]
        
        # Apply max_reward adjustment first (using action_scale for magnitude control)
        old_max_reward = self.max_reward
        # Increase the magnitude of max_reward changes to prevent getting stuck at low values
        max_reward_change = max_reward_adjustment * self.action_scale * self.initial_max_reward
        self.max_reward = max(self.min_max_reward, old_max_reward + max_reward_change)  # Use min bound
        
        # Apply reward adjustments - actions are in range [-1, 1] from tanh activation
        for i, version in enumerate(controllable_versions):
            if i < len(reward_adjustments):
                # IMPROVED: More generous reward mapping
                # Map action from [-1, 1] to [0.1 * max_reward, max_reward] instead of [0, max_reward]
                # This ensures some minimum reward is always given
                action_normalized = (reward_adjustments[i] + 1) / 2  # Maps to [0, 1]
                min_reward = 0.1 * self.max_reward  # Minimum 10% of max_reward
                new_reward = min_reward + action_normalized * (self.max_reward - min_reward)
                self.current_rewards[version] = new_reward
        
        # Always keep NO_version reward at 0
        self.current_rewards['NO_version'] = 0
        
        # Log max_reward changes periodically
        if self.steps_done % 1000 == 0:
            print(f"Max reward: {old_max_reward:.1f} -> {self.max_reward:.1f}, Avg reward: {np.mean([r for s, r in self.current_rewards.items() if s != 'NO_version']):.1f}")
            print(f"Current rewards: {', '.join([f'{s}: {r:.1f}' for s, r in self.current_rewards.items() if s != 'NO_version'])}")
    
    def calculate_reward(self, current_diversity, ideal_diversity, current_total_rewards):
        """Calculate reward for RL training"""
        # Normalize diversity to [0, 1] range
        diversity_ratio = current_diversity / ideal_diversity if ideal_diversity > 0 else 0
        
        # Base reward for diversity
        base_reward = min(diversity_ratio, 1.0)
        
        max_reward_penalty = 0
        if diversity_ratio < 0.1:
            # Strong negative reward when both max_reward is low AND diversity is poor
            max_reward_penalty = base_reward
    
        # Total reward
        total_reward = base_reward + max_reward_penalty
        
        return total_reward
    
    def update_rewards_direct(self, agents, current_versions, version_rewards):
        """Apply current rewards from RL"""
        # Count agents in each version
        version_counts = {version: 0 for version in current_versions}
        for agent in agents:
            version_counts[agent.declared_version[self.attribute_idx]] += 1
        
        # Apply current rewards from RL
        total_rewards = 0
        for version in current_versions:
            if version == 'NO_version':
                version_rewards[self.attribute_idx][version] = 0
                self.current_rewards[version] = 0
            else:
                # Use the reward value determined by RL
                reward_val = self.current_rewards.get(version, 0)
                version_rewards[self.attribute_idx][version] = reward_val
                total_rewards += abs(reward_val)
        
        # Store total rewards for efficiency tracking
        self.total_rewards_history.append(total_rewards)
        
        return version_rewards, total_rewards
    
    def update_model(self):
        """Update the policy network using a batch of experiences"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
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
            
            # Compute loss
            loss = F.mse_loss(current_q_values, target_q_values.repeat(1, current_q_values.shape[1]))
            
            # Update policy network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        except Exception as e:
            print(f"Error in update_model: {e}")
            import traceback
            traceback.print_exc()
            self.replay_buffer.clear()
    
    def update_rewards(self, agents, current_versions, version_rewards, current_diversity=None, 
                      ideal_diversity=None, epoch=None, *args, **kwargs):
        """Update rewards using direct RL control with learnable max_reward"""
        # Increment epoch counter
        self.epoch = epoch if epoch is not None else self.epoch + 1
        
        # Check if versions changed
        versions_changed = False
        if len(current_versions) != len(self.versions) or set(current_versions) != set(self.versions):
            print(f"version set changed: {len(self.versions)} -> {len(current_versions)}")
            
            # Store old dimensions
            old_version_dim = self.version_dim
            old_action_dim = self.action_dim
            old_versions = self.versions.copy()
            old_num_controllable = self.num_controllable_versions
            
            # RESET EXPLORATION RATE TO INITIAL VALUE
            old_epsilon = self.epsilon
            self.epsilon = self.initial_epsilon
            print(f"New version detected - resetting exploration rate: {old_epsilon:.3f} -> {self.epsilon:.3f}")
            
            # Update version set and dimensions
            self.versions = list(current_versions)
            self.version_dim = len(self.versions) * 2 + 1  # agent dist + rewards + max_reward
            self.num_controllable_versions = len([s for s in self.versions if s != 'NO_version'])
            self.action_dim = self.num_controllable_versions + 1  # +1 for max_reward
            
            # Initialize rewards for new versions BEFORE any operations
            for version in current_versions:
                if version not in self.current_rewards:
                    if version == 'NO_version':
                        self.current_rewards[version] = 0
                    else:
                        # Initialize with same reward as existing versions
                        existing_rewards = [r for s, r in self.current_rewards.items() if s != 'NO_version' and r > 0]
                        initial_reward = np.mean(existing_rewards)
                        self.current_rewards[version] = initial_reward
            
            # Update version mapping
            self.version_to_idx = {version: i for i, version in enumerate(self.versions)}
            self.idx_to_version = {i: version for i, version in enumerate(self.versions)}
            
            versions_changed = True

            # Adapt networks if dimensions changed
            if old_version_dim != self.version_dim or old_action_dim != self.action_dim:
                print(f"Adapting RL model for new dimensions")
                print(f"version dim: {old_version_dim} -> {self.version_dim}")
                print(f"Action dim: {old_action_dim} -> {self.action_dim}")
                
                # Create new networks
                new_policy_net = DirectRewardDQNModel(self.version_dim, self.action_dim)
                new_target_net = DirectRewardDQNModel(self.version_dim, self.action_dim)
                
                # Manual weight transfer for compatible layers
                with torch.no_grad():
                    # Get old weights
                    old_policy_version = self.policy_net.state_dict()
                    old_target_version = self.target_net.state_dict()
                    
                    # Transfer weights for hidden layers (fc2 and fc3 have compatible dimensions)
                    new_policy_net.fc2.weight.data = old_policy_version['fc2.weight']
                    new_policy_net.fc2.bias.data = old_policy_version['fc2.bias']
                    new_policy_net.fc3.weight.data = old_policy_version['fc3.weight']
                    new_policy_net.fc3.bias.data = old_policy_version['fc3.bias']
                    
                    new_target_net.fc2.weight.data = old_target_version['fc2.weight']
                    new_target_net.fc2.bias.data = old_target_version['fc2.bias']
                    new_target_net.fc3.weight.data = old_target_version['fc3.weight']
                    new_target_net.fc3.bias.data = old_target_version['fc3.bias']
                    
                    # For fc1, handle the input dimension change
                    old_num_versions = len(old_versions)
                    new_num_versions = len(self.versions)
                    
                    # Copy agent distribution weights
                    min_versions = min(old_num_versions, new_num_versions)
                    new_policy_net.fc1.weight.data[:, :min_versions] = old_policy_version['fc1.weight'][:, :min_versions]
                    new_target_net.fc1.weight.data[:, :min_versions] = old_target_version['fc1.weight'][:, :min_versions]
                    
                    # Copy reward weights if dimensions allow
                    old_reward_start = old_num_versions
                    new_reward_start = new_num_versions
                    min_reward_dims = min(old_num_versions, new_num_versions)
                    
                    if old_version_dim > old_reward_start and self.version_dim > new_reward_start:
                        new_policy_net.fc1.weight.data[:, new_reward_start:new_reward_start+min_reward_dims] = \
                            old_policy_version['fc1.weight'][:, old_reward_start:old_reward_start+min_reward_dims]
                        new_target_net.fc1.weight.data[:, new_reward_start:new_reward_start+min_reward_dims] = \
                            old_target_version['fc1.weight'][:, old_reward_start:old_reward_start+min_reward_dims]
                    
                    # Copy max_reward weight (last dimension)
                    if old_version_dim >= 1 and self.version_dim >= 1:
                        new_policy_net.fc1.weight.data[:, -1] = old_policy_version['fc1.weight'][:, -1]
                        new_target_net.fc1.weight.data[:, -1] = old_target_version['fc1.weight'][:, -1]
                    
                    # Copy bias
                    new_policy_net.fc1.bias.data = old_policy_version['fc1.bias']
                    new_target_net.fc1.bias.data = old_target_version['fc1.bias']
                    
                    # For fc4 (output layer), handle action dimension change
                    min_action_dim = min(old_action_dim, self.action_dim)
                    new_policy_net.fc4.weight.data[:min_action_dim, :] = old_policy_version['fc4.weight'][:min_action_dim, :]
                    new_policy_net.fc4.bias.data[:min_action_dim] = old_policy_version['fc4.bias'][:min_action_dim]
                    
                    new_target_net.fc4.weight.data[:min_action_dim, :] = old_target_version['fc4.weight'][:min_action_dim, :]
                    new_target_net.fc4.bias.data[:min_action_dim] = old_target_version['fc4.bias'][:min_action_dim]
                    
                    # Initialize new action dimensions (for new versions)
                    if self.action_dim > old_action_dim:
                        # Initialize new version action weights with small random values
                        nn.init.xavier_uniform_(new_policy_net.fc4.weight.data[old_action_dim-1:-1, :], gain=0.1)  # -1 because max_reward is last
                        nn.init.constant_(new_policy_net.fc4.bias.data[old_action_dim-1:-1], 0.0)
                        
                        nn.init.xavier_uniform_(new_target_net.fc4.weight.data[old_action_dim-1:-1, :], gain=0.1)
                        nn.init.constant_(new_target_net.fc4.bias.data[old_action_dim-1:-1], 0.0)
                        
                        # The max_reward action weight should already be copied above
                
                # Update networks
                self.policy_net = new_policy_net
                self.target_net = new_target_net
                self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
                
                # Filter replay buffer for compatibility
                print(f"Filtering replay buffer for dimension compatibility...")
                compatible_experiences = []
                
                if len(self.replay_buffer) > 0:
                    # Get all experiences from replay buffer
                    all_experiences = []
                    for _ in range(len(self.replay_buffer)):
                        exp = self.replay_buffer.buffer.popleft()
                        all_experiences.append(exp)
                    
                    # Check each experience for compatibility
                    for exp in all_experiences:
                        version, action, reward, next_version, done = exp
                        
                        # Check if dimensions can be adapted
                        if (len(version) <= self.version_dim and 
                            len(next_version) <= self.version_dim and 
                            len(action) <= self.action_dim):
                            
                            # Adapt version to new dimensions
                            adapted_version = list(version)
                            if len(adapted_version) < self.version_dim:
                                adapted_version = adapted_version + [0.0] * (self.version_dim - len(adapted_version))
                            elif len(adapted_version) > self.version_dim:
                                adapted_version = adapted_version[:self.version_dim]
                            
                            adapted_next_version = list(next_version)
                            if len(adapted_next_version) < self.version_dim:
                                adapted_next_version = adapted_next_version + [0.0] * (self.version_dim - len(adapted_next_version))
                            elif len(adapted_next_version) > self.version_dim:
                                adapted_next_version = adapted_next_version[:self.version_dim]
                            
                            # Adapt action to new dimensions
                            adapted_action = list(action)
                            if len(adapted_action) < self.action_dim:
                                adapted_action = adapted_action + [0.0] * (self.action_dim - len(adapted_action))
                            elif len(adapted_action) > self.action_dim:
                                adapted_action = adapted_action[:self.action_dim]
                            
                            compatible_experiences.append((adapted_version, adapted_action, reward, adapted_next_version, done))
                
                # Clear buffer and add compatible experiences back
                self.replay_buffer.clear()
                for exp in compatible_experiences:
                    self.replay_buffer.add(*exp)
                
                print(f"Preserved {len(compatible_experiences)} compatible experiences in replay buffer")
                
                # Adapt prev_version to new dimensions
                if self.prev_version is not None and len(self.prev_version) > 0:
                    if len(self.prev_version) < self.version_dim:
                        self.prev_version = self.prev_version + [0.0] * (self.version_dim - len(self.prev_version))
                    elif len(self.prev_version) > self.version_dim:
                        self.prev_version = self.prev_version[:self.version_dim]
                    print(f"Adapted prev_version to new dimensions: {len(self.prev_version)}")
        
        try:
            # Update agent counts
            agent_counts = {version: 0 for version in current_versions}
            for agent in agents:
                agent_version = agent.declared_version[self.attribute_idx]
                agent_counts[agent_version] = agent_counts.get(agent_version, 0) + 1
            self.agent_counts = agent_counts
            
            # Get current version representation
            current_version = self.get_version_representation(agents, current_versions)
            
            # Calculate diversity ratio
            diversity_ratio = 0
            if current_diversity is not None and ideal_diversity is not None and ideal_diversity > 0:
                diversity_ratio = current_diversity / ideal_diversity

            if diversity_ratio > self.best_diversity_ratio:
                self.best_diversity_ratio = diversity_ratio
                self.best_rewards = self.current_rewards.copy()
                self.best_max_reward = self.max_reward  # Store best max_reward
                print(f"New best diversity ratio: {diversity_ratio:.3f}, max_reward: {self.max_reward:.1f}")
            
            # Check for stability condition
            if diversity_ratio > 0.995:
                if self.steps_done % 100 == 0:
                    print(f"Diversity ratio {diversity_ratio:.3f} > 0.995 - Applying gentle reward reduction")
                
                # Apply gentle reward reduction while maintaining relative proportions
                reduction_factor =1  # Reduce by 0.1% each step
                
                # Reduce all rewards proportionally while maintaining their relative ratios
                for version in current_versions:
                    if version != 'NO_version' and self.current_rewards[version] > 0:
                        self.current_rewards[version] *= reduction_factor
                
                # Also reduce max_reward slightly to maintain consistency
                self.max_reward *= reduction_factor
                
                # Apply current rewards
                version_rewards, total_rewards = self.update_rewards_direct(agents, current_versions, version_rewards)
                self.prev_total_rewards = total_rewards
                
                # Log the reduction periodically
                if self.steps_done % 1000 == 0:
                    controllable_rewards = {s: r for s, r in self.current_rewards.items() if s != 'NO_version'}
                    avg_reward = np.mean(list(controllable_rewards.values())) if controllable_rewards else 0
                    print(f"Gentle reduction applied - Avg reward: {avg_reward:.1f}, Max bound: {self.max_reward:.1f}, Total: {total_rewards:.1f}")
                
                if len(current_version) == self.version_dim:
                    self.prev_version = current_version 
                self.prev_diversity = current_diversity if current_diversity is not None else self.prev_diversity
                self.steps_done += 1
                
                return version_rewards
            
            # Normal RL operation
            if self.prev_version is None or versions_changed:
                self.prev_version = current_version
                self.prev_diversity = current_diversity if current_diversity is not None else 0
                # Apply current rewards when versions change
                version_rewards, total_rewards = self.update_rewards_direct(agents, current_versions, version_rewards)
                self.prev_total_rewards = total_rewards
            else:
                actions = self.select_action(current_version)
                
                # Apply actions to rewards and max_reward
                self.apply_actions_to_rewards(actions, current_versions)
                
                # Calculate reward and update if diversity metrics available
                if current_diversity is not None and ideal_diversity is not None:
                    # Apply rewards
                    version_rewards, total_rewards = self.update_rewards_direct(agents, current_versions, version_rewards)
                    
                    # Calculate RL training reward
                    reward = self.calculate_reward(current_diversity, ideal_diversity, total_rewards)
                    self.ideal_diversity = ideal_diversity
                    
                    # Store experience
                    done = False
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
                    
                    self.prev_total_rewards = total_rewards
                    
                    # Store best rewards if diversity improved
                    if diversity_ratio > self.best_diversity_ratio:
                        self.best_diversity_ratio = diversity_ratio
                        self.best_rewards = self.current_rewards.copy()
                        self.best_max_reward = self.max_reward
                        print(f"New best diversity ratio: {diversity_ratio:.3f}")
                        print(f"Current rewards: {', '.join([f'{s}: {r:.1f}' for s, r in self.current_rewards.items() if s != 'NO_version'])}")
                        print(f"Max reward: {self.max_reward:.1f}")
                else:
                    # No diversity metrics, just apply rewards
                    version_rewards, total_rewards = self.update_rewards_direct(agents, current_versions, version_rewards)
                    self.prev_total_rewards = total_rewards
            
            # Print status periodically
            if self.steps_done % 1000 == 0:
                controllable_rewards = {s: r for s, r in self.current_rewards.items() if s != 'NO_version'}
                avg_reward = np.mean(list(controllable_rewards.values())) if controllable_rewards else 0
                min_reward = min(controllable_rewards.values()) if controllable_rewards else 0
                max_single_reward = max(controllable_rewards.values()) if controllable_rewards else 0
                print(f"Direct rewards - Avg: {avg_reward:.1f}, Min: {min_reward:.1f}, Max: {max_single_reward:.1f}, Max bound: {self.max_reward:.1f}, Total: {self.prev_total_rewards:.1f}, Diversity: {diversity_ratio:.3f}, Epsilon: {self.epsilon:.3f}")
            
            # Update version and action for next iteration
            if len(current_version) == self.version_dim:
                self.prev_version = current_version 
            self.prev_action = actions if 'actions' in locals() else None
            self.prev_diversity = current_diversity if current_diversity is not None else self.prev_diversity
            
            # Increment step counter
            self.steps_done += 1
            
            return version_rewards
            
        except Exception as e:
            print(f"Error in update_rewards: {e}")
            import traceback
            traceback.print_exc()
            return version_rewards