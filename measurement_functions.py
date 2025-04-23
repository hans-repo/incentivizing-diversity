import matplotlib.pyplot as plt
import math
import numpy as np

def plot_stacked_area(agents_history, epochs, possible_states):
    """
    Plot a stacked area chart showing the distribution of agents across states over time.
    Modified to handle dynamic state sets properly and avoid empty plots.
    
    Args:
        agents_history: List of agent states for each epoch
        epochs: Number of epochs to plot
        possible_states: List of all possible states (including those added during the simulation)
    """
    
    # Use only the available epochs
    available_epochs = min(len(agents_history), epochs)
    
    if available_epochs == 0:
        print("Warning: No agent history data available to plot")
        plt.figure(figsize=(10, 6))
        plt.title("No Agent Data Available")
        return plt
    
    # Initialize state counts dictionary with all possible states
    state_counts = {state: np.zeros(available_epochs) for state in possible_states}
    
    # Count agents in each state for each epoch
    for epoch in range(available_epochs):
        # Skip if there's no data for this epoch
        if epoch >= len(agents_history) or not agents_history[epoch]:
            continue
            
        # Count agents in each state for this epoch
        for agent_state in agents_history[epoch]:
            if agent_state in state_counts:
                state_counts[agent_state][epoch] += 1
    
    
    # Filter out NO_STATE and find states that actually have agents
    states_to_plot = []
    for state in possible_states:
        if state != 'NO_STATE' and np.any(state_counts[state] > 0):
            states_to_plot.append(state)
    
    # If no states have any agents, show an empty plot with a message
    if not states_to_plot:
        plt.title("No Agents Found in Any State")
        plt.xlabel('Epoch')
        plt.ylabel('Number of Agents')
        return plt
    
    # Create a consistent color map for states
    cmap = plt.cm.get_cmap('tab10', len(states_to_plot) + 1)  # +1 to avoid repeating first color
    colors = [cmap(i) for i in range(len(states_to_plot))]
    
    # Create the stacked area plot
    plt.stackplot(range(available_epochs),
                 [state_counts[state] for state in states_to_plot],
                 labels=states_to_plot,
                 colors=colors,
                 alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Number of Agents')
    plt.title('Evolution of Agent States Over Epochs')
    
    # Only add legend if we have states to plot
    if states_to_plot:
        plt.legend(loc='upper right')
        
    plt.grid(True, linestyle='--', alpha=0.5)

    # Set x-ticks to show reasonable intervals
    step_size = max(1, available_epochs // 10)
    plt.xticks(range(0, available_epochs + 1, step_size))
    
    return plt


def calculate_diversity(agents_real_history, epochs, possible_states, n_agents):
    """
    Calculate diversity metrics for each epoch based on Shannon entropy.
    Modified to handle dynamic state sets by using the state set specific to each epoch.
    
    Args:
        agents_real_history: List of agent states for each attribute and epoch
        epochs: List of epochs to calculate diversity for
        possible_states: List/set of possible states for each epoch (can be dynamic)
        n_agents: Total number of agents
        
    Returns:
        Dictionary of diversity values for each attribute, epoch
    """
    NUM_ATTRIBUTES = len(agents_real_history)
    diversity_all_epoch = [{epoch: [] for epoch in epochs} for _ in range(NUM_ATTRIBUTES)]
    
    for k in range(NUM_ATTRIBUTES):
        for epoch in epochs:
            # Skip if we don't have data for this epoch
            if epoch >= len(agents_real_history[k]):
                continue
                
            # Get the correct set of states for this epoch (excluding NO_STATE)
            valid_states = [s for s in possible_states if s != 'NO_STATE']
            
            # Calculate Shannon entropy for this epoch
            diversity_this_epoch = 0
            state_counts = {state: 0 for state in valid_states}
            
            # Count agents in each state
            for agent_state in agents_real_history[k][epoch]:
                if agent_state in state_counts:
                    state_counts[agent_state] += 1
            
            # Calculate entropy components for each state
            for state in valid_states:
                count = state_counts[state]
                if count > 0:
                    # Shannon entropy calculation: -p*log2(p) for each state
                    p = count / n_agents
                    diversity_this_epoch -= p * math.log2(p)
            
            diversity_all_epoch[k][epoch].append(diversity_this_epoch)
            
    return diversity_all_epoch


def get_largest_state(agents_history, epochs, possible_states, n_agents):
    """
    Calculate the proportion of agents in the largest state for each epoch.
    Modified to handle dynamic state sets.
    
    Args:
        agents_history: List of agent states for each attribute and epoch
        epochs: List of epochs to calculate for
        possible_states: List/set of possible states (can be dynamic)
        n_agents: Total number of agents
        
    Returns:
        Dictionary of largest state proportions for each attribute, epoch
    """
    NUM_ATTRIBUTES = len(agents_history)
    largest_state_all_epochs = [{epoch: [] for epoch in epochs} for _ in range(NUM_ATTRIBUTES)]
    
    for k in range(NUM_ATTRIBUTES):
        for epoch in epochs:
            # Skip if we don't have data for this epoch
            if epoch >= len(agents_history[k]):
                continue
                
            # Get valid states for this epoch (excluding NO_STATE)
            valid_states = [s for s in possible_states if s != 'NO_STATE']
            
            # Find the largest state for this epoch
            largest_state_this_epoch = 0
            state_counts = {state: 0 for state in valid_states}
            
            # Count agents in each state
            for agent_state in agents_history[k][epoch]:
                if agent_state in state_counts:
                    state_counts[agent_state] += 1
            
            # Find the largest count
            largest_state_this_epoch = max(state_counts.values()) if state_counts else 0
            
            # Record as a proportion of total agents
            largest_state_all_epochs[k][epoch].append(largest_state_this_epoch / n_agents)
            
    return largest_state_all_epochs


def get_ideal_diversity(possible_states):
    """
    Calculate the ideal (maximum) diversity for a given set of states.
    Modified to handle dynamic state sets.
    
    Args:
        possible_states: List/set of possible states (including NO_STATE)
        
    Returns:
        Ideal diversity value (Shannon entropy)
    """
    # Count valid states (excluding NO_STATE)
    valid_states = [s for s in possible_states if s != 'NO_STATE']
    num_valid_states = len(valid_states)
    
    if num_valid_states <= 1:
        return 0.0  # No diversity possible with 0 or 1 valid states
    
    # Ideal distribution is equal probability across all valid states
    p = 1 / num_valid_states
    
    # Shannon entropy calculation: -p*log2(p) for each state
    entropy = -num_valid_states * (p * math.log2(p))
    
    return entropy


def get_avg_loss(NUM_ATTRIBUTES, average_last_25_percent, ideal_diversity_all_experiments):
    """
    Calculate the average loss in diversity compared to the ideal.
    
    Args:
        NUM_ATTRIBUTES: Number of attributes being tracked
        average_last_25_percent: Average diversity in the last 25% of epochs
        ideal_diversity_all_experiments: Ideal diversity values
        
    Returns:
        Average proportional loss across all attributes
    """
    total = 0
    for k in range(NUM_ATTRIBUTES):
        # Skip if ideal diversity is 0 to avoid division by zero
        if ideal_diversity_all_experiments[k][0] == 0:
            continue
            
        proportional_loss = abs(ideal_diversity_all_experiments[k][0] - average_last_25_percent[k]) / ideal_diversity_all_experiments[k][0]
        total += proportional_loss
        
    average_loss = total / NUM_ATTRIBUTES
    return average_loss


def plot_diversity_over_time(diversity_data, epochs, ideal_diversity_before, ideal_diversity_after, 
                            new_state_epoch, new_state_name, pid_params=None):
    """
    Plot diversity metrics over time with annotations for state changes.
    
    Args:
        diversity_data: Dictionary of diversity values for each epoch
        epochs: Total number of epochs
        ideal_diversity_before: Ideal diversity before new state was added
        ideal_diversity_after: Ideal diversity after new state was added
        new_state_epoch: Epoch when new state was added
        new_state_name: Name of the new state
        pid_params: Optional tuple of (P, I, D) values for title
    """
    plt.figure(figsize=(12, 7))
    
    # Plot diversity for each attribute
    for k in range(len(diversity_data)):
        # Extract diversity values from dictionary
        epochs_list = range(epochs)
        diversity_values = [diversity_data[k][epoch][0] for epoch in epochs_list if epoch in diversity_data[k]]
        
        plt.plot(epochs_list[:len(diversity_values)], diversity_values, 
                 label=f"Diversity - Attribute {k}")
    
    # Add vertical line for new state addition
    plt.axvline(x=new_state_epoch, color='r', linestyle='--', 
                label=f"New state ({new_state_name}) added")
    
    # Add horizontal lines for ideal diversity
    plt.axhline(y=ideal_diversity_before, color='g', linestyle=':', 
                label=f"Ideal diversity - before")
    plt.axhline(y=ideal_diversity_after, color='g', linestyle='-', 
                label=f"Ideal diversity - after")
    
    # Title with PID parameters if provided
    if pid_params:
        p, i, d = pid_params
        plt.title(f'Diversity Evolution with PID (P={p:.1f}, I={i:.1f}, D={d:.1f})')
    else:
        plt.title('Diversity Evolution with Dynamic State Addition')
        
    plt.xlabel('Epochs')
    plt.ylabel('Diversity (Shannon Entropy)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    return plt


def analyze_convergence(diversity_data, epochs, ideal_diversity, window_size=50):
    """
    Analyze how quickly the system converges to steady-state diversity.
    
    Args:
        diversity_data: Dictionary of diversity values for each epoch
        epochs: Total number of epochs
        ideal_diversity: Ideal diversity value
        window_size: Window size for moving average
        
    Returns:
        Dictionary with convergence metrics
    """
    results = {}
    
    for k in range(len(diversity_data)):
        # Extract diversity values
        epochs_list = range(epochs)
        diversity_values = np.array([diversity_data[k][epoch][0] for epoch in epochs_list 
                                    if epoch in diversity_data[k]])
        
        # Calculate moving average to smooth out oscillations
        if len(diversity_values) > window_size:
            smoothed = np.convolve(diversity_values, np.ones(window_size)/window_size, mode='valid')
            
            # Calculate distance from ideal at each point
            distance = np.abs(smoothed - ideal_diversity) / ideal_diversity
            
            # Find first point where distance is consistently below threshold (5%)
            threshold = 0.05
            converged_at = None
            
            for i in range(len(distance) - window_size//2):
                if np.all(distance[i:i+window_size//2] < threshold):
                    converged_at = i + window_size//2
                    break
            
            results[k] = {
                'converged_at': converged_at,
                'final_distance': distance[-1] if len(distance) > 0 else None,
                'avg_last_10pct': np.mean(diversity_values[-len(diversity_values)//10:])
            }
        else:
            results[k] = {
                'converged_at': None,
                'final_distance': None,
                'avg_last_10pct': np.mean(diversity_values[-max(1, len(diversity_values)//10):]) 
                                   if len(diversity_values) > 0 else None
            }
            
    return results

def measure_dual_phase_convergence(diversity_values, transition_epoch, 
                                  ideal_before, ideal_after, target_percentage=0.9):
    """
    Measures convergence in a two-phase system with changing ideal diversity.
    
    Args:
        diversity_values: List of diversity values for each epoch
        transition_epoch: Epoch where new state is added
        ideal_before: Ideal diversity before state addition
        ideal_after: Ideal diversity after state addition
        target_percentage: Target percentage of ideal diversity
        
    Returns:
        Dictionary with convergence metrics for both phases
    """
    results = {
        'phase1': {
            'target': ideal_before * target_percentage,
            'epochs_to_converge': None,
            'percentage_reached': 0
        },
        'phase2': {
            'target': ideal_after * target_percentage,
            'epochs_to_converge': None,
            'percentage_reached': 0,
            're_convergence_time': None  # How long after transition to reach target
        },
        'overall': {
            'adaptation_shock': None,  # How much diversity dropped at transition
            'recovery_time': None      # How long to recover from shock
        }
    }
    
    # Phase 1 convergence (before new state)
    phase1_values = diversity_values[:transition_epoch]
    if phase1_values:
        max_p1_diversity = max(phase1_values)
        results['phase1']['percentage_reached'] = max_p1_diversity / ideal_before
        
        for epoch, diversity in enumerate(phase1_values):
            if diversity >= results['phase1']['target']:
                results['phase1']['epochs_to_converge'] = epoch
                break
    
    # Phase 2 convergence (after new state)
    if transition_epoch < len(diversity_values):
        phase2_values = diversity_values[transition_epoch:]
        
        # Calculate adaptation shock (diversity drop at transition)
        if transition_epoch > 0 and len(phase2_values) > 0:
            before_change = diversity_values[transition_epoch-1]
            after_change = diversity_values[transition_epoch]
            results['overall']['adaptation_shock'] = (before_change - after_change) / before_change
        
        # Find when system reaches target in phase 2
        for epoch, diversity in enumerate(phase2_values):
            # Check if we've reached target after transition
            if diversity >= results['phase2']['target']:
                results['phase2']['epochs_to_converge'] = transition_epoch + epoch
                results['phase2']['re_convergence_time'] = epoch
                break
        
        # Calculate max percentage reached in phase 2
        if phase2_values:
            max_p2_diversity = max(phase2_values)
            results['phase2']['percentage_reached'] = max_p2_diversity / ideal_after
        
        # Calculate recovery time (how long to get back to pre-shock diversity level)
        if results['overall']['adaptation_shock'] is not None and transition_epoch > 0:
            pre_change_level = diversity_values[transition_epoch-1]
            for epoch, diversity in enumerate(phase2_values):
                if diversity >= pre_change_level:
                    results['overall']['recovery_time'] = epoch
                    break
    
    return results

def analyze_system_adaptability(diversity_values, transition_epoch, ideal_before, ideal_after):
    """
    Analyzes how well the system adapts to the introduction of a new state.
    
    Returns metrics about adaptation quality and speed.
    """
    # Skip if we don't have enough data
    if transition_epoch >= len(diversity_values) or transition_epoch < 10:
        return {"error": "Insufficient data for analysis"}
    
    # Get average diversity before transition (last 10% of phase 1)
    phase1_stable_period = max(1, int(transition_epoch * 0.1))
    pre_change_avg = sum(diversity_values[transition_epoch-phase1_stable_period:transition_epoch]) / phase1_stable_period
    
    # Get data for after transition
    phase2_values = diversity_values[transition_epoch:]
    
    # Skip if phase 2 is too short
    if len(phase2_values) < 20:
        return {"error": "Phase 2 too short for analysis"}
        
    # Initial drop percentage
    initial_drop = None
    if len(phase2_values) > 0:
        drop = pre_change_avg - phase2_values[0]
        initial_drop = drop / pre_change_avg
    
    # Time to recover from drop
    recovery_time = None
    for i, div in enumerate(phase2_values):
        if div >= pre_change_avg:
            recovery_time = i
            break
    
    # Time to reach 90% of new ideal
    target90 = ideal_after * 0.9
    time_to_90pct = None
    for i, div in enumerate(phase2_values):
        if div >= target90:
            time_to_90pct = i
            break
    
    # Calculate stabilization time in phase 2
    settling_window = 20
    settling_threshold = 0.05  # 5% variation
    settling_time = None
    
    if len(phase2_values) > settling_window:
        for i in range(len(phase2_values) - settling_window):
            window = phase2_values[i:i+settling_window]
            avg = sum(window) / len(window)
            variation = max(abs(v - avg) / avg for v in window)
            
            if variation < settling_threshold:
                settling_time = i
                break
    
    # Calculate final adaptation quality
    final_periods = min(20, len(phase2_values))
    if final_periods > 0:
        final_avg = sum(phase2_values[-final_periods:]) / final_periods
        adaptation_quality = final_avg / ideal_after
    else:
        adaptation_quality = None
        
    return {
        "pre_change_diversity_avg": pre_change_avg,
        "initial_drop_pct": initial_drop,
        "recovery_time_epochs": recovery_time,
        "time_to_90pct_new_ideal": time_to_90pct,
        "phase2_settling_time": settling_time,
        "final_adaptation_quality": adaptation_quality
    }