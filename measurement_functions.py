import matplotlib.pyplot as plt
import math
import numpy as np

def plot_multiple_experiment_results(experiment_indices, metrics, metric_name, y_label=None, title=None, ylim=None):
    """
    Plot metrics across multiple experiments.
    
    Args:
        experiment_indices: List of experiment indices (x-axis)
        metrics: List of metric values to plot
        metric_name: Name of the metric for the legend
        y_label: Label for y-axis
        title: Plot title
        ylim: Optional tuple for y-axis limits (min, max)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(experiment_indices, metrics, 'o-', linewidth=2, markersize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Experiment Number')
    plt.ylabel(y_label if y_label else metric_name)
    plt.title(title if title else f'{metric_name} Across Experiments')
    
    # Add values as text above points
    for i, value in enumerate(metrics):
        plt.annotate(f'{value:.4f}', 
                    (experiment_indices[i], value),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    # Set y-axis limits if provided
    if ylim:
        plt.ylim(ylim)
        
    # Ensure x-axis shows all experiment numbers
    plt.xticks(experiment_indices)
    
    # Optimize for display
    plt.tight_layout()
    
    return plt

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
            
            # Fix: Handle division by zero case
            if before_change > 0:
                results['overall']['adaptation_shock'] = (before_change - after_change) / before_change
            else:
                # If before_change is 0, we can't calculate a meaningful percentage drop
                results['overall']['adaptation_shock'] = 0 if after_change == 0 else 1
        
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
        # Fix: Handle division by zero
        if pre_change_avg > 0:
            initial_drop = drop / pre_change_avg
        else:
            initial_drop = 0 if phase2_values[0] == 0 else 1
    
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
            # Fix: Handle zero values in the window
            avg = sum(window) / len(window)
            if avg > 0:
                variation = max(abs(v - avg) / avg for v in window)
                
                if variation < settling_threshold:
                    settling_time = i
                    break
    
    # Calculate final adaptation quality
    final_periods = min(20, len(phase2_values))
    if final_periods > 0:
        final_avg = sum(phase2_values[-final_periods:]) / final_periods
        # Fix: Handle division by zero
        if ideal_after > 0:
            adaptation_quality = final_avg / ideal_after
        else:
            adaptation_quality = 0
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

"""
Enhanced metrics for comparing PID and RL controllers.
These functions will be added to measurement_functions.py
"""

def calculate_convergence_metrics(diversity_values, ideal_diversity, tolerance=0.9, window_size=20):
    """
    Calculate comprehensive convergence metrics for a system.
    
    Parameters:
    -----------
    diversity_values : list
        Time series of diversity values
    ideal_diversity : float
        The ideal diversity value to reach
    tolerance : float, optional (default=0.9)
        Fraction of ideal diversity that must be reached (0.9 = 90% of ideal)
    window_size : int, optional (default=20)
        Number of consecutive epochs that must stay above threshold
        
    Returns:
    --------
    dict
        Dictionary containing convergence metrics
    """
    threshold = ideal_diversity * tolerance
    converged = False
    convergence_epoch = None
    
    # Time to convergence - when diversity stays above threshold for window_size epochs
    for i in range(len(diversity_values) - window_size + 1):
        window = diversity_values[i:i+window_size]
        if all(d >= threshold for d in window):
            converged = True
            convergence_epoch = i
            break
    
    # Calculate convergence rate (how quickly system approaches target)
    convergence_rate = None
    if convergence_epoch is not None and convergence_epoch > 10:
        # Use an exponential fit to the pre-convergence data
        # to estimate the convergence rate
        try:
            import numpy as np
            from scipy.optimize import curve_fit
            
            def exp_func(x, a, b, c):
                return a * (1 - np.exp(-b * x)) + c
            
            x_data = np.arange(convergence_epoch)
            y_data = np.array(diversity_values[:convergence_epoch])
            
            # Only attempt curve fitting if we have enough data points
            if len(x_data) > 5:
                # Initial parameter guesses
                p0 = [ideal_diversity, 0.1, 0]
                try:
                    popt, _ = curve_fit(exp_func, x_data, y_data, p0=p0, maxfev=5000)
                    convergence_rate = popt[1]  # Extract the rate parameter
                except:
                    # If curve fitting fails, fall back to simpler method
                    convergence_rate = None
        except ImportError:
            # If scipy is not available, use a simpler method
            convergence_rate = None
    
    # If scipy curve fitting failed or wasn't available, calculate a simpler rate
    if convergence_rate is None and convergence_epoch is not None and convergence_epoch > 5:
        # Calculate average rate of increase in first portion of convergence
        initial_value = diversity_values[0]
        midpoint = min(convergence_epoch // 2, len(diversity_values) - 1)
        if midpoint > 0:
            midpoint_value = diversity_values[midpoint]
            convergence_rate = (midpoint_value - initial_value) / midpoint
    
    # Calculate steady-state error after convergence
    steady_state_error = None
    steady_state_stability = None
    
    if converged and convergence_epoch < len(diversity_values) - window_size:
        post_convergence = diversity_values[convergence_epoch:]
        mean_post = sum(post_convergence) / len(post_convergence)
        steady_state_error = (ideal_diversity - mean_post) / ideal_diversity
        
        # Calculate stability (coefficient of variation in steady state)
        variance = sum((x - mean_post) ** 2 for x in post_convergence) / len(post_convergence)
        std_dev = variance ** 0.5
        steady_state_stability = std_dev / mean_post if mean_post > 0 else float('inf')
    
    # Return comprehensive metrics
    return {
        "time_to_convergence": convergence_epoch,
        "convergence_rate": convergence_rate,
        "steady_state_error": steady_state_error, 
        "steady_state_stability": steady_state_stability,
        "final_diversity_quality": diversity_values[-1] / ideal_diversity if ideal_diversity > 0 else 0,
        "converged": converged
    }

def calculate_resilience_metrics(diversity_values, transition_epoch, ideal_before, ideal_after, 
                              recovery_threshold=0.9):
    """
    Calculate comprehensive resilience metrics for a system responding to a new state.
    
    Parameters:
    -----------
    diversity_values : list
        Time series of diversity values
    transition_epoch : int
        Epoch at which the new state was introduced
    ideal_before : float
        Ideal diversity before transition
    ideal_after : float
        Ideal diversity after transition
    recovery_threshold : float, optional (default=0.9)
        Fraction of pre-transition quality that must be recovered
        
    Returns:
    --------
    dict
        Dictionary with resilience metrics
    """
    # Ensure we have enough data
    if transition_epoch >= len(diversity_values) or transition_epoch < 5:
        return {
            "error": "Insufficient data for analysis"
        }
    
    # Extract pre-transition and post-transition diversity values
    pre_values = diversity_values[:transition_epoch]
    post_values = diversity_values[transition_epoch:]
    
    # Calculate pre-transition baseline (average of last 20% of pre-transition)
    baseline_window = max(5, int(len(pre_values) * 0.2))
    pre_baseline = sum(pre_values[-baseline_window:]) / baseline_window
    pre_quality = pre_baseline / ideal_before if ideal_before > 0 else 0
    
    # Calculate immediate impact
    if post_values:
        initial_post = post_values[0]
        impact_magnitude = pre_baseline - initial_post
        impact_percentage = impact_magnitude / pre_baseline if pre_baseline > 0 else 0
    else:
        impact_magnitude = None
        impact_percentage = None
    
    # Calculate recovery time (time to reach recovery_threshold of pre-quality adjusted for new ideal)
    recovery_target = pre_quality * recovery_threshold * ideal_after / ideal_before
    absolute_recovery_target = recovery_target * ideal_after
    
    recovery_time = None
    for i, val in enumerate(post_values):
        if val >= absolute_recovery_target:
            recovery_time = i
            break
    
    # Calculate re-convergence time (time to reach 90% of new ideal diversity)
    reconvergence_target = 0.9 * ideal_after
    reconvergence_time = None
    for i, val in enumerate(post_values):
        if val >= reconvergence_target:
            reconvergence_time = i
            break
    
    # Calculate final adaptation quality
    if post_values:
        # Use the last 20% of post-transition values for final assessment
        final_window = max(5, int(len(post_values) * 0.2))
        final_avg = sum(post_values[-final_window:]) / final_window
        final_quality = final_avg / ideal_after if ideal_after > 0 else 0
    else:
        final_quality = None
    
    # Calculate return to stability (coefficient of variation less than 0.05)
    stable_window = 20  # Window size to check stability
    stability_threshold = 0.05  # CV threshold for stability
    
    stability_time = None
    if len(post_values) > stable_window:
        for i in range(len(post_values) - stable_window + 1):
            window = post_values[i:i+stable_window]
            mean_window = sum(window) / stable_window
            variance = sum((x - mean_window) ** 2 for x in window) / stable_window
            cv = (variance ** 0.5) / mean_window if mean_window > 0 else float('inf')
            
            if cv < stability_threshold:
                stability_time = i
                break
    
    return {
        "pre_transition_quality": pre_quality,
        "initial_impact_magnitude": impact_magnitude,
        "initial_impact_percentage": impact_percentage * 100 if impact_percentage is not None else None,
        "recovery_time_epochs": recovery_time,
        "reconvergence_time_epochs": reconvergence_time,
        "stability_time_epochs": stability_time,
        "final_adaptation_quality": final_quality
    }

def calculate_resource_efficiency(diversity_values, rewards_history, ideal_diversity):
    """
    Calculate resource efficiency metrics - how effectively rewards are used to achieve diversity.
    
    Parameters:
    -----------
    diversity_values : list
        Time series of diversity values
    rewards_history : list
        Time series of total rewards allocated
    ideal_diversity : float
        The ideal diversity target
        
    Returns:
    --------
    dict
        Dictionary with resource efficiency metrics
    """
    # Ensure inputs have matching lengths
    min_length = min(len(diversity_values), len(rewards_history))
    diversity_values = diversity_values[:min_length]
    rewards_history = rewards_history[:min_length]
    
    if min_length == 0:
        return {
            "error": "No data available for analysis"
        }
    
    # Calculate total reward usage
    total_rewards = sum(rewards_history)
    avg_reward_per_epoch = total_rewards / min_length
    
    # Calculate absolute reward values (since negative values can offset positive ones)
    absolute_rewards = [abs(r) for r in rewards_history]
    total_absolute_rewards = sum(absolute_rewards)
    avg_absolute_reward = total_absolute_rewards / min_length
    
    # Calculate average diversity achieved
    avg_diversity = sum(diversity_values) / min_length
    
    # Calculate resource efficiency (diversity per unit of reward)
    efficiency = avg_diversity / avg_absolute_reward if avg_absolute_reward > 0 else float('inf')
    
    # Calculate normalized efficiency (as percentage of ideal)
    normalized_efficiency = (avg_diversity / ideal_diversity) / avg_absolute_reward * 100 if ideal_diversity > 0 and avg_absolute_reward > 0 else 0
    
    # Calculate reward volatility
    reward_variance = sum((r - avg_reward_per_epoch) ** 2 for r in rewards_history) / min_length
    reward_volatility = (reward_variance ** 0.5) / avg_reward_per_epoch if avg_reward_per_epoch != 0 else float('inf')
    
    # Calculate efficiency over time (improving or degrading?)
    if min_length >= 20:
        # Compare first 25% vs last 25%
        quarter_length = min_length // 4
        
        # First quarter metrics
        first_quarter_div = sum(diversity_values[:quarter_length]) / quarter_length
        first_quarter_rew = sum(absolute_rewards[:quarter_length]) / quarter_length
        first_quarter_eff = first_quarter_div / first_quarter_rew if first_quarter_rew > 0 else 0
        
        # Last quarter metrics
        last_quarter_div = sum(diversity_values[-quarter_length:]) / quarter_length
        last_quarter_rew = sum(absolute_rewards[-quarter_length:]) / quarter_length
        last_quarter_eff = last_quarter_div / last_quarter_rew if last_quarter_rew > 0 else 0
        
        # Calculate efficiency trend
        if first_quarter_eff > 0:
            efficiency_trend = (last_quarter_eff - first_quarter_eff) / first_quarter_eff
        else:
            efficiency_trend = float('inf') if last_quarter_eff > 0 else 0
    else:
        efficiency_trend = None
    
    return {
        "avg_reward_per_epoch": avg_reward_per_epoch,
        "avg_absolute_reward": avg_absolute_reward,
        "total_cumulative_reward": total_rewards,
        "total_absolute_reward": total_absolute_rewards,
        "diversity_per_reward_unit": efficiency,
        "normalized_efficiency": normalized_efficiency,
        "reward_volatility": reward_volatility,
        "efficiency_trend": efficiency_trend
    }

def analyze_pid_parameter_evolution(pid_params_history, diversity_values, transition_epoch=None):
    """
    Analyze how PID parameters evolve and affect system performance (for RL-tuned PID).
    
    Parameters:
    -----------
    pid_params_history : list
        Time series of tuples (P, I, D) representing PID parameters at each epoch
    diversity_values : list
        Time series of diversity values
    transition_epoch : int, optional
        Epoch at which a new state was introduced
        
    Returns:
    --------
    dict
        Dictionary with PID parameter analysis metrics
    """
    # Ensure we have enough data
    if len(pid_params_history) < 10:
        return {
            "error": "Insufficient data for analysis"
        }
    
    # Extract P, I, D parameter series
    p_values = [params[0] for params in pid_params_history]
    i_values = [params[1] for params in pid_params_history]
    d_values = [params[2] for params in pid_params_history]
    
    # Calculate parameter stability (coefficient of variation)
    def calculate_stability(values):
        if not values:
            return None
        mean = sum(values) / len(values)
        if mean == 0:
            return float('inf')
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return (variance ** 0.5) / mean
    
    # Overall parameter stability
    p_stability = calculate_stability(p_values)
    i_stability = calculate_stability(i_values)
    d_stability = calculate_stability(d_values)
    
    # Calculate parameter ratios over time
    pid_ratios = []
    for p, i, d in pid_params_history:
        total = p + i + d
        if total > 0:
            pid_ratios.append((p/total, i/total, d/total))
        else:
            pid_ratios.append((0, 0, 0))
    
    # Calculate final parameter composition
    final_p, final_i, final_d = pid_params_history[-1]
    final_total = final_p + final_i + final_d
    if final_total > 0:
        final_composition = {
            "p_percentage": (final_p / final_total) * 100,
            "i_percentage": (final_i / final_total) * 100,
            "d_percentage": (final_d / final_total) * 100
        }
    else:
        final_composition = {
            "p_percentage": 0,
            "i_percentage": 0,
            "d_percentage": 0
        }
    
    # If transition epoch is provided, analyze adaptation speed
    if transition_epoch is not None and transition_epoch < len(pid_params_history) - 10:
        # Compare parameter change speed after transition
        pre_transition = pid_params_history[max(0, transition_epoch-10):transition_epoch]
        post_transition = pid_params_history[transition_epoch:transition_epoch+10]
        
        # Calculate average rate of change before and after
        def calculate_change_rate(params_list):
            if len(params_list) < 2:
                return (0, 0, 0)
            
            p_changes = [abs(params_list[i+1][0] - params_list[i][0]) for i in range(len(params_list)-1)]
            i_changes = [abs(params_list[i+1][1] - params_list[i][1]) for i in range(len(params_list)-1)]
            d_changes = [abs(params_list[i+1][2] - params_list[i][2]) for i in range(len(params_list)-1)]
            
            return (
                sum(p_changes) / len(p_changes) if p_changes else 0,
                sum(i_changes) / len(i_changes) if i_changes else 0,
                sum(d_changes) / len(d_changes) if d_changes else 0
            )
        
        pre_change_rate = calculate_change_rate(pre_transition)
        post_change_rate = calculate_change_rate(post_transition)
        
        adaptation_speed = {
            "p_adaptation_ratio": post_change_rate[0] / pre_change_rate[0] if pre_change_rate[0] > 0 else float('inf'),
            "i_adaptation_ratio": post_change_rate[1] / pre_change_rate[1] if pre_change_rate[1] > 0 else float('inf'),
            "d_adaptation_ratio": post_change_rate[2] / pre_change_rate[2] if pre_change_rate[2] > 0 else float('inf')
        }
    else:
        adaptation_speed = None
    
    # Parameter-performance correlation
    # Calculate correlation between parameter changes and diversity improvements
    if len(diversity_values) >= len(pid_params_history):
        diversity_values = diversity_values[:len(pid_params_history)]
        
        diversity_changes = [diversity_values[i+1] - diversity_values[i] for i in range(len(diversity_values)-1)]
        p_changes = [p_values[i+1] - p_values[i] for i in range(len(p_values)-1)]
        i_changes = [i_values[i+1] - i_values[i] for i in range(len(i_values)-1)]
        d_changes = [d_values[i+1] - d_values[i] for i in range(len(d_values)-1)]
        
        # Calculate Pearson correlation
        def calculate_correlation(x, y):
            if len(x) != len(y) or len(x) < 2:
                return None
            
            x_mean = sum(x) / len(x)
            y_mean = sum(y) / len(y)
            
            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
            denom_x = sum((val - x_mean) ** 2 for val in x) ** 0.5
            denom_y = sum((val - y_mean) ** 2 for val in y) ** 0.5
            
            if denom_x > 0 and denom_y > 0:
                return numerator / (denom_x * denom_y)
            else:
                return None
        
        parameter_effectiveness = {
            "p_effectiveness": calculate_correlation(p_changes, diversity_changes),
            "i_effectiveness": calculate_correlation(i_changes, diversity_changes),
            "d_effectiveness": calculate_correlation(d_changes, diversity_changes)
        }
    else:
        parameter_effectiveness = None
    
    return {
        "parameter_stability": {
            "p_stability": p_stability,
            "i_stability": i_stability,
            "d_stability": d_stability
        },
        "final_composition": final_composition,
        "adaptation_speed": adaptation_speed,
        "parameter_effectiveness": parameter_effectiveness
    }

def compile_system_performance_metrics(diversity_values, rewards_history, pid_params_history, 
                                     transition_epoch, ideal_before, ideal_after):
    """
    Compile all performance metrics into a single comprehensive report.
    
    Parameters:
    -----------
    diversity_values : list
        Time series of diversity values
    rewards_history : list
        Time series of total rewards allocated
    pid_params_history : list or None
        Time series of tuples (P, I, D) representing PID parameters (for RL mode)
    transition_epoch : int
        Epoch at which the new state was introduced
    ideal_before : float
        Ideal diversity before transition
    ideal_after : float
        Ideal diversity after transition
        
    Returns:
    --------
    dict
        Dictionary with all performance metrics
    """
    # Calculate Phase 1 convergence (before transition)
    phase1_values = diversity_values[:transition_epoch]
    phase1_convergence = calculate_convergence_metrics(phase1_values, ideal_before)
    
    # Calculate Phase 2 convergence (after transition)
    phase2_values = diversity_values[transition_epoch:]
    phase2_convergence = calculate_convergence_metrics(phase2_values, ideal_after)
    
    # Calculate resilience metrics
    resilience = calculate_resilience_metrics(diversity_values, transition_epoch, ideal_before, ideal_after)
    
    # Calculate resource efficiency
    # For phase 1
    phase1_rewards = rewards_history[:transition_epoch] if transition_epoch <= len(rewards_history) else rewards_history
    phase1_efficiency = calculate_resource_efficiency(phase1_values, phase1_rewards, ideal_before)
    
    # For phase 2
    phase2_rewards = rewards_history[transition_epoch:] if transition_epoch < len(rewards_history) else []
    phase2_efficiency = calculate_resource_efficiency(phase2_values, phase2_rewards, ideal_after)
    
    # For overall
    overall_efficiency = calculate_resource_efficiency(diversity_values, rewards_history, 
                                                     ideal_after)  # Use final ideal as reference
    
    # Calculate PID parameter evolution metrics if available
    pid_evolution = None
    if pid_params_history is not None and len(pid_params_history) > 0:
        pid_evolution = analyze_pid_parameter_evolution(pid_params_history, diversity_values, transition_epoch)
    
    # Compile all metrics
    return {
        "phase1_convergence": phase1_convergence,
        "phase2_convergence": phase2_convergence,
        "resilience": resilience,
        "phase1_efficiency": phase1_efficiency,
        "phase2_efficiency": phase2_efficiency,
        "overall_efficiency": overall_efficiency,
        "pid_evolution": pid_evolution
    }