# Enhanced plotting_functions.py with additional functions moved from multiple_experiments.py

import numpy as np
import matplotlib.pyplot as plt
from measurement_functions import get_ideal_diversity

def plot_diversity_over_time(final_diversity, epochs_list, NUM_ATTRIBUTES, NEW_STATE_EPOCH, 
                           NEW_STATE_NAME, INITIAL_STATES, POSSIBLE_STATES, controller_type,
                           REWARDS_ADAPTIVE_PARAM=None, REWARDS_INTEGRAL_PARAM=None, REWARDS_DERIVATIVE_PARAM=None,
                           pid_params_history=None):
    """Plot diversity over time for all attributes."""
    plt.figure(figsize=(12, 7))
    for k in range(NUM_ATTRIBUTES):
        # Extract diversity values from dictionary
        diversity_values = [final_diversity[k][epoch][0] for epoch in epochs_list]  # [0] to get the first (and only) value
        plt.plot(epochs_list, diversity_values, label=f"Diversity - Attribute")
    
    # Add vertical line to mark when new state was added
    plt.axvline(x=NEW_STATE_EPOCH, color='r', linestyle='--', 
                label=f"New state added")
    
    # Add horizontal line for ideal diversity before and after new state
    ideal_before = get_ideal_diversity(INITIAL_STATES)
    ideal_after = get_ideal_diversity(POSSIBLE_STATES)
    plt.axhline(y=ideal_before, color='g', linestyle=':', 
                label=f"Ideal diversity - {len(INITIAL_STATES)-1} states")
    plt.axhline(y=ideal_after, color='g', linestyle='-', 
                label=f"Ideal diversity - {len(POSSIBLE_STATES)-1} states")
    
    plt.xlabel('Epochs')
    plt.ylabel('Diversity')
    
    # Adjust title based on controller type
    if controller_type == "pid":
        plt.title(f'Diversity with Ziegler-Nichols-tuned PID')
    elif controller_type == "rlnopid":
        plt.title(f'Diversity with Pure RL rewards')
    else:
        plt.title(f'Diversity with RL-tuned PID')
        
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_total_rewards(epochs_list, total_rewards_history, NEW_STATE_EPOCH, NEW_STATE_NAME, controller_type,
                     REWARDS_ADAPTIVE_PARAM=None, REWARDS_INTEGRAL_PARAM=None, REWARDS_DERIVATIVE_PARAM=None,
                     pid_params_history=None):
    """Plot total rewards over time."""
    plt.figure(figsize=(12, 7))
    plt.plot(epochs_list, total_rewards_history, 'b-', label='Total Rewards Allocated')
    
    # Add vertical line for when new state was added
    plt.axvline(x=NEW_STATE_EPOCH, color='r', linestyle='--', 
                label=f"New state added")
    
    plt.xlabel('Epochs')
    plt.ylabel('Total Rewards')
    
    # Adjust title based on controller type
    if controller_type == "pid":
        plt.title(f'Total Rewards Allocated with Ziegler-Nichols-tuned PID')
    elif controller_type == "rlnopid":
        plt.title(f'Total Rewards Allocated with Pure RL rewards')
    else:
        plt.title(f'Total Rewards Allocated with RL-tuned PID')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_rewards_per_state(epochs_list, reward_per_state_history, POSSIBLE_STATES, NEW_STATE_EPOCH, NEW_STATE_NAME, 
                         controller_type, REWARDS_ADAPTIVE_PARAM=None, REWARDS_INTEGRAL_PARAM=None, 
                         REWARDS_DERIVATIVE_PARAM=None, pid_params_history=None):
    """Plot rewards per state over time."""
    plt.figure(figsize=(14, 8))
    
    # Create a colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(POSSIBLE_STATES)))
    color_dict = {state: colors[i] for i, state in enumerate(POSSIBLE_STATES) if state != 'NO_STATE'}
    
    # Plot rewards for each state
    for state in POSSIBLE_STATES:
        if state != 'NO_STATE':  # Skip NO_STATE as its reward is always 0
            state_rewards_series = [reward_dict.get(state, 0) for reward_dict in reward_per_state_history]
            plt.plot(epochs_list, state_rewards_series, label=f"Rewards - {state}", 
                    color=color_dict.get(state, 'gray'))
    
    # Add vertical line for when new state was added
    plt.axvline(x=NEW_STATE_EPOCH, color='r', linestyle='--', 
              label=f"New state added")
    
    plt.xlabel('Epochs')
    plt.ylabel('Reward Value')
    
    # Adjust title based on controller type
    if controller_type == "pid":
        plt.title(f'Rewards Per State with Ziegler-Nichols-tuned PID')
    elif controller_type == "rlnopid":
        plt.title(f'Rewards Per State with Pure RL rewards')
    else:
        plt.title(f'Rewards Per State with RL-tuned PID')

    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_pid_parameters(epochs_list, pid_params_history, NEW_STATE_EPOCH, NEW_STATE_NAME):
    """Plot PID parameter evolution for RL-tuned PID controller."""
    plt.figure(figsize=(12, 7))
    plt.plot(epochs_list, [p[0] for p in pid_params_history], 'r-', label='P Parameter')
    plt.plot(epochs_list, [p[1] for p in pid_params_history], 'g-', label='I Parameter')
    plt.plot(epochs_list, [p[2] for p in pid_params_history], 'b-', label='D Parameter')
    
    # Add vertical line for when new state was added
    plt.axvline(x=NEW_STATE_EPOCH, color='k', linestyle='--', 
              label=f"New state added")
    
    plt.xlabel('Epochs')
    plt.ylabel('Parameter Value')
    plt.title('PID Parameter Evolution with RL Tuning')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

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
    state_data_to_plot = []
    
    for state in possible_states:
        if state != 'NO_STATE' and np.any(state_counts[state] > 0):
            states_to_plot.append(state)
            # Ensure all arrays have the same length
            state_data = state_counts[state][:available_epochs]
            # Pad with zeros if necessary (shouldn't be needed but safety check)
            if len(state_data) < available_epochs:
                padded_data = np.zeros(available_epochs)
                padded_data[:len(state_data)] = state_data
                state_data = padded_data
            state_data_to_plot.append(state_data)
    
    # If no states have any agents, show an empty plot with a message
    if not states_to_plot:
        plt.title("No Agents Found in Any State")
        plt.xlabel('Epoch')
        plt.ylabel('Number of Agents')
        return plt
    
    # Create a consistent color map for states
    cmap = plt.cm.get_cmap('tab10', len(states_to_plot) + 1)  # +1 to avoid repeating first color
    colors = [cmap(i) for i in range(len(states_to_plot))]
    
    # Verify all arrays have the same length before plotting
    for i, data in enumerate(state_data_to_plot):
        if len(data) != available_epochs:
            print(f"Warning: State {states_to_plot[i]} has data length {len(data)}, expected {available_epochs}")
            # Fix the length
            if len(data) < available_epochs:
                padded_data = np.zeros(available_epochs)
                padded_data[:len(data)] = data
                state_data_to_plot[i] = padded_data
            else:
                state_data_to_plot[i] = data[:available_epochs]
    
    # Create the stacked area plot with verified data
    try:
        plt.stackplot(range(available_epochs),
                     *state_data_to_plot,  # Unpack the list of arrays
                     labels=states_to_plot,
                     colors=colors,
                     alpha=0.7)
    except ValueError as e:
        print(f"Error creating stackplot: {e}")
        print(f"Available epochs: {available_epochs}")
        print(f"States to plot: {len(states_to_plot)}")
        print(f"Data shapes: {[len(data) for data in state_data_to_plot]}")
        
        # Fallback: create a simple line plot instead
        for i, (state, data) in enumerate(zip(states_to_plot, state_data_to_plot)):
            plt.plot(range(available_epochs), data, label=state, color=colors[i])
        plt.fill_between(range(available_epochs), 0, sum(state_data_to_plot), alpha=0.3)
    
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
    
def plot_agent_distribution(agents_declared_history, epochs_list, POSSIBLE_STATES, NEW_STATE_EPOCH, 
                          NEW_STATE_NAME, controller_type, k=0):
    """Plot stacked area chart showing agent distribution for a specific attribute."""
    plt.figure(figsize=(12, 7))
    agents_history = [agents_declared_history[k][epoch] for epoch in epochs_list]
    
    # Use our updated plotting function
    plot_stacked_area(agents_history, len(epochs_list), POSSIBLE_STATES)
    
    # Add vertical line for when new state was added
    plt.axvline(x=NEW_STATE_EPOCH, color='r', linestyle='--', 
              label=f"New state added")
    
    if controller_type == "pid":
        plt.title(f'Agent Distribution Over Time with Ziegler-Nichols-tuned PID')
    elif controller_type == "rlnopid":
        plt.title(f'Agent Distribution Over Time with Pure RL rewards')
    else:
        plt.title(f'Agent Distribution Over Time with RL-tuned PID')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_multiple_experiment_metrics(x_axis, values, metric_name, y_label=None, title=None, ylim=None):
    """Plot metrics across multiple experiments."""
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, values, 'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Experiment Number')
    plt.ylabel(y_label if y_label else metric_name)
    plt.title(title if title else f'{metric_name} Across Experiments')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(x_axis)
    
    if ylim:
        plt.ylim(ylim)
    
    # Add values as annotations
    for i in range(len(x_axis)):
        plt.annotate(f'{values[i]:.2f}' if isinstance(values[i], float) else f'{values[i]}', 
                    (x_axis[i], values[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.tight_layout()
    plt.show()

def plot_recovery_times(x_axis, recovery_times):
    """Plot recovery times across experiments."""
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, recovery_times, 'o-', 
            linewidth=2, markersize=8, color='green')
    plt.xlabel('Experiment Number')
    plt.ylabel('Recovery Time (epochs)')
    plt.title('System Recovery Time Across Experiments')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(x_axis)
    
    # Add values as annotations
    for i in range(len(x_axis)):
        plt.annotate(f'{recovery_times[i]}', 
                    (x_axis[i], recovery_times[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.tight_layout()
    plt.show()

def plot_convergence_times(x_axis, phase1_convergence_times, phase2_convergence_times):
    """Plot convergence times for each phase across experiments."""
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, phase1_convergence_times, 'o-', label='Phase 1 Convergence Time', 
            linewidth=2, markersize=8, color='blue')
    plt.plot(x_axis, phase2_convergence_times, 'o-', label='Phase 2 Convergence Time', 
            linewidth=2, markersize=8, color='green')
    plt.xlabel('Experiment Number')
    plt.ylabel('Convergence Time (epochs)')
    plt.title('System Convergence Times Across Experiments')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(x_axis)
    
    # Add values as annotations
    for i in range(len(x_axis)):
        plt.annotate(f'{phase1_convergence_times[i]}', 
                    (x_axis[i], phase1_convergence_times[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', color='blue')
        plt.annotate(f'{phase2_convergence_times[i]}', 
                    (x_axis[i], phase2_convergence_times[i]),
                    textcoords="offset points", xytext=(0, -15), ha='center', color='green')
    
    plt.tight_layout()
    plt.show()

def plot_pid_parameters_across_experiments(x_axis, all_pid_params):
    """Plot PID parameter values across multiple experiments."""
    plt.figure(figsize=(14, 7))
    p_values = [params[0] for params in all_pid_params]
    i_values = [params[1] for params in all_pid_params]
    d_values = [params[2] for params in all_pid_params]
    
    plt.subplot(1, 3, 1)
    plt.plot(x_axis, p_values, 'o-', color='red', linewidth=2, markersize=8)
    plt.title('P Parameter')
    plt.xlabel('Experiment Number')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.7)
    plt.xticks(x_axis)
    
    plt.subplot(1, 3, 2)
    plt.plot(x_axis, i_values, 'o-', color='green', linewidth=2, markersize=8)
    plt.title('I Parameter')
    plt.xlabel('Experiment Number')
    plt.grid(True, alpha=0.7)
    plt.xticks(x_axis)
    
    plt.subplot(1, 3, 3)
    plt.plot(x_axis, d_values, 'o-', color='blue', linewidth=2, markersize=8)
    plt.title('D Parameter')
    plt.xlabel('Experiment Number')
    plt.grid(True, alpha=0.7)
    plt.xticks(x_axis)
    
    plt.suptitle('PID Parameters Across Experiments', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_largest_state_share(x_axis, largest_state_all_experiments, NUM_ATTRIBUTES):
    """Plot largest state share metric across experiments."""
    plt.figure(figsize=(10, 6))
    for k in range(NUM_ATTRIBUTES):
        plt.plot(x_axis, largest_state_all_experiments[k], 'o-', 
                linewidth=2, label=f"Attribute")
    plt.xlabel('Experiment Number')
    plt.ylabel('Largest State Share')
    plt.title('Largest State Metric Across Experiments')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(0,1)
    plt.xticks(x_axis)
    plt.tight_layout()
    plt.show()

def plot_adaptation_quality(x_axis, adaptation_metrics, controller_type):
    """Plot adaptation quality across experiments."""
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, adaptation_metrics, 'o-', 
            linewidth=2, markersize=8, color='blue')
    plt.xlabel('Experiment Number')
    plt.ylabel('Adaptation Quality (% of ideal)')
    plt.title(f'System Adaptation Quality Across Experiments with {controller_type.upper()} Controller')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    plt.xticks(x_axis)
    
    # Add values as annotations
    for i in range(len(x_axis)):
        plt.annotate(f'{adaptation_metrics[i]:.2f}', 
                    (x_axis[i], adaptation_metrics[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.tight_layout()
    plt.show()

def plot_average_diversity(x_axis, average_diversity_by_experiment, NUM_ATTRIBUTES, controller_type, ideal_div=None):
    """Plot average diversity across experiments."""
    plt.figure(figsize=(10, 6))
    for k in range(NUM_ATTRIBUTES):
        x_axis = np.arange(0, len(average_diversity_by_experiment[k])) * 20
        plt.plot(x_axis, average_diversity_by_experiment[k], 'x-', label="Average Diversity")
    plt.xlabel('Cost factor between cheapest and most expensive state ')
    plt.ylabel('Diversity')
    plt.title(f'Diversity Across Experiments with {controller_type.upper()}')
    plt.xticks(x_axis)
    if ideal_div is not None:
        plt.axhline(y=ideal_div, color='g', linestyle=':', label=f"Ideal diversity")
        plt.ylim(0, ideal_div+0.2)
    plt.tight_layout()
    plt.legend()
    plt.show()

# New functions added from multiple_experiments.py

def print_experiment_header(experiment_num, total_experiments):
    """Print a header for each experiment."""
    print(f"\n\n{'='*50}")
    print(f"STARTING EXPERIMENT {experiment_num}/{total_experiments}")
    print(f"{'='*50}\n")

def print_pid_parameters(p, i, d):
    """Print PID parameters."""
    print("\n=== Using PID parameters: ===")
    print(f"P: {p}")
    print(f"I: {i}")
    print(f"D: {d}")

def print_rl_parameters(epsilon, epsilon_decay, learning_rate, action_scale,
                      initial_p, initial_i, initial_d, p_scale_factor, i_scale_factor, d_scale_factor):
    """Print RL controller parameters."""
    print("\n=== Initializing RL-tuned PID controller ===")
    print(f"Epsilon: {epsilon}")
    print(f"Epsilon decay: {epsilon_decay}")
    print(f"Learning rate: {learning_rate}")
    print(f"Action scale: {action_scale}")
    print(f"Initial PID parameters: P={initial_p}, I={initial_i}, D={initial_d}")
    print(f"PID scale factors: P={p_scale_factor}, I={i_scale_factor}, D={d_scale_factor}")
    print(f"PID parameters will be adaptively adjusted based on system behavior")

def print_new_state_added(new_state_name, epoch, num_states, switch_cost):
    """Print information when a new state is added."""
    print(f"\nAdding new state {new_state_name} at epoch {epoch}")
    print(f"New state added")
    print(f"Total states now: {num_states}")
    print(f"Switch cost for {new_state_name}: {switch_cost}")

def print_epoch_status(epoch, state_counts, state_rewards_last_epoch, total_rewards, 
                     controller_type, reward_controller=None, diversity=None):
    """Print status at key epochs."""
    print(f"Epoch {epoch} - Agents per state: {state_counts}")
    print(f"Epoch {epoch} - Rewards per node per state: {state_rewards_last_epoch}")
    print(f"Epoch {epoch} - Total rewards allocated: {total_rewards}")
    
    # Print controller-specific info
    if controller_type == "pid" and reward_controller:
        print(f"accumulated error : {reward_controller.accumulated_error}")
        print(f"last error : {reward_controller.last_error}")
        print(f"PID params: P={reward_controller.p_param:.1f}, I={reward_controller.i_param:.1f}, D={reward_controller.d_param:.1f}")

    elif controller_type == "rl" and reward_controller:
        print(f"RL exploration rate (epsilon): {reward_controller.epsilon:.4f}")
        print(f"RL steps done: {reward_controller.steps_done}")
        print(f"Current PID parameters: P={reward_controller.p_param:.1f}, "
              f"I={reward_controller.i_param:.1f}, D={reward_controller.d_param:.1f}")
     # Print current diversity if provided
    if diversity is not None:
        print(f"Current diversity: {diversity}")

def print_new_state_status(epoch, new_state_epoch, new_state_name, state_counts, state_rewards_last_epoch):
    """Print new state status for specific epochs after adding a new state."""
    if epoch >= new_state_epoch and epoch <= new_state_epoch + 10:
        print(f"Agents in new state {new_state_name}: {state_counts.get(new_state_name, 0)}")
        print(f"Current reward for {new_state_name}: {state_rewards_last_epoch[0].get(new_state_name, 0)}")

def print_single_experiment_results(POSSIBLE_STATES, state_rewards_last_epoch,
                                  controller_type, total_rewards_history, pid_params_history=None,
                                  reward_controller=None):
    """Print final results for a single experiment."""
    print("\n=== Final Results ===")
    print("Final states in system:", POSSIBLE_STATES)
    print("Rewards per state at the end:", state_rewards_last_epoch)
    print(f"Final ideal diversity ({len(POSSIBLE_STATES)-1} states):", get_ideal_diversity(POSSIBLE_STATES))
    print(f"Final total rewards allocated: {total_rewards_history[-1]}")
    
    # Print controller-specific information
    if controller_type == "pid" and reward_controller:
        print(f"PID parameters used: P={reward_controller.p_param:.2f}, I={reward_controller.i_param:.2f}, D={reward_controller.d_param:.2f}")
    elif controller_type == "rl" and pid_params_history:
        # For RL, print the final PID parameters learned
        final_p, final_i, final_d = pid_params_history[-1]
        print(f"Final RL-tuned PID parameters: P={final_p:.2f}, I={final_i:.2f}, D={final_d:.2f}")
        print(f"RL controller final epsilon: {reward_controller.epsilon:.4f}")
        print(f"RL steps completed: {reward_controller.steps_done}")

def print_adaptability_metrics(adapt_metrics):
    """Print detailed adaptability metrics."""
    print("\n" + "="*50)
    print("           DETAILED ADAPTABILITY METRICS           ")
    print("="*50)

    if "error" in adapt_metrics:
        print(f"\n⚠️ ANALYSIS ERROR: {adapt_metrics['error']}")
    else:
        print(f"\n📊 PRE-CHANGE STABILITY")
        if "pre_change_diversity_avg" in adapt_metrics:
            print(f"   • Average diversity before change: {adapt_metrics['pre_change_diversity_avg']:.4f} bits")
        else:
            print(f"   • Pre-change diversity metrics not available")

        print(f"\n📊 ADAPTATION SHOCK")
        if "initial_impact_percentage" in adapt_metrics and adapt_metrics['initial_impact_percentage'] is not None:
            print(f"   • Initial diversity drop:         {adapt_metrics['initial_impact_percentage']:.1f}%")
        elif "initial_drop_pct" in adapt_metrics and adapt_metrics['initial_drop_pct'] is not None:
            print(f"   • Initial diversity drop:         {adapt_metrics['initial_drop_pct']*100:.1f}%")
        else:
            print(f"   • Initial impact metrics not available")
        
        print(f"\n📊 RECOVERY SPEED")
        if "recovery_time_epochs" in adapt_metrics and adapt_metrics['recovery_time_epochs'] is not None:
            print(f"   • Time to return to pre-change level: {adapt_metrics['recovery_time_epochs']} epochs")
        else:
            print(f"   • System did not return to pre-change level or metrics not available")
            
        if "time_to_90pct_new_ideal" in adapt_metrics and adapt_metrics['time_to_90pct_new_ideal'] is not None:
            print(f"   • Time to reach 90% of new ideal:    {adapt_metrics['time_to_90pct_new_ideal']} epochs")
        elif "reconvergence_time_epochs" in adapt_metrics and adapt_metrics['reconvergence_time_epochs'] is not None:
            print(f"   • Time to reach 90% of new ideal:    {adapt_metrics['reconvergence_time_epochs']} epochs")
        else:
            print(f"   • System did not reach 90% of new ideal or metrics not available")
        
        print(f"\n📊 FINAL PERFORMANCE")
        if "phase2_settling_time" in adapt_metrics and adapt_metrics['phase2_settling_time'] is not None:
            print(f"   • System settled after:              {adapt_metrics['phase2_settling_time']} epochs")
        elif "stability_time_epochs" in adapt_metrics and adapt_metrics['stability_time_epochs'] is not None:
            print(f"   • System settled after:              {adapt_metrics['stability_time_epochs']} epochs")
        else:
            print(f"   • System did not fully settle or metrics not available")
            
        if "final_adaptation_quality" in adapt_metrics and adapt_metrics['final_adaptation_quality'] is not None:
            print(f"   • Final adaptation quality:          {adapt_metrics['final_adaptation_quality']*100:.1f}% of ideal")
        else:
            print(f"   • Final adaptation quality metrics not available")

def print_convergence_analysis(diversity_values, NEW_STATE_EPOCH, INITIAL_STATES, POSSIBLE_STATES, convergence_results):
    """Print detailed convergence analysis."""
    ideal_before = get_ideal_diversity(INITIAL_STATES)
    ideal_after = get_ideal_diversity(POSSIBLE_STATES)

    # Print convergence analysis with better formatting
    print("\n" + "="*50)
    print("           SYSTEM CONVERGENCE ANALYSIS           ")
    print("="*50)

    print(f"\n💡 DIVERSITY TARGETS:")
    print(f"   • Initial phase ({len(INITIAL_STATES)-1} states): {ideal_before:.4f} bits")
    print(f"   • Final phase ({len(POSSIBLE_STATES)-1} states):   {ideal_after:.4f} bits")
    print(f"   • State transition occurred at epoch {NEW_STATE_EPOCH}")

    # Format Phase 1 results
    print("\n📈 PHASE 1 CONVERGENCE (EPOCHS 0-{})".format(NEW_STATE_EPOCH-1))
    
    # Use the new key name 'time_to_convergence' (or fallback to 'epochs_to_converge' for backward compatibility)
    p1_conv = None
    if 'time_to_convergence' in convergence_results['phase1']:
        p1_conv = convergence_results['phase1']['time_to_convergence']
    elif 'epochs_to_converge' in convergence_results['phase1']:
        p1_conv = convergence_results['phase1']['epochs_to_converge']
    
    if p1_conv is not None:
        print(f"   • Time to reach 90% of ideal:   {p1_conv} epochs")
    else:
        print("   • System did not reach 90% of ideal diversity")
    
    # Use the new key name 'final_diversity_quality' (or calculate from 'percentage_reached' for backward compatibility)
    if 'final_diversity_quality' in convergence_results['phase1']:
        p1_pct = convergence_results['phase1']['final_diversity_quality'] * 100
    elif 'percentage_reached' in convergence_results['phase1']:
        p1_pct = convergence_results['phase1']['percentage_reached'] * 100
    else:
        p1_pct = 0
        
    print(f"   • Maximum diversity achieved:    {p1_pct:.1f}% of ideal")

    # Format Phase 2 results
    print("\n📉 PHASE 2 CONVERGENCE (EPOCHS {}-END)".format(NEW_STATE_EPOCH))
    
    # Use the new key name 'time_to_convergence' (or fallback to 're_convergence_time' for backward compatibility)
    p2_conv = None
    if 'time_to_convergence' in convergence_results['phase2']:
        p2_conv = convergence_results['phase2']['time_to_convergence']
    elif 're_convergence_time' in convergence_results['phase2']:
        p2_conv = convergence_results['phase2']['re_convergence_time']
    
    if p2_conv is not None:
        print(f"   • Time to reach 90% of new ideal: {p2_conv} epochs after transition")
    else:
        print("   • System did not reach 90% of new ideal diversity")
    
    # Use the new key name 'final_diversity_quality' (or calculate from 'percentage_reached' for backward compatibility)
    if 'final_diversity_quality' in convergence_results['phase2']:
        p2_pct = convergence_results['phase2']['final_diversity_quality'] * 100
    elif 'percentage_reached' in convergence_results['phase2']:
        p2_pct = convergence_results['phase2']['percentage_reached'] * 100
    else:
        p2_pct = 0
        
    print(f"   • Maximum diversity achieved:    {p2_pct:.1f}% of ideal")

    # Format adaptation metrics
    print("\n🔄 ADAPTATION METRICS")
    
    # Check if 'overall' section exists (old format) or get from resilience metrics (new format)
    if 'overall' in convergence_results and 'adaptation_shock' in convergence_results['overall']:
        shock = convergence_results['overall']['adaptation_shock']
        if shock is not None:
            print(f"   • Diversity drop at transition:  {shock*100:.1f}% of pre-change value")
        else:
            print("   • Could not calculate diversity drop")
    elif 'resilience' in convergence_results and 'initial_impact_percentage' in convergence_results['resilience']:
        shock = convergence_results['resilience']['initial_impact_percentage']
        if shock is not None:
            print(f"   • Diversity drop at transition:  {shock:.1f}% of pre-change value")
        else:
            print("   • Could not calculate diversity drop")
    else:
        print("   • Diversity drop metrics not available")
    
    # Check for recovery time in various possible locations
    recovery = None
    if 'overall' in convergence_results and 'recovery_time' in convergence_results['overall']:
        recovery = convergence_results['overall']['recovery_time']
    elif 'resilience' in convergence_results and 'recovery_time_epochs' in convergence_results['resilience']:
        recovery = convergence_results['resilience']['recovery_time_epochs']
    
    if recovery is not None:
        print(f"   • Recovery time:                 {recovery} epochs after transition")
    else:
        print("   • System did not recover to pre-change diversity levels")

def print_reward_allocation_metrics(total_rewards_history, NEW_STATE_EPOCH):
    """Print reward allocation metrics."""
    print("\n" + "="*50)
    print("           REWARD ALLOCATION METRICS           ")
    print("="*50)
    print(f"   • Initial total rewards: {total_rewards_history[0]:.2f}")
    print(f"   • Final total rewards: {total_rewards_history[-1]:.2f}")
    print(f"   • Maximum total rewards: {max(total_rewards_history):.2f} at epoch {total_rewards_history.index(max(total_rewards_history))}")
    print(f"   • Minimum total rewards: {min(total_rewards_history):.2f} at epoch {total_rewards_history.index(min(total_rewards_history))}")
    print(f"   • Average total rewards: {sum(total_rewards_history)/len(total_rewards_history):.2f}")
    
    # Calculate the change in reward efficiency
    pre_change_rewards = total_rewards_history[:NEW_STATE_EPOCH]
    post_change_rewards = total_rewards_history[NEW_STATE_EPOCH:]
    avg_pre_change = sum(pre_change_rewards)/len(pre_change_rewards) if pre_change_rewards else 0
    avg_post_change = sum(post_change_rewards)/len(post_change_rewards) if post_change_rewards else 0
    
    print(f"   • Average pre-change rewards: {avg_pre_change:.2f}")
    print(f"   • Average post-change rewards: {avg_post_change:.2f}")
    print(f"   • Change in reward allocation: {((avg_post_change-avg_pre_change)/avg_pre_change)*100:.1f}%")
    print("\n" + "="*50)

def print_multi_experiment_header(num_experiments, controller_type):
    """Print header for multiple experiments summary."""
    print("\n\n" + "="*60)
    print(f"{'':^10}SUMMARY RESULTS FOR {num_experiments} EXPERIMENTS{'':^10}")

def print_single_experiment_results(states, state_rewards_last_epoch,
                                  controller_type, total_rewards_history, pid_params_history,
                                  reward_controller):
    """Print final results and analysis for a single experiment"""
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    
    num_active_states = len([state for state in states if state != 'NO_STATE'])
    total_rewards_allocated = sum([state_rewards_last_epoch[0][state] for state in states])
    
    print(f"\nFinal state rewards (last epoch):")
    for state in states:
        if state != 'NO_STATE':
            reward = state_rewards_last_epoch[0][state]
            print(f"  • {state}: {reward:.2f}")
    
    print(f"\nEfficiency metrics:")
    print(f"  • Total rewards allocated: {total_rewards_allocated:.2f}")
    print(f"  • Number of active states: {num_active_states}")
    
    avg_total_rewards = sum(total_rewards_history[-100:]) / min(100, len(total_rewards_history))
    print(f"  • Average total rewards (last 100 epochs): {avg_total_rewards:.2f}")
    
    if controller_type == "rl" and pid_params_history:
        # Calculate final parameter composition
        final_p, final_i, final_d = pid_params_history[-1]
        total = final_p + final_i + final_d
        
        if total > 0:
            print(f"\nFinal PID parameters:")
            print(f"  • P: {final_p:.2f} ({100*final_p/total:.1f}%)")
            print(f"  • I: {final_i:.2f} ({100*final_i/total:.1f}%)")
            print(f"  • D: {final_d:.2f} ({100*final_d/total:.1f}%)")
            
            # Add reward scale info
            print(f"  • Reward Scale: {reward_controller.reward_scale:.2f}")
        else:
            print("\nFinal PID parameters are all zero.")
        
        # Print best parameters found
        best_p, best_i, best_d = reward_controller.best_pid_params
        best_scale = reward_controller.best_reward_scale
        best_total = best_p + best_i + best_d
        
        if best_total > 0:
            print(f"\nBest PID parameters found (diversity ratio: {reward_controller.best_diversity_ratio:.3f}):")
            print(f"  • P: {best_p:.2f} ({100*best_p/best_total:.1f}%)")
            print(f"  • I: {best_i:.2f} ({100*best_i/best_total:.1f}%)")
            print(f"  • D: {best_d:.2f} ({100*best_d/best_total:.1f}%)")
            print(f"  • Reward Scale: {best_scale:.2f}")
    
    print("\n" + "="*60)

def print_reward_allocation_metrics(total_rewards_history, new_state_epoch):
    """Print metrics related to reward allocation efficiency"""
    if not total_rewards_history:
        return
        
    # Calculate average rewards before and after new state
    pre_new_state = total_rewards_history[:new_state_epoch]
    post_new_state = total_rewards_history[new_state_epoch:]
    
    if pre_new_state:
        avg_pre = sum(pre_new_state) / len(pre_new_state)
        print(f"\nReward Allocation Metrics:")
        print(f"  • Average total rewards before new state: {avg_pre:.2f}")
        
        if post_new_state:
            avg_post = sum(post_new_state) / len(post_new_state)
            print(f"  • Average total rewards after new state: {avg_post:.2f}")
            print(f"  • Change: {((avg_post-avg_pre)/avg_pre)*100:.1f}%")
            
            # Calculate stability metrics
            std_pre = np.std(pre_new_state)
            std_post = np.std(post_new_state)
            print(f"  • Reward stability (std dev) before: {std_pre:.2f}")
            print(f"  • Reward stability (std dev) after: {std_post:.2f}")

"""
Comparative plotting functions for PID vs RL comparison.
These functions will be added to plotting_functions.py
"""

def plot_side_by_side_diversity(pid_diversity, rl_diversity, epochs_list, NUM_ATTRIBUTES, 
                              NEW_STATE_EPOCH, NEW_STATE_NAME, INITIAL_STATES, POSSIBLE_STATES,
                              pid_params=None, final_rl_params=None):
    """
    Plot diversity comparison between PID and RL controllers on the same graph.
    """

    plt.figure(figsize=(14, 8))
    
    # Define colors for PID and RL
    pid_color = 'blue'
    rl_color = 'orange'  # Changed from 'red' to 'orange'
    
    # Plot diversity for each attribute for both controllers
    for k in range(NUM_ATTRIBUTES):
        # Extract PID diversity values 
        pid_values = [pid_diversity[k][epoch][0] for epoch in epochs_list if epoch in pid_diversity[k]]
        
        # Extract RL diversity values
        rl_values = [rl_diversity[k][epoch][0] for epoch in epochs_list if epoch in rl_diversity[k]]
        
        # Ensure we only plot available epochs
        pid_epochs = epochs_list[:len(pid_values)]
        rl_epochs = epochs_list[:len(rl_values)]
        
        # Plot PID with solid line
        plt.plot(pid_epochs, pid_values, color=pid_color, linestyle='-', 
                 label=f"PID - Attribute")
        
        # Plot RL with dashed line
        plt.plot(rl_epochs, rl_values, color=rl_color, linestyle='--', 
                 label=f"RL - Attribute")
    
    # Add vertical line for new state addition
    plt.axvline(x=NEW_STATE_EPOCH, color='r', linestyle=':', 
                label=f"New state added")
    
    # Add horizontal lines for ideal diversity
    ideal_before = get_ideal_diversity(INITIAL_STATES)
    ideal_after = get_ideal_diversity(POSSIBLE_STATES)
    plt.axhline(y=ideal_before, color='g', linestyle='-.', 
                label=f"Ideal diversity - {len(INITIAL_STATES)-1} states")
    plt.axhline(y=ideal_after, color='g', linestyle='-', 
                label=f"Ideal diversity - {len(POSSIBLE_STATES)-1} states")
    
    # Title with controller parameters if provided
    title = 'Diversity Comparison: PID vs RL-tuned PID'
    if pid_params and final_rl_params:
        p, i, d = pid_params
        rl_p, rl_i, rl_d = final_rl_params
        title += f'\nPID (P={p:.1f}, I={i:.1f}, D={d:.1f}) vs RL-final (P={rl_p:.1f}, I={rl_i:.1f}, D={rl_d:.1f})'
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Diversity (Shannon Entropy)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    return plt

def plot_rewards_comparison(pid_rewards, rl_rewards, epochs_list, NEW_STATE_EPOCH, NEW_STATE_NAME):
    """
    Plot comparison of reward allocation between PID and RL controllers.
    """
    
    plt.figure(figsize=(14, 8))
    
    # Define colors for PID and RL
    pid_color = 'blue'
    rl_color = 'orange'  # Changed from 'red' to 'orange'
    
    # Ensure we only plot available data
    pid_epochs = epochs_list[:len(pid_rewards)]
    rl_epochs = epochs_list[:len(rl_rewards)]
    
    # Plot PID rewards
    plt.plot(pid_epochs, pid_rewards, color=pid_color, linestyle='-', label='PID Controller')
    
    # Plot RL rewards
    plt.plot(rl_epochs, rl_rewards, color=rl_color, linestyle='-', label='RL Controller')
    
    # Add vertical line for new state addition
    plt.axvline(x=NEW_STATE_EPOCH, color='k', linestyle=':', 
                label=f"New state added")
    
    plt.title('Total Rewards Comparison: PID vs RL-tuned PID')
    plt.xlabel('Epochs')
    plt.ylabel('Total Rewards Allocated')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    return plt


"""
Comparative plotting functions continued
"""

def plot_rl_pid_parameter_evolution(pid_params_history, epochs_list, NEW_STATE_EPOCH, NEW_STATE_NAME,
                                   initial_pid_params=None):
    """
    Plot the evolution of PID parameters in the RL-tuned controller over time.
    """
    # Extract parameter series
    p_values = [params[0] for params in pid_params_history]
    i_values = [params[1] for params in pid_params_history]
    d_values = [params[2] for params in pid_params_history]
    
    # Plot parameters evolution
    plt.figure(figsize=(14, 8))
    
    # Epochs available
    available_epochs = epochs_list[:len(pid_params_history)]
    
    # Plot P parameter
    plt.plot(available_epochs, p_values, 'r-', label='P Parameter')
    
    # Plot I parameter
    plt.plot(available_epochs, i_values, 'g-', label='I Parameter')
    
    # Plot D parameter
    plt.plot(available_epochs, d_values, 'b-', label='D Parameter')
    
    # Add horizontal lines for reference PID values if provided
    if initial_pid_params:
        p, i, d = initial_pid_params
        plt.axhline(y=p, color='r', linestyle=':', label=f'Fixed PID P={p:.1f}')
        plt.axhline(y=i, color='g', linestyle=':', label=f'Fixed PID I={i:.1f}')
        plt.axhline(y=d, color='b', linestyle=':', label=f'Fixed PID D={d:.1f}')
    
    # Add vertical line for new state addition
    plt.axvline(x=NEW_STATE_EPOCH, color='k', linestyle='--', 
                label=f"New state added")
    
    plt.title('PID Parameter Evolution in RL-tuned Controller')
    plt.xlabel('Epochs')
    plt.ylabel('Parameter Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add annotations for final values
    final_p = p_values[-1] if p_values else 0
    final_i = i_values[-1] if i_values else 0
    final_d = d_values[-1] if d_values else 0
    
    plt.annotate(f'Final P={final_p:.1f}', xy=(available_epochs[-1], final_p),
                xytext=(10, 10), textcoords='offset points')
    plt.annotate(f'Final I={final_i:.1f}', xy=(available_epochs[-1], final_i),
                xytext=(10, -20), textcoords='offset points')
    plt.annotate(f'Final D={final_d:.1f}', xy=(available_epochs[-1], final_d),
                xytext=(10, -50), textcoords='offset points')
    
    return plt

def plot_agent_distribution_comparison(pid_agents_history, rl_agents_history, epochs_list, POSSIBLE_STATES, 
                                     NEW_STATE_EPOCH, NEW_STATE_NAME, attribute_idx=0):
    """
    Plot and compare agent distributions between PID and RL controllers.
    
    Parameters:
    -----------
    pid_agents_history : list
        Agent state history for the PID controller
    rl_agents_history : list
        Agent state history for the RL controller
    epochs_list : list
        List of epoch numbers for x-axis
    POSSIBLE_STATES : list
        List of all possible states
    NEW_STATE_EPOCH : int
        Epoch when new state was added
    NEW_STATE_NAME : str
        Name of the new state
    attribute_idx : int, optional (default=0)
        Index of the attribute to plot
    """

    # Create a 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Process PID agent history
    valid_states = [s for s in POSSIBLE_STATES if s != 'NO_STATE']
    pid_state_counts = {state: [] for state in valid_states}
    
    # Count agents in each state for each epoch
    for epoch in range(min(len(pid_agents_history[attribute_idx]), len(epochs_list))):
        epoch_states = pid_agents_history[attribute_idx][epoch]
        current_counts = {state: 0 for state in valid_states}
        
        for agent_state in epoch_states:
            if agent_state in current_counts:
                current_counts[agent_state] += 1
        
        for state in valid_states:
            pid_state_counts[state].append(current_counts[state])
    
    # Process RL agent history
    rl_state_counts = {state: [] for state in valid_states}
    
    for epoch in range(min(len(rl_agents_history[attribute_idx]), len(epochs_list))):
        epoch_states = rl_agents_history[attribute_idx][epoch]
        current_counts = {state: 0 for state in valid_states}
        
        for agent_state in epoch_states:
            if agent_state in current_counts:
                current_counts[agent_state] += 1
        
        for state in valid_states:
            rl_state_counts[state].append(current_counts[state])
    
    # Plot PID agent distribution
    pid_epochs = epochs_list[:len(pid_state_counts[valid_states[0]])]
    ax1.stackplot(pid_epochs, 
                [pid_state_counts[state] for state in valid_states],
                labels=valid_states, alpha=0.7)
    
    # Plot RL agent distribution
    rl_epochs = epochs_list[:len(rl_state_counts[valid_states[0]])]
    ax2.stackplot(rl_epochs, 
                [rl_state_counts[state] for state in valid_states],
                labels=valid_states, alpha=0.7)
    
    # Add vertical lines for new state addition
    ax1.axvline(x=NEW_STATE_EPOCH, color='r', linestyle='--', 
              label=f"New state added")
    ax2.axvline(x=NEW_STATE_EPOCH, color='r', linestyle='--', 
              label=f"New state added")
    
    # Set titles and labels
    ax1.set_title('Agent Distribution with PID Controller')
    ax2.set_title('Agent Distribution with RL-tuned PID Controller')
    
    ax1.set_xlabel('Epochs')
    ax2.set_xlabel('Epochs')
    
    ax1.set_ylabel('Number of Agents')
    
    # Add legend to the second subplot only to avoid duplication
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc='upper right', title='States')
    
    # Add new state annotation
    ax1.annotate(f'New state: {NEW_STATE_NAME}', xy=(NEW_STATE_EPOCH, 0), 
               xytext=(10, 30), textcoords='offset points',
               arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    return plt

def plot_performance_metrics_comparison(pid_metrics, rl_metrics):
    """
    Create a bar chart comparing key performance metrics between PID and RL controllers.
    """
    
    # Define colors for PID and RL
    pid_color = 'blue'
    rl_color = 'orange'  # Changed from 'red' to 'orange'
    
    # Extract key metrics for comparison
    metrics_to_compare = [
        ('Time to Convergence (Phase 1)', 
         pid_metrics['phase1_convergence']['time_to_convergence'], 
         rl_metrics['phase1_convergence']['time_to_convergence']),
        
        ('Time to Re-convergence (Phase 2)', 
         pid_metrics['phase2_convergence']['time_to_convergence'], 
         rl_metrics['phase2_convergence']['time_to_convergence']),
        
        ('Recovery Time after State Addition', 
         pid_metrics['resilience']['recovery_time_epochs'], 
         rl_metrics['resilience']['recovery_time_epochs']),
        
        ('Final Diversity Quality (% of ideal)', 
         pid_metrics['phase2_convergence']['final_diversity_quality'] * 100, 
         rl_metrics['phase2_convergence']['final_diversity_quality'] * 100),
        
        ('Avg. Reward per Epoch (Phase 1)', 
         pid_metrics['phase1_efficiency']['avg_absolute_reward'],
         rl_metrics['phase1_efficiency']['avg_absolute_reward']),
        
        ('Avg. Reward per Epoch (Phase 2)', 
         pid_metrics['phase2_efficiency']['avg_absolute_reward'],
         rl_metrics['phase2_efficiency']['avg_absolute_reward'])
    ]
    
    # Filter out metrics with None values
    valid_metrics = []
    for name, pid_val, rl_val in metrics_to_compare:
        if pid_val is not None and rl_val is not None:
            valid_metrics.append((name, pid_val, rl_val))
    
    if not valid_metrics:
        print("No valid metrics available for comparison")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up bar positions
    bar_width = 0.35
    index = np.arange(len(valid_metrics))
    
    # Create bars with updated colors
    pid_bars = ax.bar(index - bar_width/2, [m[1] for m in valid_metrics], bar_width,
                     label='PID Controller', color=pid_color, alpha=0.7)
    rl_bars = ax.bar(index + bar_width/2, [m[2] for m in valid_metrics], bar_width,
                    label='RL-tuned PID Controller', color=rl_color, alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Performance Metric')
    ax.set_ylabel('Value')
    ax.set_title('PID vs RL-tuned PID Performance Comparison')
    ax.set_xticks(index)
    ax.set_xticklabels([m[0] for m in valid_metrics], rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')
    
    add_labels(pid_bars)
    add_labels(rl_bars)
    
    plt.tight_layout()
    return plt

def create_metrics_summary_table(pid_metrics, rl_metrics):
    """
    Create a text-based metrics summary table comparing PID and RL performance.
    
    Parameters:
    -----------
    pid_metrics : dict
        Performance metrics for the PID controller
    rl_metrics : dict
        Performance metrics for the RL controller
        
    Returns:
    --------
    str
        Formatted table as a string
    """
    # Define the metrics to display
    metric_definitions = [
        ("🔍 Convergence Metrics", "", ""),
        ("  Time to initial convergence (Phase 1)", 
         pid_metrics['phase1_convergence']['time_to_convergence'], 
         rl_metrics['phase1_convergence']['time_to_convergence']),
        ("  Convergence rate (Phase 1)", 
         pid_metrics['phase1_convergence']['convergence_rate'],
         rl_metrics['phase1_convergence']['convergence_rate']),
        ("  Steady-state error (Phase 1)", 
         pid_metrics['phase1_convergence']['steady_state_error'],
         rl_metrics['phase1_convergence']['steady_state_error']),
        ("  Time to re-convergence (Phase 2)", 
         pid_metrics['phase2_convergence']['time_to_convergence'],
         rl_metrics['phase2_convergence']['time_to_convergence']),
        ("  Final diversity quality (% of ideal)", 
         pid_metrics['phase2_convergence']['final_diversity_quality'] * 100 if pid_metrics['phase2_convergence']['final_diversity_quality'] is not None else None,
         rl_metrics['phase2_convergence']['final_diversity_quality'] * 100 if rl_metrics['phase2_convergence']['final_diversity_quality'] is not None else None),
         
        ("🔄 Resilience Metrics", "", ""),
        ("  Initial impact magnitude", 
         pid_metrics['resilience']['initial_impact_magnitude'],
         rl_metrics['resilience']['initial_impact_magnitude']),
        ("  Impact percentage", 
         pid_metrics['resilience']['initial_impact_percentage'],
         rl_metrics['resilience']['initial_impact_percentage']),
        ("  Recovery time (epochs)", 
         pid_metrics['resilience']['recovery_time_epochs'],
         rl_metrics['resilience']['recovery_time_epochs']),
        ("  Stability time after transition", 
         pid_metrics['resilience']['stability_time_epochs'],
         rl_metrics['resilience']['stability_time_epochs']),
        ("  Final adaptation quality", 
         pid_metrics['resilience']['final_adaptation_quality'] * 100 if pid_metrics['resilience']['final_adaptation_quality'] is not None else None,
         rl_metrics['resilience']['final_adaptation_quality'] * 100 if rl_metrics['resilience']['final_adaptation_quality'] is not None else None),
         
        ("💰 Resource Efficiency", "", ""),
        ("  Avg reward per epoch (overall)", 
         pid_metrics['overall_efficiency']['avg_reward_per_epoch'],
         rl_metrics['overall_efficiency']['avg_reward_per_epoch']),
        ("  Total cumulative reward", 
         pid_metrics['overall_efficiency']['total_cumulative_reward'],
         rl_metrics['overall_efficiency']['total_cumulative_reward']),
        ("  Diversity per reward unit", 
         pid_metrics['overall_efficiency']['diversity_per_reward_unit'],
         rl_metrics['overall_efficiency']['diversity_per_reward_unit']),
        ("  Reward volatility", 
         pid_metrics['overall_efficiency']['reward_volatility'],
         rl_metrics['overall_efficiency']['reward_volatility']),
        ("  Efficiency trend", 
         pid_metrics['overall_efficiency']['efficiency_trend'],
         rl_metrics['overall_efficiency']['efficiency_trend'])
    ]
    
    # Build the table header
    header = f"{'Performance Metric':<40} {'PID Controller':<20} {'RL-tuned PID':<20} {'RL Improvement':<15}"
    divider = "-" * 95
    
    # Build the table rows
    rows = [header, divider]
    
    for metric, pid_val, rl_val in metric_definitions:
        # For section headers
        if pid_val == "" and rl_val == "":
            rows.append(f"\n{metric}")
            continue
            
        # Skip metrics with None values
        if pid_val is None or rl_val is None:
            pid_str = "N/A" if pid_val is None else f"{pid_val:.2f}"
            rl_str = "N/A" if rl_val is None else f"{rl_val:.2f}"
            diff_str = "N/A"
            rows.append(f"{metric:<40} {pid_str:<20} {rl_str:<20} {diff_str:<15}")
            continue
        
        # Calculate improvement percentage
        if pid_val == 0:
            improvement = "∞" if rl_val > 0 else "0%"
        else:
            improvement = f"{((rl_val - pid_val) / abs(pid_val)) * 100:.1f}%"
            
            # Determine if improvement is positive or negative based on the metric
            # For metrics where lower is better (times, errors, etc.)
            if "time" in metric.lower() or "error" in metric.lower() or "volatility" in metric.lower():
                # Negative change is better
                if rl_val < pid_val:
                    improvement = f"🟢 {improvement}"  # Green indicator
                else:
                    improvement = f"🔴 {improvement}"  # Red indicator
            else:
                # Positive change is better for diversity, quality, etc.
                if rl_val > pid_val:
                    improvement = f"🟢 {improvement}"
                else:
                    improvement = f"🔴 {improvement}"
        
        # Format the values
        pid_str = f"{pid_val:.2f}"
        rl_str = f"{rl_val:.2f}"
        
        rows.append(f"{metric:<40} {pid_str:<20} {rl_str:<20} {improvement:<15}")
    
    # Join the rows and return
    return "\n".join(rows)

def plot_metric_over_time_comparison(metric_name, pid_values, rl_values, epochs_list, NEW_STATE_EPOCH, NEW_STATE_NAME):
    """
    Plot a metric over time for both PID and RL controllers.
    
    Parameters:
    -----------
    metric_name : str
        Name of the metric being plotted
    pid_values : list
        Values of the metric for the PID controller
    rl_values : list
        Values of the metric for the RL controller
    epochs_list : list
        List of epoch numbers for x-axis
    NEW_STATE_EPOCH : int
        Epoch when new state was added
    NEW_STATE_NAME : str
        Name of the new state
    """

    plt.figure(figsize=(14, 8))
    
    # Ensure we only plot available data
    pid_epochs = epochs_list[:len(pid_values)]
    rl_epochs = epochs_list[:len(rl_values)]
    
    # Plot PID metric
    plt.plot(pid_epochs, pid_values, 'b-', label='PID Controller')
    
    # Plot RL metric
    plt.plot(rl_epochs, rl_values, 'r-', label='RL Controller')
    
    # Add vertical line for new state addition
    plt.axvline(x=NEW_STATE_EPOCH, color='k', linestyle=':', 
                label=f"New state added")
    
    plt.title(f'{metric_name} Comparison: PID vs RL-tuned PID')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    return plt

def compare_and_visualize_all_controllers(comparison_results, args):
    """
    Compare and visualize the results from all controllers (PID, RL, RL-no-PID).
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary containing results from all controllers
    args : argparse.Namespace
        Command line arguments
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from measurement_functions import get_ideal_diversity
    
    print(f"Comparison results keys: {list(comparison_results.keys())}")  # Debug print
    print(f"Number of controllers being compared: {len(comparison_results)}")  # Debug print
    
    # Extract common parameters from the first available result
    first_result = list(comparison_results.values())[0]
    epochs = first_result['epochs']
    epochs_list = list(range(epochs))
    NEW_STATE_EPOCH = first_result['NEW_STATE_EPOCH']
    NEW_STATE_NAME = first_result['NEW_STATE_NAME']
    INITIAL_STATES = first_result['INITIAL_STATES']
    POSSIBLE_STATES = first_result['POSSIBLE_STATES']
    
    print("\n\n" + "="*80)
    print(f"{'':^10}COMPARISON RESULTS: ALL CONTROLLERS{'':^10}")
    print("="*80)
    
    # Plot comparisons
    plot_diversity_comparison_all_controllers(comparison_results, epochs_list, NEW_STATE_EPOCH, 
                                            NEW_STATE_NAME, INITIAL_STATES, POSSIBLE_STATES)
    
    plot_rewards_comparison_all_controllers(comparison_results, epochs_list, NEW_STATE_EPOCH, 
                                          NEW_STATE_NAME)
    
    plot_pid_parameters_comparison(comparison_results, epochs_list, NEW_STATE_EPOCH, 
                                 NEW_STATE_NAME)
    
    # Print summary statistics
    print_comparison_summary_statistics(comparison_results, epochs, INITIAL_STATES, POSSIBLE_STATES)

def plot_diversity_comparison_all_controllers(comparison_results, epochs_list, NEW_STATE_EPOCH, 
                                            NEW_STATE_NAME, INITIAL_STATES, POSSIBLE_STATES):
    """
    Plot diversity over time comparison for all controllers.
    """
    import matplotlib.pyplot as plt
    from measurement_functions import get_ideal_diversity
    
    # Define colors and labels for each controller
    colors = {
        'pid': 'blue',
        'rl': 'orange', 
        'rlnopid': 'green'
    }
    
    labels = {
        'pid': 'Ziegler-Nichols-tuned PID',
        'rl': 'RL-tuned PID',
        'rlnopid': 'Pure RL'
    }
    
    plt.figure(figsize=(14, 8))
    
    print(f"Plotting diversity for controllers: {list(comparison_results.keys())}")  # Debug print
    
    for controller_type, results in comparison_results.items():
        print(f"Processing controller: {controller_type}")  # Debug print
        # Extract diversity values for the first attribute
        diversity_values = [results['final_diversity'][0][epoch][0] for epoch in epochs_list 
                           if epoch in results['final_diversity'][0]]
        
        print(f"Controller {controller_type} has {len(diversity_values)} diversity values")  # Debug print
        
        # Plot with appropriate styling
        plt.plot(epochs_list[:len(diversity_values)], diversity_values, 
                color=colors[controller_type], linestyle='-', linewidth=2,
                label=labels[controller_type])
    
    # Add vertical line for new state addition
    plt.axvline(x=NEW_STATE_EPOCH, color='red', linestyle=':', linewidth=2,
                label=f"New state added")
    
    # Add horizontal lines for ideal diversity
    ideal_before = get_ideal_diversity(INITIAL_STATES)
    ideal_after = get_ideal_diversity(POSSIBLE_STATES)
    plt.axhline(y=ideal_before, color='gray', linestyle='-.', alpha=0.7,
                label=f"Ideal diversity - {len(INITIAL_STATES)-1} states")
    plt.axhline(y=ideal_after, color='gray', linestyle='-', alpha=0.7,
                label=f"Ideal diversity - {len(POSSIBLE_STATES)-1} states")
    
    plt.title('Diversity Comparison: All Controllers', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Diversity (Shannon Entropy)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_rewards_comparison_all_controllers(comparison_results, epochs_list, NEW_STATE_EPOCH, 
                                          NEW_STATE_NAME):
    """
    Plot total rewards over time comparison for all controllers.
    """
    import matplotlib.pyplot as plt
    
    # Define colors and labels for each controller
    colors = {
        'pid': 'blue',
        'rl': 'orange',
        'rlnopid': 'green'
    }
    
    labels = {
        'pid': 'Ziegler-Nichols-tuned PID',
        'rl': 'RL-tuned PID',
        'rlnopid': 'Pure RL'
    }
    
    plt.figure(figsize=(14, 8))
    
    for controller_type, results in comparison_results.items():
        total_rewards = results['total_rewards_history']
        
        plt.plot(epochs_list[:len(total_rewards)], total_rewards, 
                color=colors[controller_type], linestyle='-', linewidth=2,
                label=labels[controller_type])
    
    # Add vertical line for new state addition
    plt.axvline(x=NEW_STATE_EPOCH, color='red', linestyle=':', linewidth=2,
                label=f"New state added")
    
    plt.title('Total Rewards Comparison: All Controllers', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Total Rewards Allocated', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_pid_parameters_comparison(comparison_results, epochs_list, NEW_STATE_EPOCH, 
                                 NEW_STATE_NAME):
    """
    Show PID parameter evolution for PID and RL controllers only (ignore RL-no-PID).
    """
    import matplotlib.pyplot as plt
    
    # Only include controllers that use PID parameters
    controllers_with_pid = ['pid', 'rl']
    available_pid_controllers = [c for c in controllers_with_pid if c in comparison_results]
    
    if len(available_pid_controllers) == 0:
        print("No controllers with PID parameters found for comparison.")
        return
    
    # Define colors for PID controllers
    colors = {
        'pid': 'blue',
        'rl': 'orange'
    }
    
    labels = {
        'pid': 'Ziegler-Nichols-tuned PID',
        'rl': 'RL-tuned PID'
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot P, I, D parameters separately
    param_names = ['P Parameter', 'I Parameter', 'D Parameter']
    param_indices = [0, 1, 2]
    
    for i, (param_name, param_idx) in enumerate(zip(param_names, param_indices)):
        ax = axes[i]
        
        for controller_type in available_pid_controllers:
            results = comparison_results[controller_type]
            pid_params_history = results['pid_params_history']
            
            # Extract parameter values
            param_values = [params[param_idx] for params in pid_params_history]
            
            # Plot parameter evolution
            ax.plot(epochs_list[:len(param_values)], param_values, 
                   color=colors[controller_type], linestyle='-', linewidth=2,
                   label=labels[controller_type])
            
            # Add vertical line for new state addition
            ax.axvline(x=NEW_STATE_EPOCH, color='red', linestyle=':', linewidth=1,
                      alpha=0.7)
        
        ax.set_title(f'{param_name} Evolution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Parameter Value', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add a main title and new state annotation
    fig.suptitle('PID Parameter Evolution: PID vs RL-tuned PID', fontsize=16, fontweight='bold')
    
    # Add annotation for new state addition
    axes[0].annotate(f'New state: {NEW_STATE_NAME}', 
                    xy=(NEW_STATE_EPOCH, axes[0].get_ylim()[1] * 0.9), 
                    xytext=(NEW_STATE_EPOCH + len(epochs_list) * 0.05, axes[0].get_ylim()[1] * 0.9),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, color='red')
    
    plt.tight_layout()
    plt.show()

def print_comparison_summary_statistics(comparison_results, epochs, INITIAL_STATES, POSSIBLE_STATES):
    """
    Print summary statistics for all controllers and relative performance comparison.
    """
    import numpy as np
    from measurement_functions import get_ideal_diversity
    
    ideal_after = get_ideal_diversity(POSSIBLE_STATES)
    
    # Define labels
    labels = {
        'pid': 'Ziegler-Nichols-tuned PID',
        'rl': 'RL-tuned PID',
        'rlnopid': 'Pure RL'
    }
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Collect metrics for all controllers
    controller_metrics = {}
    
    for controller_type, results in comparison_results.items():
        print(f"\n{labels[controller_type].upper()}:")
        print("-" * 40)
        
        # Calculate final diversity metrics
        diversity_values = [results['final_diversity'][0][epoch][0] for epoch in range(epochs)]
        final_diversity = diversity_values[-1]
        avg_diversity_last_25pct = np.mean(diversity_values[-int(epochs*0.25):])
        
        # Calculate reward metrics
        total_rewards = results['total_rewards_history']
        final_total_rewards = total_rewards[-1]
        avg_rewards_last_25pct = np.mean(total_rewards[-int(epochs*0.25):])
        
        # Store metrics for comparison
        controller_metrics[controller_type] = {
            'final_diversity': final_diversity,
            'avg_diversity': avg_diversity_last_25pct,
            'final_rewards': final_total_rewards,
            'avg_rewards': avg_rewards_last_25pct,
            'diversity_ratio': final_diversity / ideal_after if ideal_after > 0 else 0
        }
        
        # Print metrics
        print(f"  • Final diversity: {final_diversity:.4f} bits")
        print(f"  • Avg diversity (last 25%): {avg_diversity_last_25pct:.4f} bits")
        print(f"  • Final total rewards: {final_total_rewards:.2f}")
        print(f"  • Avg total rewards (last 25%): {avg_rewards_last_25pct:.2f}")
        print(f"  • Diversity achievement: {controller_metrics[controller_type]['diversity_ratio']*100:.1f}% of ideal")
        
        # Print controller-specific info
        if controller_type == 'pid':
            pid_params = results['pid_params']
            print(f"  • PID parameters: P={pid_params[0]:.1f}, I={pid_params[1]:.1f}, D={pid_params[2]:.1f}")
        elif controller_type == 'rl':
            final_pid_params = results['final_pid_params']
            print(f"  • Final RL-tuned PID: P={final_pid_params[0]:.1f}, I={final_pid_params[1]:.1f}, D={final_pid_params[2]:.1f}")
        elif controller_type == 'rlnopid':
            print(f"  • Uses direct reward control (no PID parameters)")
    
    # Print relative performance comparison
    if 'pid' in controller_metrics:
        print("\n" + "="*80)
        print("RELATIVE PERFORMANCE COMPARISON")
        print("="*80)
        
        pid_metrics = controller_metrics['pid']
        
        print(f"\nUsing PID Controller as baseline:")
        print("-" * 40)
        
        for controller_type, metrics in controller_metrics.items():
            if controller_type == 'pid':
                continue
                
            # Calculate relative performance
            diversity_improvement = ((metrics['final_diversity'] - pid_metrics['final_diversity']) 
                                   / pid_metrics['final_diversity']) * 100
            reward_change = ((metrics['avg_rewards'] - pid_metrics['avg_rewards']) 
                           / pid_metrics['avg_rewards']) * 100
            
            print(f"\n{labels[controller_type]}:")
            print(f"  • Diversity improvement: {diversity_improvement:+.1f}%")
            print(f"  • Reward allocation change: {reward_change:+.1f}%")
    
    # Print convergence and adaptation metrics
    print_convergence_adaptation_metrics(comparison_results, epochs, INITIAL_STATES, POSSIBLE_STATES, labels)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

def print_convergence_adaptation_metrics(comparison_results, epochs, INITIAL_STATES, POSSIBLE_STATES, labels):
    """
    Print convergence and adaptation metrics for all controllers.
    """
    import numpy as np
    from measurement_functions import get_ideal_diversity
    
    ideal_before = get_ideal_diversity(INITIAL_STATES)
    ideal_after = get_ideal_diversity(POSSIBLE_STATES)
    
    # Extract NEW_STATE_EPOCH from results
    first_result = list(comparison_results.values())[0]
    NEW_STATE_EPOCH = first_result['NEW_STATE_EPOCH']
    
    print("\n" + "="*80)
    print("CONVERGENCE AND ADAPTATION METRICS")
    print("="*80)
    
    for controller_type, results in comparison_results.items():
        print(f"\n{labels[controller_type]}:")
        print("-" * 40)
        
        diversity_values = [results['final_diversity'][0][epoch][0] for epoch in range(epochs)]
        
        # Phase 1 convergence (before new state)
        phase1_values = diversity_values[:NEW_STATE_EPOCH]
        if len(phase1_values) > 0:
            phase1_target = ideal_before * 0.9  # 90% of ideal
            phase1_convergence = None
            for i, val in enumerate(phase1_values):
                if val >= phase1_target:
                    phase1_convergence = i
                    break
            
            if phase1_convergence is not None:
                print(f"  • Phase 1 convergence time: {phase1_convergence} epochs")
            else:
                print(f"  • Phase 1: Did not reach 90% of ideal diversity")
        
        # Phase 2 adaptation (after new state)
        phase2_values = diversity_values[NEW_STATE_EPOCH:]
        if len(phase2_values) > 0:
            phase2_target = ideal_after * 0.9  # 90% of new ideal
            phase2_convergence = None
            for i, val in enumerate(phase2_values):
                if val >= phase2_target:
                    phase2_convergence = i
                    break
            
            if phase2_convergence is not None:
                print(f"  • Phase 2 convergence time: {phase2_convergence} epochs after state addition")
            else:
                print(f"  • Phase 2: Did not reach 90% of new ideal diversity")
            
            # Calculate adaptation shock (immediate impact of new state)
            if len(phase1_values) > 10 and len(phase2_values) > 10:
                pre_change_avg = np.mean(phase1_values[-10:])  # Last 10 epochs before change
                immediate_impact = phase2_values[0]  # First epoch after change
                adaptation_shock = ((pre_change_avg - immediate_impact) / pre_change_avg) * 100
                print(f"  • Adaptation shock: {adaptation_shock:.1f}% diversity drop")