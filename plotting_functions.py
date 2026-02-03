# Enhanced plotting_functions.py - Only visualization/plotting functions

import numpy as np
import matplotlib.pyplot as plt
from measurement_functions import get_ideal_diversity

# Import printing functions from the separate module
from printing_functions import *


plt.rc('font', size=28)
plt.rc('axes', titlesize=28, labelsize=28)
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('legend', fontsize=28)
plt.rc('figure', figsize=(14, 7))
plt.rc('lines', linewidth=2.5)
FIGURE_COUNTER = 0 #start number of saved figures

def save_and_show(filename=None):
    """Save figure as PDF then display it."""
    global FIGURE_COUNTER
    if filename is None:
        filename = f"figure_{FIGURE_COUNTER}"
        FIGURE_COUNTER += 1
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')
    plt.show()

    
def plot_diversity_over_time(final_diversity, epochs_list, NUM_ATTRIBUTES, NEW_version_EPOCH, 
                           NEW_version_NAME, INITIAL_versionS, POSSIBLE_versionS, controller_type,
                           REWARDS_ADAPTIVE_PARAM=None, REWARDS_INTEGRAL_PARAM=None, REWARDS_DERIVATIVE_PARAM=None,
                           pid_params_history=None, add_new_version=True):
    """Plot diversity over time for all attributes."""
    plt.figure()
    for k in range(NUM_ATTRIBUTES):
        # Extract diversity values from dictionary
        diversity_values = [final_diversity[k][epoch][0] for epoch in epochs_list]  # [0] to get the first (and only) value
        plt.plot(epochs_list, diversity_values, label=f"Diversity - Attribute")
    
    # Add vertical line to mark when new version was added (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None:
        plt.axvline(x=NEW_version_EPOCH, color='r', linestyle='--', 
                    label=f"New version added")
    
    # Add horizontal line for ideal diversity
    if add_new_version and NEW_version_EPOCH is not None:
        # Show both before and after ideal diversity
        ideal_before = get_ideal_diversity(INITIAL_versionS)
        ideal_after = get_ideal_diversity(POSSIBLE_versionS)
        plt.axhline(y=ideal_before, color='g', linestyle=':', 
                    label=f"Ideal diversity - {len(INITIAL_versionS)-1} versions")
        plt.axhline(y=ideal_after, color='g', linestyle='-', 
                    label=f"Ideal diversity - {len(POSSIBLE_versionS)-1} versions")
    else:
        # Show only the target ideal diversity
        ideal_diversity = get_ideal_diversity(INITIAL_versionS)
        plt.axhline(y=ideal_diversity, color='g', linestyle='-', 
                    label=f"Ideal diversity - {len(INITIAL_versionS)-1} versions")
    
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
    save_and_show()

def plot_total_rewards(epochs_list, total_rewards_history, NEW_version_EPOCH, NEW_version_NAME, controller_type,
                     REWARDS_ADAPTIVE_PARAM=None, REWARDS_INTEGRAL_PARAM=None, REWARDS_DERIVATIVE_PARAM=None,
                     pid_params_history=None, add_new_version=True):
    """Plot total rewards over time."""
    plt.figure()
    plt.plot(epochs_list, total_rewards_history, 'b-', label='Total Rewards Allocated')
    
    # Add vertical line for when new version was added (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None:
        plt.axvline(x=NEW_version_EPOCH, color='r', linestyle='--', 
                  label=f"New version added")
    
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
    save_and_show()

def plot_rewards_per_version(epochs_list, reward_per_version_history, POSSIBLE_versionS, NEW_version_EPOCH, NEW_version_NAME, 
                         controller_type, REWARDS_ADAPTIVE_PARAM=None, REWARDS_INTEGRAL_PARAM=None, 
                         REWARDS_DERIVATIVE_PARAM=None, pid_params_history=None, add_new_version=True):
    """Plot rewards per version over time."""
    plt.figure()
    
    # Create a colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(POSSIBLE_versionS)))
    color_dict = {version: colors[i] for i, version in enumerate(POSSIBLE_versionS) if version != 'NO_version'}
    
    # Plot rewards for each version
    for version in POSSIBLE_versionS:
        if version != 'NO_version':  # Skip NO_version as its reward is always 0
            version_rewards_series = [reward_dict.get(version, 0) for reward_dict in reward_per_version_history]
            plt.plot(epochs_list, version_rewards_series, label=f"Rewards - {version}", 
                    color=color_dict.get(version, 'gray'))
    
    # Add vertical line for when new version was added (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None:
        plt.axvline(x=NEW_version_EPOCH, color='r', linestyle='--', 
                  label=f"New version added")
    
    plt.xlabel('Epochs')
    plt.ylabel('Reward Value')
    
    # Adjust title based on controller type
    if controller_type == "pid":
        plt.title(f'Rewards Per version with Ziegler-Nichols-tuned PID')
    elif controller_type == "rlnopid":
        plt.title(f'Rewards Per version with Pure RL rewards')
    else:
        plt.title(f'Rewards Per version with RL-tuned PID')

    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    save_and_show()

def plot_pid_parameters(epochs_list, pid_params_history, NEW_version_EPOCH, NEW_version_NAME, add_new_version=True):
    """Plot PID parameter evolution for RL-tuned PID controller."""
    plt.figure( )
    plt.plot(epochs_list, [p[0] for p in pid_params_history], 'r-', label='P Parameter')
    plt.plot(epochs_list, [p[1] for p in pid_params_history], 'g-', label='I Parameter')
    plt.plot(epochs_list, [p[2] for p in pid_params_history], 'b-', label='D Parameter')
    
    # Add vertical line for when new version was added (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None:
        plt.axvline(x=NEW_version_EPOCH, color='k', linestyle='--', 
                  label=f"New version added")
    
    plt.xlabel('Epochs')
    plt.ylabel('Parameter Value')
    plt.title('PID Parameter Evolution with RL Tuning')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    save_and_show()

def plot_stacked_area(agents_history, epochs, possible_versions):
    """
    Plot a stacked area chart showing the distribution of agents across versions over time.
    Modified to handle dynamic version sets properly and avoid empty plots.
    
    Args:
        agents_history: List of agent versions for each epoch
        epochs: Number of epochs to plot
        possible_versions: List of all possible versions (including those added during the simulation)
    """
    
    # Use only the available epochs
    available_epochs = min(len(agents_history), epochs)
    
    if available_epochs == 0:
        print("Warning: No agent history data available to plot")
        plt.figure()
        plt.title("No Agent Data Available")
        return plt
    
    # Initialize version counts dictionary with all possible versions
    version_counts = {version: np.zeros(available_epochs) for version in possible_versions}
    
    # Count agents in each version for each epoch
    for epoch in range(available_epochs):
        # Skip if there's no data for this epoch
        if epoch >= len(agents_history) or not agents_history[epoch]:
            continue
            
        # Count agents in each version for this epoch
        for agent_version in agents_history[epoch]:
            if agent_version in version_counts:
                version_counts[agent_version][epoch] += 1
    
    # Filter out NO_version and find versions that actually have agents
    versions_to_plot = []
    version_data_to_plot = []
    
    for version in possible_versions:
        if version != 'NO_version' and np.any(version_counts[version] > 0):
            versions_to_plot.append(version)
            # Ensure all arrays have the same length
            version_data = version_counts[version][:available_epochs]
            # Pad with zeros if necessary (shouldn't be needed but safety check)
            if len(version_data) < available_epochs:
                padded_data = np.zeros(available_epochs)
                padded_data[:len(version_data)] = version_data
                version_data = padded_data
            version_data_to_plot.append(version_data)
    
    # If no versions have any agents, show an empty plot with a message
    if not versions_to_plot:
        plt.title("No Agents Found in Any version")
        plt.xlabel('Epoch')
        plt.ylabel('Number of Agents')
        return plt
    
    # Create a consistent color map for versions
    cmap = plt.cm.get_cmap('tab10', len(versions_to_plot) + 1)  # +1 to avoid repeating first color
    colors = [cmap(i) for i in range(len(versions_to_plot))]
    
    # Verify all arrays have the same length before plotting
    for i, data in enumerate(version_data_to_plot):
        if len(data) != available_epochs:
            print(f"Warning: version {versions_to_plot[i]} has data length {len(data)}, expected {available_epochs}")
            # Fix the length
            if len(data) < available_epochs:
                padded_data = np.zeros(available_epochs)
                padded_data[:len(data)] = data
                version_data_to_plot[i] = padded_data
            else:
                version_data_to_plot[i] = data[:available_epochs]
    
    # Create the stacked area plot with verified data
    try:
        plt.stackplot(range(available_epochs),
                     *version_data_to_plot,  # Unpack the list of arrays
                     labels=versions_to_plot,
                     colors=colors,
                     alpha=0.7)
    except ValueError as e:
        print(f"Error creating stackplot: {e}")
        print(f"Available epochs: {available_epochs}")
        print(f"versions to plot: {len(versions_to_plot)}")
        print(f"Data shapes: {[len(data) for data in version_data_to_plot]}")
        
        # Fallback: create a simple line plot instead
        for i, (version, data) in enumerate(zip(versions_to_plot, version_data_to_plot)):
            plt.plot(range(available_epochs), data, label=version, color=colors[i])
        plt.fill_between(range(available_epochs), 0, sum(version_data_to_plot), alpha=0.3)
    
    plt.xlabel('Epoch')
    plt.ylabel('Number of Agents')
    plt.title('Evolution of Agent versions Over Epochs')
    
    # Only add legend if we have versions to plot
    if versions_to_plot:
        plt.legend(loc='upper right')
        
    plt.grid(True, linestyle='--', alpha=0.5)

    # Set x-ticks to show reasonable intervals
    step_size = max(1, available_epochs // 10)
    plt.xticks(range(0, available_epochs + 1, step_size))
    
    return plt
    
def plot_agent_distribution(agents_declared_history, epochs_list, POSSIBLE_versionS, NEW_version_EPOCH, 
                          NEW_version_NAME, controller_type, k=0, add_new_version=True):
    """Plot stacked area chart showing agent distribution for a specific attribute."""
    plt.figure()
    agents_history = [agents_declared_history[k][epoch] for epoch in epochs_list]
    
    # Use our updated plotting function
    plot_stacked_area(agents_history, len(epochs_list), POSSIBLE_versionS)
    
    # Add vertical line for when new version was added (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None:
        plt.axvline(x=NEW_version_EPOCH, color='r', linestyle='--', 
                  label=f"New version added")
    
    if controller_type == "pid":
        plt.title(f'Agent Distribution Over Time with Ziegler-Nichols-tuned PID')
    elif controller_type == "rlnopid":
        plt.title(f'Agent Distribution Over Time with Pure RL rewards')
    else:
        plt.title(f'Agent Distribution Over Time with RL-tuned PID')

    plt.legend(loc='upper left')
    plt.tight_layout()
    save_and_show()

def plot_multiple_experiment_metrics(x_axis, values, metric_name, y_label=None, title=None, ylim=None):
    """Plot metrics across multiple experiments."""
    plt.figure()
    plt.plot(x_axis, values, 'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Switch Frequency Parameter')
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
    save_and_show()

def plot_recovery_times(x_axis, recovery_times):
    """Plot recovery times across experiments."""
    plt.figure()
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
    save_and_show()

def plot_convergence_times(x_axis, phase1_convergence_times, phase2_convergence_times):
    """Plot convergence times for each phase across experiments."""
    plt.figure()
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
    save_and_show()

def plot_single_phase_convergence_times(x_axis, convergence_times):
    """Plot convergence times for single phase experiments."""
    plt.figure()
    plt.plot(x_axis, convergence_times, 'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Experiment Number')
    plt.ylabel('Convergence Time (epochs)')
    plt.title('System Convergence Times Across Experiments')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(x_axis)
    
    # Add values as annotations
    for i in range(len(x_axis)):
        plt.annotate(f'{convergence_times[i]}', 
                    (x_axis[i], convergence_times[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.tight_layout()
    save_and_show()

def plot_pid_parameters_across_experiments(x_axis, all_pid_params):
    """Plot PID parameter values across multiple experiments."""
    plt.figure()
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
    
    plt.suptitle('PID Parameters Across Experiments')
    plt.tight_layout()
    save_and_show()

def plot_largest_version_share(x_axis, largest_version_all_experiments, NUM_ATTRIBUTES):
    """Plot largest version share metric across experiments."""
    plt.figure()
    for k in range(NUM_ATTRIBUTES):
        plt.plot(x_axis, largest_version_all_experiments[k], 'o-', 
                linewidth=2, label=f"Attribute")
    plt.xlabel('Experiment Number')
    plt.ylabel('Largest version Share')
    plt.title('Largest version Metric Across Experiments')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(0,1)
    plt.xticks(x_axis)
    plt.tight_layout()
    save_and_show()

def plot_adaptation_quality(x_axis, adaptation_metrics, controller_type):
    """Plot adaptation quality across experiments."""
    plt.figure()
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
    save_and_show()

def plot_average_diversity(x_axis, average_diversity_by_experiment, NUM_ATTRIBUTES, controller_type, ideal_div=None):
    """Plot average diversity across experiments."""
    plt.figure()
    for k in range(NUM_ATTRIBUTES):
        x_axis = np.arange(0, len(average_diversity_by_experiment[k])) * 10
        plt.plot(x_axis, average_diversity_by_experiment[k], 'x-', label="Average Diversity")
    plt.xlabel('Cost factor between cheapest and most expensive version ')
    plt.ylabel('Diversity')
    plt.title(f'Diversity Across Experiments with {controller_type.upper()}')
    plt.xticks(x_axis)
    if ideal_div is not None:
        plt.axhline(y=ideal_div, color='g', linestyle=':', label=f"Ideal diversity")
        plt.ylim(0, ideal_div+0.2)
    plt.tight_layout()
    plt.legend()
    save_and_show()

# Comparative plotting functions for PID vs RL comparison

def plot_side_by_side_diversity(pid_diversity, rl_diversity, epochs_list, NUM_ATTRIBUTES, 
                              NEW_version_EPOCH, NEW_version_NAME, INITIAL_versionS, POSSIBLE_versionS,
                              pid_params=None, final_rl_params=None, add_new_version=True):
    """
    Plot diversity comparison between PID and RL controllers on the same graph.
    """

    plt.figure()
    
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
    
    # Add vertical line for new version addition (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None:
        plt.axvline(x=NEW_version_EPOCH, color='r', linestyle=':', 
                    label=f"New version added")
    
    # Add horizontal lines for ideal diversity
    if add_new_version and NEW_version_EPOCH is not None:
        ideal_before = get_ideal_diversity(INITIAL_versionS)
        ideal_after = get_ideal_diversity(POSSIBLE_versionS)
        plt.axhline(y=ideal_before, color='g', linestyle='-.', 
                    label=f"Ideal diversity - {len(INITIAL_versionS)-1} versions")
        plt.axhline(y=ideal_after, color='g', linestyle='-', 
                    label=f"Ideal diversity - {len(POSSIBLE_versionS)-1} versions")
    else:
        ideal_diversity = get_ideal_diversity(INITIAL_versionS)
        plt.axhline(y=ideal_diversity, color='g', linestyle='-', 
                    label=f"Ideal diversity - {len(INITIAL_versionS)-1} versions")
    
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

def plot_rewards_comparison(pid_rewards, rl_rewards, epochs_list, NEW_version_EPOCH, NEW_version_NAME, add_new_version=True):
    """
    Plot comparison of reward allocation between PID and RL controllers.
    """
    
    plt.figure()
    
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
    
    # Add vertical line for new version addition (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None:
        plt.axvline(x=NEW_version_EPOCH, color='k', linestyle=':', 
                    label=f"New version added")
    
    plt.title('Total Rewards Comparison: PID vs RL-tuned PID')
    plt.xlabel('Epochs')
    plt.ylabel('Total Rewards Allocated')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    return plt

def plot_rl_pid_parameter_evolution(pid_params_history, epochs_list, NEW_version_EPOCH, NEW_version_NAME,
                                   initial_pid_params=None, add_new_version=True):
    """
    Plot the evolution of PID parameters in the RL-tuned controller over time.
    """
    # Extract parameter series
    p_values = [params[0] for params in pid_params_history]
    i_values = [params[1] for params in pid_params_history]
    d_values = [params[2] for params in pid_params_history]
    
    # Plot parameters evolution
    plt.figure()
    
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
    
    # Add vertical line for new version addition (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None:
        plt.axvline(x=NEW_version_EPOCH, color='k', linestyle='--', 
                    label=f"New version added")
    
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

def plot_agent_distribution_comparison(pid_agents_history, rl_agents_history, epochs_list, POSSIBLE_versionS, 
                                     NEW_version_EPOCH, NEW_version_NAME, attribute_idx=0, add_new_version=True):
    """
    Plot and compare agent distributions between PID and RL controllers.
    
    Parameters:
    -----------
    pid_agents_history : list
        Agent version history for the PID controller
    rl_agents_history : list
        Agent version history for the RL controller
    epochs_list : list
        List of epoch numbers for x-axis
    POSSIBLE_versionS : list
        List of all possible versions
    NEW_version_EPOCH : int
        Epoch when new version was added
    NEW_version_NAME : str
        Name of the new version
    attribute_idx : int, optional (default=0)
        Index of the attribute to plot
    add_new_version : bool
        Whether new version was added during experiment
    """

    # Create a 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Process PID agent history
    valid_versions = [s for s in POSSIBLE_versionS if s != 'NO_version']
    pid_version_counts = {version: [] for version in valid_versions}
    
    # Count agents in each version for each epoch
    for epoch in range(min(len(pid_agents_history[attribute_idx]), len(epochs_list))):
        epoch_versions = pid_agents_history[attribute_idx][epoch]
        current_counts = {version: 0 for version in valid_versions}
        
        for agent_version in epoch_versions:
            if agent_version in current_counts:
                current_counts[agent_version] += 1
        
        for version in valid_versions:
            pid_version_counts[version].append(current_counts[version])
    
    # Process RL agent history
    rl_version_counts = {version: [] for version in valid_versions}
    
    for epoch in range(min(len(rl_agents_history[attribute_idx]), len(epochs_list))):
        epoch_versions = rl_agents_history[attribute_idx][epoch]
        current_counts = {version: 0 for version in valid_versions}
        
        for agent_version in epoch_versions:
            if agent_version in current_counts:
                current_counts[agent_version] += 1
        
        for version in valid_versions:
            rl_version_counts[version].append(current_counts[version])
    
    # Plot PID agent distribution
    pid_epochs = epochs_list[:len(pid_version_counts[valid_versions[0]])]
    ax1.stackplot(pid_epochs, 
                [pid_version_counts[version] for version in valid_versions],
                labels=valid_versions, alpha=0.7)
    
    # Plot RL agent distribution
    rl_epochs = epochs_list[:len(rl_version_counts[valid_versions[0]])]
    ax2.stackplot(rl_epochs, 
                [rl_version_counts[version] for version in valid_versions],
                labels=valid_versions, alpha=0.7)
    
    # Add vertical lines for new version addition (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None:
        ax1.axvline(x=NEW_version_EPOCH, color='r', linestyle='--', 
                  label=f"New version added")
        ax2.axvline(x=NEW_version_EPOCH, color='r', linestyle='--', 
                  label=f"New version added")
    
    # Set titles and labels
    ax1.set_title('Agent Distribution with PID Controller')
    ax2.set_title('Agent Distribution with RL-tuned PID Controller')
    
    ax1.set_xlabel('Epochs')
    ax2.set_xlabel('Epochs')
    
    ax1.set_ylabel('Number of Agents')
    
    # Add legend to the second subplot only to avoid duplication
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc='upper right', title='versions')
    
    # Add new version annotation (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None and NEW_version_NAME is not None:
        ax1.annotate(f'New version: {NEW_version_NAME}', xy=(NEW_version_EPOCH, 0), 
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
        
        ('Recovery Time after version Addition', 
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
    fig, ax = plt.subplots()
    
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

def plot_metric_over_time_comparison(metric_name, pid_values, rl_values, epochs_list, NEW_version_EPOCH, NEW_version_NAME, add_new_version=True):
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
    NEW_version_EPOCH : int
        Epoch when new version was added
    NEW_version_NAME : str
        Name of the new version
    add_new_version : bool
        Whether new version was added during experiment
    """

    plt.figure()
    
    # Ensure we only plot available data
    pid_epochs = epochs_list[:len(pid_values)]
    rl_epochs = epochs_list[:len(rl_values)]
    
    # Plot PID metric
    plt.plot(pid_epochs, pid_values, 'b-', label='PID Controller')
    
    # Plot RL metric
    plt.plot(rl_epochs, rl_values, 'r-', label='RL Controller')
    
    # Add vertical line for new version addition (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None:
        plt.axvline(x=NEW_version_EPOCH, color='k', linestyle=':', 
                    label=f"New version added")
    
    plt.title(f'{metric_name} Comparison: PID vs RL-tuned PID')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    return plt

# Multi-controller comparison functions

def compare_and_visualize_all_controllers(comparison_results, args):
    """
    Compare and visualize results from multiple controllers across multiple experiments.
    
    Parameters:
    -----------
    comparison_results : dict
        Results from all controllers, with structure:
        {
            'controller_name': {
                'all_experiments': [list of experiment results],
                'metrics': {aggregated metrics},
                'representative_experiment': single experiment for detailed plots
            }
        }
    args : argparse.Namespace
        Command line arguments
    """
    
    print("\n\n" + "="*80)
    print(f"{'':^20}COMPARISON RESULTS VISUALIZATION{'':^20}")
    print("="*80)
    
    controllers = list(comparison_results.keys())
    num_experiments = args.num_experiments
    
    # Check if we have multiple experiments or single experiments
    if num_experiments > 1:
        # Multiple experiments mode - show experiment-wise comparisons
        print(f"Analyzing {num_experiments} experiments per controller...")
        
        # 1. Plot average diversity across experiments for each controller
        plot_diversity_comparison_across_experiments(comparison_results, controllers, num_experiments)
        
        # 2. Plot adaptation metrics comparison (only if new version was added)
        if args.add_new_version:
            plot_adaptation_comparison_across_experiments(comparison_results, controllers, num_experiments)
        
        # 3. Plot convergence times comparison
        plot_convergence_comparison_across_experiments(comparison_results, controllers, num_experiments, args.add_new_version)
        
        # 4. Plot representative diversity over time for each controller
        plot_representative_diversity_comparison(comparison_results, controllers, args.add_new_version)
        
        # 5. Print statistical summary
        from printing_functions import print_statistical_comparison_summary
        print_statistical_comparison_summary(comparison_results, controllers, num_experiments)
    
    else:
        # Single experiment mode - use existing detailed comparison
        print("Analyzing single experiment comparison...")
        
        # Extract common parameters from the first available result
        first_result = list(comparison_results.values())[0]['representative_experiment']
        epochs = first_result['epochs']
        epochs_list = list(range(epochs))
        NEW_version_EPOCH = first_result.get('NEW_version_EPOCH')
        NEW_version_NAME = first_result.get('NEW_version_NAME')
        INITIAL_versionS = first_result['INITIAL_versionS']
        POSSIBLE_versionS = first_result['POSSIBLE_versionS']
        
        # Use existing single-experiment comparison functions
        plot_diversity_comparison_all_controllers(comparison_results, epochs_list, NEW_version_EPOCH, 
                                                NEW_version_NAME, INITIAL_versionS, POSSIBLE_versionS, args.add_new_version)
        
        plot_rewards_comparison_all_controllers(comparison_results, epochs_list, NEW_version_EPOCH, 
                                              NEW_version_NAME, args.add_new_version)
        
        plot_pid_parameters_comparison(comparison_results, epochs_list, NEW_version_EPOCH, 
                                     NEW_version_NAME, args.add_new_version)
        
        # Print summary statistics
        from printing_functions import print_comparison_summary_statistics
        print_comparison_summary_statistics(comparison_results, epochs, INITIAL_versionS, POSSIBLE_versionS)

def plot_diversity_comparison_all_controllers(comparison_results, epochs_list, NEW_version_EPOCH, 
                                            NEW_version_NAME, INITIAL_versionS, POSSIBLE_versionS, add_new_version=True):
    """
    Plot diversity over time comparison for all controllers.
    """
    
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
    
    plt.figure()
    
    print(f"Plotting diversity for controllers: {list(comparison_results.keys())}")  # Debug print
    
    plotorder = len(comparison_results.items())+1
    for controller_type, results in comparison_results.items():
        print(f"Processing controller: {controller_type}")  # Debug print
        
        # Access data from representative_experiment
        rep_exp = results['representative_experiment']
        diversity_values = [rep_exp['final_diversity'][0][epoch][0] for epoch in epochs_list 
                           if epoch in rep_exp['final_diversity'][0]]
        
        print(f"Controller {controller_type} has {len(diversity_values)} diversity values")  # Debug print
        
        # Plot with appropriate styling
        plt.plot(epochs_list[:len(diversity_values)], diversity_values, 
                color=colors[controller_type], linestyle='-', linewidth=2,
                label=labels[controller_type], zorder=plotorder)
        plotorder = plotorder -1
    # Add vertical line for new version addition (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None:
        plt.axvline(x=NEW_version_EPOCH, color='red', linestyle=':', linewidth=2,
                    label=f"New version added")
    
    # Add horizontal lines for ideal diversity
    if add_new_version and NEW_version_EPOCH is not None:
        ideal_before = get_ideal_diversity(INITIAL_versionS)
        ideal_after = get_ideal_diversity(POSSIBLE_versionS)
        plt.axhline(y=ideal_before, color='gray', linestyle='-.', alpha=0.7,
                    label=f"Ideal diversity - {len(INITIAL_versionS)-1} versions")
        plt.axhline(y=ideal_after, color='gray', linestyle='-', alpha=0.7,
                    label=f"Ideal diversity - {len(POSSIBLE_versionS)-1} versions")
    else:
        ideal_diversity = get_ideal_diversity(INITIAL_versionS)
        plt.axhline(y=ideal_diversity, color='gray', linestyle='-', alpha=0.7,
                    label=f"Ideal diversity - {len(INITIAL_versionS)-1} versions")
    
    plt.title('Diversity Comparison: All Controllers', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Diversity (Shannon Entropy)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_and_show()

def plot_rewards_comparison_all_controllers(comparison_results, epochs_list, NEW_version_EPOCH, 
                                          NEW_version_NAME, add_new_version=True):
    """
    Plot total rewards over time comparison for all controllers.
    """
    
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
    
    plt.figure()
    
    for controller_type, results in comparison_results.items():
        # Access data from representative_experiment
        rep_exp = results['representative_experiment']
        total_rewards = rep_exp['total_rewards_history']
        
        plt.plot(epochs_list[:len(total_rewards)], total_rewards, 
                color=colors[controller_type], linestyle='-', linewidth=2,
                label=labels[controller_type])
    
    # Add vertical line for new version addition (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None:
        plt.axvline(x=NEW_version_EPOCH, color='red', linestyle=':', linewidth=2,
                    label=f"New version added")
    
    plt.title('Total Rewards Comparison: All Controllers', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Total Rewards Allocated')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_and_show()

def plot_pid_parameters_comparison(comparison_results, epochs_list, NEW_version_EPOCH, 
                                 NEW_version_NAME, add_new_version=True):
    """
    Show PID parameter evolution for PID and RL controllers only (ignore RL-no-PID).
    """
    
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
            # Access data from representative_experiment
            rep_exp = results['representative_experiment']
            pid_params_history = rep_exp['pid_params_history']
            
            # Extract parameter values
            param_values = [params[param_idx] for params in pid_params_history]
            
            # Plot parameter evolution
            ax.plot(epochs_list[:len(param_values)], param_values, 
                   color=colors[controller_type], linestyle='-', linewidth=2,
                   label=labels[controller_type])
            
            # Add vertical line for new version addition (only if new version was added)
            if add_new_version and NEW_version_EPOCH is not None:
                ax.axvline(x=NEW_version_EPOCH, color='red', linestyle=':', linewidth=1,
                          alpha=0.7)
        
        ax.set_title(f'{param_name} Evolution', fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Parameter Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add a main title and new version annotation
    fig.suptitle('PID Parameter Evolution: PID vs RL-tuned PID', fontweight='bold')
    
    # Add annotation for new version addition (only if new version was added)
    if add_new_version and NEW_version_EPOCH is not None and NEW_version_NAME is not None:
        axes[0].annotate(f'New version: {NEW_version_NAME}', 
                        xy=(NEW_version_EPOCH, axes[0].get_ylim()[1] * 0.9), 
                        xytext=(NEW_version_EPOCH + len(epochs_list) * 0.05, axes[0].get_ylim()[1] * 0.9),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        color='red')
    
    plt.tight_layout()
    save_and_show()

def plot_diversity_comparison_across_experiments(comparison_results, controllers, num_experiments):
    """Plot average diversity across experiments for each controller - simple line plot."""
    
    plt.figure()
    
    # Plot diversity by experiment number for each controller
    # x_axis = np.arange(0, num_experiments) * 0.4 #switch frequency experiment
    x_axis = np.arange(0, num_experiments) * 0.4 #cost experiment
    # Define colors for different controllers
    colors = {'pid': 'blue', 'rl': 'orange', 'rlnopid': 'green'}
    labels = {'pid': 'ZN-PID', 'rl': 'RL-tuned PID', 'rlnopid': 'Pure RL'}
    
    for controller in controllers:
        metrics = comparison_results[controller]['metrics']
        diversity_values = metrics['average_diversity_by_experiment']
        ideal_diversity = metrics['ideal_diversity']
        
        color = colors.get(controller, 'black')
        label = labels.get(controller, controller.upper())
        
        plt.plot(x_axis, diversity_values, 'o-', label=label, 
                linewidth=2, markersize=6, color=color)
    
    plt.xlabel('Cost Gap')
    plt.ylabel('Average Diversity')
    plt.title(f'Diversity Performance With Varying Cost Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x_axis)
    
    plt.tight_layout()
    save_and_show()

def plot_adaptation_comparison_across_experiments(comparison_results, controllers, num_experiments):
    """Plot adaptation metrics comparison across experiments."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    x_axis = list(range(1, num_experiments + 1))
    
    # Plot 1: Adaptation Quality by experiment
    for controller in controllers:
        metrics = comparison_results[controller]['metrics']
        adaptation_values = [a*100 for a in metrics['adaptation_metrics']]  # Convert to percentage
        
        ax1.plot(x_axis, adaptation_values, 'o-', label=f'{controller.upper()}', 
                linewidth=2, markersize=6)
    
    ax1.set_xlabel('Experiment Number')
    ax1.set_ylabel('Adaptation Quality (%)')
    ax1.set_title('Adaptation Quality Across Experiments')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Recovery Time by experiment
    for controller in controllers:
        metrics = comparison_results[controller]['metrics']
        recovery_values = metrics['recovery_times']
        
        ax2.plot(x_axis, recovery_values, 'o-', label=f'{controller.upper()}', 
                linewidth=2, markersize=6)
    
    ax2.set_xlabel('Experiment Number')
    ax2.set_ylabel('Recovery Time (epochs)')
    ax2.set_title('Recovery Time Across Experiments')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot for adaptation quality
    adaptation_data = []
    labels = []
    
    for controller in controllers:
        metrics = comparison_results[controller]['metrics']
        adaptation_values = [a*100 for a in metrics['adaptation_metrics']]
        adaptation_data.append(adaptation_values)
        labels.append(controller.upper())
    
    bp1 = ax3.boxplot(adaptation_data, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp1['boxes'], colors[:len(controllers)]):
        patch.set_facecolor(color)
    
    ax3.set_ylabel('Adaptation Quality (%)')
    ax3.set_title('Adaptation Quality Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Box plot for recovery time
    recovery_data = []
    
    for controller in controllers:
        metrics = comparison_results[controller]['metrics']
        recovery_values = metrics['recovery_times']
        recovery_data.append(recovery_values)
    
    bp2 = ax4.boxplot(recovery_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors[:len(controllers)]):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('Recovery Time (epochs)')
    ax4.set_title('Recovery Time Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_show()

def plot_convergence_comparison_across_experiments(comparison_results, controllers, num_experiments, add_new_version=True):
    """Plot convergence times comparison across experiments."""
    
    if add_new_version:
        # Two-phase convergence plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        x_axis = list(range(1, num_experiments + 1))
        
        # Plot 1: Phase 1 convergence by experiment
        for controller in controllers:
            metrics = comparison_results[controller]['metrics']
            phase1_values = metrics['phase1_convergence_times']
            
            ax1.plot(x_axis, phase1_values, 'o-', label=f'{controller.upper()}', 
                    linewidth=2, markersize=6)
        
        ax1.set_xlabel('Experiment Number')
        ax1.set_ylabel('Phase 1 Convergence Time (epochs)')
        ax1.set_title('Phase 1 Convergence Across Experiments')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Phase 2 convergence by experiment
        for controller in controllers:
            metrics = comparison_results[controller]['metrics']
            phase2_values = metrics['phase2_convergence_times']
            
            ax2.plot(x_axis, phase2_values, 'o-', label=f'{controller.upper()}', 
                    linewidth=2, markersize=6)
        
        ax2.set_xlabel('Experiment Number')
        ax2.set_ylabel('Phase 2 Convergence Time (epochs)')
        ax2.set_title('Phase 2 Convergence Across Experiments')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Box plot for Phase 1 convergence
        phase1_data = []
        labels = []
        
        for controller in controllers:
            metrics = comparison_results[controller]['metrics']
            phase1_values = metrics['phase1_convergence_times']
            phase1_data.append(phase1_values)
            labels.append(controller.upper())
        
        bp1 = ax3.boxplot(phase1_data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp1['boxes'], colors[:len(controllers)]):
            patch.set_facecolor(color)
        
        ax3.set_ylabel('Phase 1 Convergence Time (epochs)')
        ax3.set_title('Phase 1 Convergence Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Box plot for Phase 2 convergence
        phase2_data = []
        
        for controller in controllers:
            metrics = comparison_results[controller]['metrics']
            phase2_values = metrics['phase2_convergence_times']
            phase2_data.append(phase2_values)
        
        bp2 = ax4.boxplot(phase2_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors[:len(controllers)]):
            patch.set_facecolor(color)
        
        ax4.set_ylabel('Phase 2 Convergence Time (epochs)')
        ax4.set_title('Phase 2 Convergence Distribution')
        ax4.grid(True, alpha=0.3)
        
    else:
        # Single-phase convergence plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        x_axis = list(range(1, num_experiments + 1))
        
        # Plot 1: Convergence by experiment
        for controller in controllers:
            metrics = comparison_results[controller]['metrics']
            phase1_values = metrics['phase1_convergence_times']
            
            ax1.plot(x_axis, phase1_values, 'o-', label=f'{controller.upper()}', 
                    linewidth=2, markersize=6)
        
        ax1.set_xlabel('Experiment Number')
        ax1.set_ylabel('Convergence Time (epochs)')
        ax1.set_title('Convergence Across Experiments')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot for convergence
        phase1_data = []
        labels = []
        
        for controller in controllers:
            metrics = comparison_results[controller]['metrics']
            phase1_values = metrics['phase1_convergence_times']
            phase1_data.append(phase1_values)
            labels.append(controller.upper())
        
        bp1 = ax2.boxplot(phase1_data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp1['boxes'], colors[:len(controllers)]):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('Convergence Time (epochs)')
        ax2.set_title('Convergence Distribution')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_show()

def plot_representative_diversity_comparison(comparison_results, controllers, add_new_version=True):
    """Plot diversity over time for representative experiments from each controller."""
    
    fig, ax = plt.subplots(1, 1)
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, controller in enumerate(controllers):
        rep_exp = comparison_results[controller]['representative_experiment']
        epochs = rep_exp['epochs']
        final_diversity = rep_exp['final_diversity']
        NEW_version_EPOCH = rep_exp.get('NEW_version_EPOCH')
        
        # Extract diversity values
        diversity_values = [final_diversity[0][epoch][0] for epoch in range(epochs)]
        epochs_list = list(range(epochs))
        
        # Plot diversity line
        ax.plot(epochs_list, diversity_values, label=f'{controller.upper()}', 
               color=colors[i % len(colors)], linewidth=2)
    
    # Add vertical line for new version addition (only if new version was added)
    if add_new_version and controllers:
        rep_exp = comparison_results[controllers[0]]['representative_experiment']
        NEW_version_EPOCH = rep_exp.get('NEW_version_EPOCH')
        if NEW_version_EPOCH is not None:
            ax.axvline(x=NEW_version_EPOCH, color='black', linestyle='--', alpha=0.7, 
                      label='New version Added')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Diversity (bits)')
    # ax.set_ylim(0, ideal_diversity + 0.2) 
    ax.set_title('Diversity Over Time - Representative Experiments')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_show()