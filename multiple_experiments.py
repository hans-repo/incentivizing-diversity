import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from agent import *
from measurement_functions import *
from ziegler_nichols_tuning import *
from reinforcement_learning import PIDController, RLPIDController
from reinforcement_learning_nopid import RLDirectRewardController
from plotting_functions import *
from printing_functions import *

# Configuration for the experiment 
# Default values, can be overridden by command line args
NUM_DIMENSIONS = 1  # Simplified for clarity


def parse_arguments():
    """
    Parse command line arguments, enhanced with comparison mode option.
    """
    parser = argparse.ArgumentParser(description='Run multiple experiments with PID or RL control')
    parser.add_argument('--num_experiments', type=int, default=20, help='Number of experiments to run')
    parser.add_argument('--controller', type=str, default='compare', 
                       choices=['pid', 'rl', 'rlnopid', 'compare'], 
                       help='Controller type: pid (fixed PID), rl (RL-tuned PID), rlnopid (RL direct reward control), or compare')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs per experiment')
    parser.add_argument('--n_agents', type=int, default=544, help='Number of agents')
    
    # Add new version or not
    parser.add_argument('--add_new_version', action='store_true', default=False, 
                       help='Add a new version during the experiment run')

    # RL specific parameters
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate for RL')
    parser.add_argument('--epsilon_decay', type=float, default=0.9995, help='Decay rate for exploration')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for RL')
    parser.add_argument('--action_scale', type=float, default=10, 
                       help='Scale factor for RL actions (lower = smaller adjustments)')
    # PID specific parameters
    parser.add_argument('--p_param', type=float, default=1000, help='P parameter for PID controller')
    parser.add_argument('--i_param', type=float, default=100, help='I parameter for PID controller')
    parser.add_argument('--d_param', type=float, default=100, help='D parameter for PID controller')
    parser.add_argument('--tune_pid', action='store_true', default=True, help='Use Ziegler-Nichols to tune PID parameters')
    parser.add_argument('--tune_once', action='store_true', default=True, help='Tune PID only once and reuse for all experiments')
    # Initial PID parameters for RL mode
    parser.add_argument('--initial_p', type=float, default=10.0, help='Initial P parameter for RL-tuned PID')
    parser.add_argument('--initial_i', type=float, default=1.0, help='Initial I parameter for RL-tuned PID')
    parser.add_argument('--initial_d', type=float, default=1.0, help='Initial D parameter for RL-tuned PID')
    
    # PID scale factor parameters (for RL-tuned PID)
    parser.add_argument('--p_scale_factor', type=float, default=5.0, 
                        help='Scale factor for P parameter relative to reward_scale')
    parser.add_argument('--i_scale_factor', type=float, default=1.0, 
                        help='Scale factor for I parameter relative to reward_scale')
    parser.add_argument('--d_scale_factor', type=float, default=1.0, 
                        help='Scale factor for D parameter relative to reward_scale')
    
    # Comparison mode options
    parser.add_argument('--compare_modes', type=str, nargs='+', 
                       choices=['pid', 'rl', 'rlnopid'], 
                       default=['pid', 'rl', 'rlnopid'],
                       help='Which controllers to compare (default: pid rl). Options: pid, rl, rlnopid')
    
    return parser.parse_args()

def main():
    """
    Main function that handles single controller or comparison mode.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if we're in comparison mode
    if args.controller == "compare":
        # Run comparison mode
        from comparison_mode import run_comparison_mode
        run_comparison_mode(args)
        return
    
    # Original code for non-comparison mode
    # Set up experiment parameters from args
    global num_experiments, controller_type
    num_experiments = args.num_experiments
    controller_type = args.controller
    epochs = args.epochs
    N_AGENTS = args.n_agents
    
    if hasattr(args, 'add_new_version') and args.add_new_version:
        ADD_NEW_version = True
    else:
        ADD_NEW_version = False
    
    print(f"\n{'='*60}")
    print(f"Running {num_experiments} experiments with {controller_type.upper()} controller")
    if ADD_NEW_version:
        print(f"New version will be added during the experiment")
    else:
        print(f"No new version will be added during the experiment")
    print(f"{'='*60}\n")
    
    # Initialize arrays for storing metrics across experiments
    # Each array will store one value per attribute per experiment
    final_diversity_all_experiments = [[] for _ in range(NUM_DIMENSIONS)]
    average_diversity_by_experiment = [[] for _ in range(NUM_DIMENSIONS)]
    ideal_diversity_all_experiments = [[] for _ in range(NUM_DIMENSIONS)]
    largest_version_all_experiments = [[] for _ in range(NUM_DIMENSIONS)]
    
    # Track experiment-specific metrics
    x_axis = []  # Experiment numbers for x-axis
    all_pid_params = []  # Store all PID parameters
    adaptation_metrics = []  # Store adaptation quality across experiments
    recovery_times = []  # Store recovery times across experiments
    phase1_convergence_times = []  # Convergence times for phase 1
    phase2_convergence_times = []  # Convergence times for phase 2
    
    # Variables to store tuned PID parameters from the first experiment
    tuned_pid_params = None
    
    # If PyTorch is available and we're using RL, set a seed for reproducibility
    if controller_type == "rl" and torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    elif controller_type == "rl":
        torch.manual_seed(42)
    
    for i in range(num_experiments):
        print_experiment_header(i+1, num_experiments)
        
        # Define the possible versions an agent can be in
        INITIAL_versionS = ['NO_version', 'version_A', 'version_B', 'version_C', 'version_D']
        # INITIAL_versionS = ['NO_version', 'version_A', 'version_B', 'version_C', 'version_D', 'version_E', 'version_F', 'version_G']
        # INITIAL_versionS = ['NO_version', 'version_A', 'version_B', 'version_C', 'version_D', 'version_E', 'version_F', 'version_G', 'version_H', 'version_I', 'version_J', 'version_K', 'version_L']
        
        # FIXED: Set up new version variables based on flag
        if ADD_NEW_version:
            NEW_version_NAME = 'version_NEW'  # The new version to be added later
            NEW_version_EPOCH = round(0.05 * epochs)  # Epoch at which the new version is added
            POSSIBLE_versionS = INITIAL_versionS.copy()  # Start with initial versions
            POSSIBLE_versionS.append(NEW_version_NAME)
        else:
            NEW_version_NAME = None
            NEW_version_EPOCH = None
            POSSIBLE_versionS = INITIAL_versionS.copy()  # Only initial versions
        
        NUM_versionS = len(POSSIBLE_versionS)

        # PID parameters (will be replaced by tuning if tune_pid is True)
        REWARDS_ADAPTIVE_PARAM = args.p_param
        REWARDS_INTEGRAL_PARAM = args.i_param
        REWARDS_DERIVATIVE_PARAM = args.d_param
        
        BASE_RUN_COST = 100 + 0
        RUN_COST_CEILING = BASE_RUN_COST + round(i*20*BASE_RUN_COST)
        BASE_SWITCH_COST = BASE_RUN_COST

        # Generate evenly spaced values for versions that will actually be used
        version_run_costs = np.linspace(BASE_RUN_COST, RUN_COST_CEILING, NUM_versionS)
        version_RUN_COSTS = {
            version: round(value, 4)
            for version, value in zip(POSSIBLE_versionS, version_run_costs)
        }
        
        # FIXED: Only add run cost for new version if it will actually be used
        if ADD_NEW_version and NEW_version_NAME not in version_RUN_COSTS:
            version_RUN_COSTS[NEW_version_NAME] = round(BASE_RUN_COST + RUN_COST_CEILING, 4)
        
        version_SWITCH_COSTS = {version: BASE_SWITCH_COST for version in POSSIBLE_versionS}
        
        # FIXED: Only set high initial switch cost for new version if it will be used
        if ADD_NEW_version and NEW_version_NAME:
            version_SWITCH_COSTS[NEW_version_NAME] = BASE_SWITCH_COST * 100  # Very high switch cost initially
        
        SWITCH_FREQUENCY_PARAM = 0.0 + 2
        
        x_axis.append(i+1)  # Store experiment number (starting from 1 for better readability)
        
        # FIXED: Initialize version rewards only for versions that will be used
        version_rewards = [{version: 0 for version in POSSIBLE_versionS} for _ in range(NUM_DIMENSIONS)]
        accumulated_error = [{version: 0 for version in POSSIBLE_versionS} for _ in range(NUM_DIMENSIONS)]
        last_error = [{version: 0 for version in POSSIBLE_versionS} for _ in range(NUM_DIMENSIONS)]
        
        # FIXED: Only pre-add new version if it will actually be used
        if ADD_NEW_version and NEW_version_NAME:
            for k in range(NUM_DIMENSIONS):
                # Set initial reward very low to ensure zero initial adoption
                version_rewards[k][NEW_version_NAME] = 0  # No reward initially
                accumulated_error[k][NEW_version_NAME] = 0
                last_error[k][NEW_version_NAME] = 0
        
        # Initialize the appropriate controller based on the controller_type
        if controller_type == "pid":
            # Only run Ziegler-Nichols tuning for the first experiment if requested
            if i == 0 and args.tune_pid and tuned_pid_params is None:
                print("\n=== Running Ziegler-Nichols tuning for experiment", i+1, "===")
                REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM = ziegler_nichols_tuning(
                    None, INITIAL_versionS, epochs=epochs, n_agents=N_AGENTS)
                
                # Store the tuned parameters for subsequent experiments
                tuned_pid_params = (REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM)
            elif i > 0 and tuned_pid_params is not None:
                # Reuse previously tuned PID parameters
                print(f"\n=== Reusing PID parameters from first experiment for experiment {i+1} ===")
                REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM = tuned_pid_params
            
            # Store the PID parameters used for this experiment
            all_pid_params.append((REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM))
            
            print_pid_parameters(REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM)
            
            # Initialize the PID controller
            reward_controller = PIDController(
                POSSIBLE_versionS,
                REWARDS_ADAPTIVE_PARAM,
                REWARDS_INTEGRAL_PARAM,
                REWARDS_DERIVATIVE_PARAM
            )

        elif controller_type == "rl":
            # RL-tuned PID mode
            print_rl_parameters(args.epsilon, args.epsilon_decay, args.learning_rate, args.action_scale, 
                            args.initial_p, args.initial_i, args.initial_d,
                            args.p_scale_factor, args.i_scale_factor, args.d_scale_factor)
            
            # Initialize the RL-tuned PID controller
            reward_controller = RLPIDController(
                POSSIBLE_versionS,
                N_AGENTS,
                initial_p=args.initial_p,
                initial_i=args.initial_i,
                initial_d=args.initial_d,
                epsilon=args.epsilon,
                epsilon_decay=args.epsilon_decay,
                learning_rate=args.learning_rate,
                action_scale=args.action_scale,
                update_frequency=5,
                batch_size=16,
                p_scale_factor=args.p_scale_factor,
                i_scale_factor=args.i_scale_factor,
                d_scale_factor=args.d_scale_factor
            )
            
            # Initialize with zeros for PID params in the tracking array (will be updated during run)
            all_pid_params.append((args.initial_p, args.initial_i, args.initial_d))

        elif controller_type == "rlnopid":
            # Direct reward control mode
            print("\n=== Using RL Direct Reward Control (no PID) ===")
            print(f"  • Epsilon: {args.epsilon}")
            print(f"  • Epsilon decay: {args.epsilon_decay}")
            print(f"  • Learning rate: {args.learning_rate}")
            print(f"  • Action scale: {args.action_scale}")

            # Initialize the RL direct reward controller
            reward_controller = RLDirectRewardController(
                POSSIBLE_versionS,
                N_AGENTS,
                epsilon=args.epsilon,
                epsilon_decay=args.epsilon_decay,
                learning_rate=args.learning_rate,
                action_scale=args.action_scale,
                update_frequency=5,
                batch_size=16
            )
            
            # For tracking purposes, use zeros for PID params
            all_pid_params.append((0, 0, 0))
            
        # FIXED: Create initial agents with only initial versions (never include new version initially)
        agents = generate_agents(N_AGENTS, INITIAL_versionS, version_rewards, version_RUN_COSTS, version_SWITCH_COSTS, SWITCH_FREQUENCY_PARAM)
        agents_real_history = [[] for _ in range(NUM_DIMENSIONS)]
        agents_declared_history = [[] for _ in range(NUM_DIMENSIONS)]
        
        # Track versions available at each epoch for proper diversity calculation
        versions_available_at_epoch = []
        
        # Track PID parameters over time (for RL mode)
        pid_params_history = []
        
        # Track total rewards allocated over time
        total_rewards_history = []
        reward_per_version_history = []
        
        # Run the main experiment
        print("\n=== Starting main experiment ===")
        for epoch in range(epochs):
            # FIXED: Only add new version if the flag is enabled
            if ADD_NEW_version and epoch == NEW_version_EPOCH:
                # Reset the switch cost for the new version to normal level
                version_SWITCH_COSTS[NEW_version_NAME] = BASE_SWITCH_COST
                
                # Update agents to know about the new version
                for agent in agents:
                    # Update agent's possible versions list
                    agent.possible_versions = POSSIBLE_versionS.copy()
                    
                    # Set normal switch cost for the new version
                    agent.switch_cost[NEW_version_NAME] = BASE_SWITCH_COST
                
                # Preserve existing reward values for existing versions
                preserved_rewards = {}
                for k in range(NUM_DIMENSIONS):
                    preserved_rewards[k] = {version: version_rewards[k][version] for version in INITIAL_versionS}
                
                # Update reward controller with the new version
                if controller_type == "rl":
                    # For RL controller, we need to reinitialize with the new version list
                    # but keep the same PID parameters that have been learned
                    current_p = reward_controller.p_param
                    current_i = reward_controller.i_param
                    current_d = reward_controller.d_param
                    current_reward_scale = reward_controller.reward_scale
                    current_epsilon = reward_controller.epsilon
                    current_steps_done = reward_controller.steps_done  # Preserve steps_done counter
                    
                    # Store current PID errors for proper scaling
                    old_accumulated_errors = {version: reward_controller.accumulated_error[version] 
                                        for version in INITIAL_versionS if version in reward_controller.accumulated_error}
                    old_last_errors = {version: reward_controller.last_error[version] 
                                    for version in INITIAL_versionS if version in reward_controller.last_error}
                    
                    # Calculate old and new ideal shares
                    old_ideal_share = 1.0 / (len(INITIAL_versionS) - 1)  # -1 for NO_version
                    new_ideal_share = 1.0 / (len(POSSIBLE_versionS) - 1)  # -1 for NO_version
                    # Scale factor for error terms (how much the ideal distribution changed)
                    error_scale_factor = new_ideal_share / old_ideal_share
                    
                    # Create new controller
                    new_controller = RLPIDController(
                        POSSIBLE_versionS,
                        N_AGENTS,
                        initial_p=current_p,
                        initial_i=current_i,
                        initial_d=current_d,
                        epsilon=current_epsilon,
                        epsilon_decay=args.epsilon_decay,
                        learning_rate=args.learning_rate,
                        action_scale=args.action_scale,
                        update_frequency=5,
                        batch_size=16,
                        p_scale_factor=args.p_scale_factor,
                        i_scale_factor=args.i_scale_factor,
                        d_scale_factor=args.d_scale_factor
                    )
                    
                    # Set the learned reward scale
                    new_controller.reward_scale = current_reward_scale
                    
                    # IMPORTANT: Restore steps_done counter to maintain exploration rate
                    new_controller.steps_done = current_steps_done
                    
                    # Copy over best parameters found so far
                    new_controller.best_pid_params = reward_controller.best_pid_params
                    new_controller.best_diversity_ratio = reward_controller.best_diversity_ratio
                    new_controller.best_reward_scale = reward_controller.best_reward_scale
                    
                    # Copy over replay buffer if possible
                    if hasattr(reward_controller, 'replay_buffer'):
                        new_controller.replay_buffer = reward_controller.replay_buffer
                    
                    # CRITICAL: Scale accumulated errors for existing versions to prevent integral term disruption
                    for version in INITIAL_versionS:
                        if version in old_accumulated_errors:
                            # Scale the errors by the ratio of new/old ideal shares
                            new_controller.accumulated_error[version] = old_accumulated_errors[version] * error_scale_factor
                        if version in old_last_errors:
                            new_controller.last_error[version] = old_last_errors[version] * error_scale_factor
                            
                    # Add a reference to previous rewards for initialization
                    new_controller.prev_rewards = {version: preserved_rewards[0][version] for version in INITIAL_versionS 
                                                if version in preserved_rewards[0]}
                    
                    # Add a flag for transition mode
                    new_controller.in_transition = True
                    new_controller.transition_epochs = 1000  # Number of epochs for smooth transition
                    new_controller.transition_progress = 0  # Current progress (0-1)
                    
                    # Add prev_rewards for each attribute
                    new_controller.prev_rewards_by_attr = {}
                    for k in range(NUM_DIMENSIONS):
                        if k in preserved_rewards:
                            new_controller.prev_rewards_by_attr[k] = preserved_rewards[k]
                    
                    # Backup prev_rewards
                    prev_rewards_backup = new_controller.prev_rewards.copy() if hasattr(new_controller, 'prev_rewards') else {}
                    
                    reward_controller = new_controller
                    
                    # CRITICAL: Double-check that prev_rewards survived and restore if needed
                    if not hasattr(reward_controller, 'prev_rewards') or len(reward_controller.prev_rewards) == 0:
                        print("WARNING: prev_rewards missing after controller assignment, restoring from backup")
                        reward_controller.prev_rewards = prev_rewards_backup
                    
                elif controller_type == "pid":
                    # For PID controller, we need to update versions and initialize new version error tracking
                    reward_controller.versions = POSSIBLE_versionS.copy()
                    reward_controller.accumulated_error[NEW_version_NAME] = 0
                    reward_controller.last_error[NEW_version_NAME] = 0
                
                # Restore preserved reward values after controller initialization
                for k in range(NUM_DIMENSIONS):
                    for version, value in preserved_rewards[k].items():
                        version_rewards[k][version] = value
                    
                    # Set new version's reward to zero initially
                    version_rewards[k][NEW_version_NAME] = 0
                
                # Print info about the new version
                print_new_version_added(NEW_version_NAME, epoch, len(POSSIBLE_versionS), 
                                    version_SWITCH_COSTS[NEW_version_NAME])
            
            # FIXED: Keep track of which versions were available at this epoch
            current_versions = INITIAL_versionS.copy()
            if ADD_NEW_version and epoch >= NEW_version_EPOCH:
                current_versions = POSSIBLE_versionS.copy()
            
            versions_available_at_epoch.append(current_versions)
            
            # Calculate current diversity for RL reward calculation
            current_diversity = None
            if (controller_type == "rl" or controller_type == "rlnopid") and epoch > 0:
                diversity_result = calculate_diversity(
                    [agents_declared_history[0]], [epoch-1], 
                    versions_available_at_epoch[epoch-1], N_AGENTS)
                current_diversity = diversity_result[0][epoch-1][0]
                
                # Update agent counts in RL controller for accurate agent version calculations
                agents_in_versions = {version: 0 for version in current_versions}
                for agent in agents:
                    agent_version = agent.declared_version[0]  # Using attribute 0
                    agents_in_versions[agent_version] = agents_in_versions.get(agent_version, 0) + 1
                reward_controller.agent_counts = agents_in_versions
                
            # Get ideal diversity for current set of versions
            ideal_diversity = get_ideal_diversity(current_versions)
                
            # Update rewards based on current version distribution using the appropriate controller
            if controller_type == "pid":
                # Use PID controller to update rewards
                version_rewards = reward_controller.update_rewards(agents, current_versions, version_rewards)
                
                # Calculate total rewards allocated
                total_rewards = 0
                for version in current_versions:
                    total_rewards += version_rewards[0][version]
                total_rewards_history.append(total_rewards)
                reward_per_version_history.append(version_rewards[0].copy())
                
                # Track PID parameters (constant for fixed PID controller)
                pid_params_history.append((
                    reward_controller.p_param,
                    reward_controller.i_param,
                    reward_controller.d_param
                ))
                
            elif controller_type == "rl":
                # Use RL-tuned PID controller to update rewards
                version_rewards = reward_controller.update_rewards(
                    agents, current_versions, version_rewards,
                    current_diversity=current_diversity, 
                    ideal_diversity=ideal_diversity,
                    epoch=epoch
                )
                # Track PID parameters for plotting
                pid_params_history.append((
                    reward_controller.p_param,
                    reward_controller.i_param,
                    reward_controller.d_param
                ))
                
                # Update the PID parameters in the tracking array (for final reporting)
                all_pid_params[i] = (
                    reward_controller.p_param,
                    reward_controller.i_param,
                    reward_controller.d_param
                )
                
                # Calculate total rewards allocated correctly
                total_rewards = 0
                for version in current_versions:
                    total_rewards += version_rewards[0][version]
                
                # Store the calculated total rewards
                total_rewards_history.append(total_rewards)
                reward_per_version_history.append(version_rewards[0].copy())

            elif controller_type == "rlnopid":
                # Use RL direct reward controller to update rewards
                version_rewards = reward_controller.update_rewards(
                    agents, current_versions, version_rewards,
                    current_diversity=current_diversity, 
                    ideal_diversity=ideal_diversity,
                    epoch=epoch
                )
                
                # No PID parameters to track
                pid_params_history.append((0, 0, 0))
                
                # Track the direct rewards being used
                if hasattr(reward_controller, 'current_rewards'):
                    current_rewards_snapshot = reward_controller.current_rewards.copy()
                    # You can store this in a separate history if needed
                
                # Update the tracking array with zeros for PID params
                all_pid_params[i] = (0, 0, 0)
                
                # Calculate total rewards allocated
                total_rewards = 0
                for version in current_versions:
                    total_rewards += version_rewards[0][version]
                
                # Store the calculated total rewards
                total_rewards_history.append(total_rewards)
                reward_per_version_history.append(version_rewards[0].copy())
            
            # Distribute rewards among agents
            version_rewards_last_epoch = distribute_rewards(agents, current_versions, version_rewards)
            
            # Record versions of all agents
            for k in range(NUM_DIMENSIONS):
                agents_real_history[k].append([agent.real_version[k] for agent in agents])
                agents_declared_history[k].append([agent.declared_version[k] for agent in agents])
            
            # Update agent decisions
            for agent in agents:
                agent.decision(version_rewards_last_epoch, malicious=False)
                
            # Print status at key epochs
            should_print = (epoch % 1000 == 0 or 
                          (ADD_NEW_version and (epoch == NEW_version_EPOCH or epoch == NEW_version_EPOCH + 1)))
            
            if should_print:
                # Count agents in each version for the first attribute
                version_counts = {version: 0 for version in current_versions}
                for agent in agents:
                    version_counts[agent.declared_version[0]] += 1
                
                # Calculate and print current diversity
                current_epoch_diversity = None
                if epoch > 0:  # Skip first epoch
                    temp_diversity = calculate_diversity([agents_declared_history[0]], [epoch-1], 
                                                       versions_available_at_epoch[epoch-1], N_AGENTS)
                    current_epoch_diversity = temp_diversity[0][epoch-1][0]
                
                # Print status for this epoch
                print_epoch_status(epoch, version_counts, version_rewards_last_epoch, 
                                 total_rewards_history[-1], controller_type, 
                                 reward_controller, current_epoch_diversity)
                
                # Specifically print new version count when it matters
                if ADD_NEW_version:
                    print_new_version_status(epoch, NEW_version_EPOCH, NEW_version_NAME, 
                                         version_counts, version_rewards_last_epoch)
        
        # Calculate final diversity with awareness of dynamic version set
        final_diversity = [{}] * NUM_DIMENSIONS
        for k in range(NUM_DIMENSIONS):
            final_diversity[k] = {}
            for epoch in range(epochs):
                # Use the correct set of possible versions for each epoch
                epoch_diversity = calculate_diversity(
                    [agents_declared_history[k]], [epoch], 
                    versions_available_at_epoch[epoch], N_AGENTS)
                final_diversity[k][epoch] = epoch_diversity[0][epoch]
        
        # Calculate the number of epochs in the last 25%
        last_25_percent = int(epochs * 0.25)
        
        for k in range(NUM_DIMENSIONS):
            # Extract the last 25% of diversity values - these are already floats!
            last_25_percent_diversity = [final_diversity[k][epoch][0] for epoch in range(epochs - last_25_percent, epochs)]
            
            # Calculate the average diversity and store it
            avg_diversity = sum(last_25_percent_diversity) / len(last_25_percent_diversity)
            average_diversity_by_experiment[k].append(avg_diversity)
            
            # Calculate ideal diversity for the actual final set of versions used
            final_versions = POSSIBLE_versionS if ADD_NEW_version else INITIAL_versionS
            ideal_diversity = get_ideal_diversity(final_versions)
            ideal_diversity_all_experiments[k].append(ideal_diversity)
            
            # Calculate largest version metric - we need to import this function or calculate it
            # For now, let's calculate it directly from the final diversity data
            # The largest version share is 1 - diversity_ratio (approximately)
            diversity_ratio = avg_diversity / ideal_diversity if ideal_diversity > 0 else 0
            largest_version_share = max(0, 1 - diversity_ratio)  # Approximate largest version share
            
            largest_version_all_experiments[k].append(largest_version_share)
        
        # Get diversity values and ideal values for convergence analysis
        diversity_values = [final_diversity[0][epoch][0] for epoch in range(epochs)]
        ideal_before = get_ideal_diversity(INITIAL_versionS)
        ideal_after = get_ideal_diversity(POSSIBLE_versionS if ADD_NEW_version else INITIAL_versionS)
        
        # Analyze convergence in both phases using new enhanced metrics
        from measurement_functions import calculate_convergence_metrics, calculate_resilience_metrics
        
        # FIXED: Calculate convergence metrics based on whether new version was added
        if ADD_NEW_version:
            phase1_values = diversity_values[:NEW_version_EPOCH]
            phase1_convergence = calculate_convergence_metrics(phase1_values, ideal_before)
            
            phase2_values = diversity_values[NEW_version_EPOCH:]  
            phase2_convergence = calculate_convergence_metrics(phase2_values, ideal_after)
            
            # Calculate resilience metrics
            resilience_metrics = calculate_resilience_metrics(
                diversity_values, NEW_version_EPOCH, ideal_before, ideal_after)
        else:
            # Single phase analysis when no new version is added
            phase1_values = diversity_values
            phase1_convergence = calculate_convergence_metrics(phase1_values, ideal_before)
            
            # No phase 2 or resilience metrics
            phase2_convergence = {'time_to_convergence': None, 'convergence_rate': None, 
                                'steady_version_error': None, 'final_diversity_quality': None}
            resilience_metrics = {'initial_impact_magnitude': None, 'initial_impact_percentage': None,
                                'recovery_time_epochs': None, 'stability_time_epochs': None,
                                'final_adaptation_quality': None}
        
        # Store convergence metrics for this experiment
        phase1_convergence_time = phase1_convergence['time_to_convergence']
        phase2_convergence_time = phase2_convergence['time_to_convergence']
        
        phase1_convergence_times.append(phase1_convergence_time if phase1_convergence_time is not None else epochs)
        phase2_convergence_times.append(phase2_convergence_time if phase2_convergence_time is not None else epochs)
        
        # FIXED: Store adaptation quality and recovery time based on whether new version was added
        if ADD_NEW_version:
            if "final_adaptation_quality" in resilience_metrics and resilience_metrics["final_adaptation_quality"] is not None:
                adaptation_metrics.append(resilience_metrics["final_adaptation_quality"])
            else:
                adaptation_metrics.append(0.0)
                
            if "recovery_time_epochs" in resilience_metrics and resilience_metrics["recovery_time_epochs"] is not None:
                recovery_times.append(resilience_metrics["recovery_time_epochs"])
            else:
                recovery_times.append(epochs)
        else:
            # No adaptation metrics needed when no new version is added
            adaptation_metrics.append(1.0)  # Perfect adaptation since no change occurred
            recovery_times.append(0)  # No recovery needed
    
        # Display results for single experiment case
        if num_experiments == 1:
            epochs_list = list(range(epochs))
            
            # Compile comprehensive performance metrics
            from measurement_functions import compile_system_performance_metrics, calculate_resource_efficiency
            
            # Calculate resource efficiency metrics
            resource_efficiency = calculate_resource_efficiency(diversity_values, total_rewards_history, ideal_after)
            
            # If using RL controller, calculate parameter evolution metrics
            pid_evolution = None
            if controller_type == "rl":
                from measurement_functions import analyze_pid_parameter_evolution
                pid_evolution = analyze_pid_parameter_evolution(pid_params_history, diversity_values, 
                                                              NEW_version_EPOCH if ADD_NEW_version else None)
            
            # Compile all metrics
            performance_metrics = compile_system_performance_metrics(
                diversity_values, total_rewards_history, pid_params_history, 
                NEW_version_EPOCH if ADD_NEW_version else None, ideal_before, ideal_after
            )
            
            # Print detailed metrics table
            from plotting_functions import create_metrics_summary_table
            
            # For RL mode, create a dummy PID metrics to print
            if controller_type == "rl":
                # Create metrics for an ideal PID controller for comparison
                # (this is just for display purposes)
                pid_metrics = {
                    'phase1_convergence': {'time_to_convergence': None, 'convergence_rate': None, 
                                         'steady_version_error': None, 'final_diversity_quality': None},
                    'phase2_convergence': {'time_to_convergence': None, 'convergence_rate': None, 
                                         'steady_version_error': None, 'final_diversity_quality': None},
                    'resilience': {'initial_impact_magnitude': None, 'initial_impact_percentage': None,
                                 'recovery_time_epochs': None, 'stability_time_epochs': None,
                                 'final_adaptation_quality': None},
                    'phase1_efficiency': {'avg_reward_per_epoch': None, 'total_cumulative_reward': None,
                                       'diversity_per_reward_unit': None, 'reward_volatility': None,
                                       'efficiency_trend': None},
                    'phase2_efficiency': {'avg_reward_per_epoch': None, 'total_cumulative_reward': None,
                                       'diversity_per_reward_unit': None, 'reward_volatility': None,
                                       'efficiency_trend': None},
                    'overall_efficiency': {'avg_reward_per_epoch': None, 'total_cumulative_reward': None,
                                        'diversity_per_reward_unit': None, 'reward_volatility': None,
                                        'efficiency_trend': None},
                    'pid_evolution': None
                }
                
                # Print the single controller metrics
                print("\n\n" + "="*80)
                print(f"DETAILED PERFORMANCE METRICS FOR {controller_type.upper()} CONTROLLER")
                print("="*80)
                
                # Print the relevant sections of the metrics
                print(f"\n🔍 CONVERGENCE METRICS")
                print(f"  Time to initial convergence (Phase 1): {phase1_convergence_time if phase1_convergence_time is not None else 'N/A'}")
                if ADD_NEW_version:
                    print(f"  Time to re-convergence (Phase 2): {phase2_convergence_time if phase2_convergence_time is not None else 'N/A'}")
                
                if ADD_NEW_version:
                    print(f"\n🔄 RESILIENCE METRICS")
                    print(f"  Recovery time (epochs): {recovery_times[-1]}")
                    print(f"  Final adaptation quality: {adaptation_metrics[-1]*100:.1f}%")
                
                print(f"\n💰 RESOURCE EFFICIENCY")
                print(f"  Avg reward per epoch: {resource_efficiency['avg_reward_per_epoch']:.2f}")
                print(f"  Total cumulative reward: {resource_efficiency['total_cumulative_reward']:.2f}")
                
                if pid_evolution:
                    final_comp = pid_evolution['final_composition']
                    print(f"\n⚙️ PID PARAMETER EVOLUTION")
                    print(f"  Final P: {pid_params_history[-1][0]:.2f} ({final_comp['p_percentage']:.1f}%)")
                    print(f"  Final I: {pid_params_history[-1][1]:.2f} ({final_comp['i_percentage']:.1f}%)")
                    print(f"  Final D: {pid_params_history[-1][2]:.2f} ({final_comp['d_percentage']:.1f}%)")
                
            # Plot diversity over time
            from plotting_functions import plot_diversity_over_time
            plot_diversity_over_time(final_diversity, epochs_list, NUM_DIMENSIONS, 
                                  NEW_version_EPOCH if ADD_NEW_version else None, 
                                  NEW_version_NAME if ADD_NEW_version else None, 
                                  INITIAL_versionS, POSSIBLE_versionS, controller_type,
                                  REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM,
                                  pid_params_history, ADD_NEW_version)
            
            # Plot total rewards over time
            from plotting_functions import plot_total_rewards
            plot_total_rewards(epochs_list, total_rewards_history, 
                            NEW_version_EPOCH if ADD_NEW_version else None, 
                            NEW_version_NAME if ADD_NEW_version else None, 
                            controller_type, REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, 
                            REWARDS_DERIVATIVE_PARAM, pid_params_history, ADD_NEW_version)
            
            # Plot rewards per version over time
            from plotting_functions import plot_rewards_per_version
            plot_rewards_per_version(epochs_list, reward_per_version_history, POSSIBLE_versionS, 
                                NEW_version_EPOCH if ADD_NEW_version else None, 
                                NEW_version_NAME if ADD_NEW_version else None, 
                                controller_type, REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, 
                                REWARDS_DERIVATIVE_PARAM, pid_params_history, ADD_NEW_version)
            
            # For RL, also plot PID parameter evolution
            if controller_type == "rl":
                from plotting_functions import plot_pid_parameters
                plot_pid_parameters(epochs_list, pid_params_history, 
                                  NEW_version_EPOCH if ADD_NEW_version else None, 
                                  NEW_version_NAME if ADD_NEW_version else None, ADD_NEW_version)
            
            # Print final results and analysis using new metrics
            from plotting_functions import print_single_experiment_results
            print_single_experiment_results(POSSIBLE_versionS, version_rewards_last_epoch, 
                                          controller_type, total_rewards_history, pid_params_history,
                                          reward_controller)
            
            # Print enhanced convergence analysis
            from plotting_functions import print_convergence_analysis
            print_convergence_analysis(diversity_values, NEW_version_EPOCH if ADD_NEW_version else None, 
                                     INITIAL_versionS, POSSIBLE_versionS, 
                                     {"phase1": phase1_convergence, "phase2": phase2_convergence}, ADD_NEW_version)

            # Print enhanced adaptability metrics (only if new version was added)
            if ADD_NEW_version:
                from plotting_functions import print_adaptability_metrics
                print_adaptability_metrics(resilience_metrics)
            
            # Print enhanced reward allocation metrics
            from plotting_functions import print_reward_allocation_metrics
            print_reward_allocation_metrics(total_rewards_history, NEW_version_EPOCH if ADD_NEW_version else None)
            
            # Plot agent distribution
            from plotting_functions import plot_agent_distribution
            for k in range(NUM_DIMENSIONS):
                plot_agent_distribution(agents_declared_history, epochs_list, POSSIBLE_versionS, 
                                      NEW_version_EPOCH if ADD_NEW_version else None, 
                                      NEW_version_NAME if ADD_NEW_version else None, 
                                      controller_type, k, ADD_NEW_version)
                
    # For multiple experiments, create summary plots with experiment number as x-axis
    if num_experiments > 1:
        from plotting_functions import print_multi_experiment_header
        print_multi_experiment_header(num_experiments, controller_type)
        print("="*60)
        
        # Print average metrics across all experiments
        for k in range(NUM_DIMENSIONS):
            avg_diversity = sum(average_diversity_by_experiment[k]) / num_experiments
            avg_largest_version = sum(largest_version_all_experiments[k]) / num_experiments
            
            print(f"\nAttribute {k} Results:")
            print(f"  • Average diversity (last 25% of epochs): {avg_diversity:.4f} bits")
            print(f"  • Average largest version share: {avg_largest_version:.2%}")
            
            ideal_div = sum(ideal_diversity_all_experiments[k]) / num_experiments
            print(f"  • Average ideal diversity: {ideal_div:.4f} bits")
            print(f"  • Average diversity achieved: {(avg_diversity/ideal_div)*100:.1f}% of ideal")
        
        # FIXED: Print adaptation metrics only if new version was added
        if ADD_NEW_version:
            avg_adaptation = sum(adaptation_metrics) / num_experiments
            avg_recovery = sum(recovery_times) / num_experiments
            avg_p1_conv = sum(phase1_convergence_times) / num_experiments
            avg_p2_conv = sum(phase2_convergence_times) / num_experiments
            
            print("\nSystem Adaptation Metrics:")
            print(f"  • Average adaptation quality: {avg_adaptation*100:.1f}% of ideal")
            print(f"  • Average recovery time: {avg_recovery:.1f} epochs")
            print(f"  • Average Phase 1 convergence time: {avg_p1_conv:.1f} epochs")
            print(f"  • Average Phase 2 convergence time: {avg_p2_conv:.1f} epochs")
        else:
            avg_p1_conv = sum(phase1_convergence_times) / num_experiments
            print("\nSystem Convergence Metrics:")
            print(f"  • Average convergence time: {avg_p1_conv:.1f} epochs")
        
        # Create and display plots
        print("\nGenerating summary plots...")
        
        # Plot average diversity across experiments
        from plotting_functions import plot_average_diversity
        plot_average_diversity(x_axis, average_diversity_by_experiment, NUM_DIMENSIONS, controller_type, ideal_div)
        
        # FIXED: Plot adaptation quality only if new version was added
        if ADD_NEW_version:
            from plotting_functions import plot_adaptation_quality
            plot_adaptation_quality(x_axis, adaptation_metrics, controller_type)
            
            # Plot recovery times
            from plotting_functions import plot_recovery_times
            plot_recovery_times(x_axis, recovery_times)
            
            # Plot convergence times
            from plotting_functions import plot_convergence_times
            plot_convergence_times(x_axis, phase1_convergence_times, phase2_convergence_times)
        else:
            # Plot only single phase convergence times
            from plotting_functions import plot_single_phase_convergence_times
            plot_single_phase_convergence_times(x_axis, phase1_convergence_times)
        
        # Plot PID parameters
        from plotting_functions import plot_pid_parameters_across_experiments
        plot_pid_parameters_across_experiments(x_axis, all_pid_params)
        
        # Plot largest version share
        from plotting_functions import plot_largest_version_share
        plot_largest_version_share(x_axis, largest_version_all_experiments, NUM_DIMENSIONS)


if __name__ == "__main__":
    main()