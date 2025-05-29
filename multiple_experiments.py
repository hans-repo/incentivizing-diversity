import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from agent import *
from measurement_functions import *
from ziegler_nichols_tuning import *
from reinforcement_learning import PIDController, RLPIDController
from plotting_functions import *

# Configuration for the experiment 
# Default values, can be overridden by command line args
NUM_ATTRIBUTES = 1  # Simplified for clarity


def parse_arguments():
    """
    Parse command line arguments, enhanced with comparison mode option.
    """
    parser = argparse.ArgumentParser(description='Run multiple experiments with PID or RL control')
    parser.add_argument('--num_experiments', type=int, default=1, help='Number of experiments to run')
    parser.add_argument('--controller', type=str, default='pid', choices=['pid', 'rl', 'compare'], 
                       help='Controller type: pid, rl (reinforcement learning), or compare (run both)')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs per experiment')
    parser.add_argument('--n_agents', type=int, default=100, help='Number of agents')
    
    # RL specific parameters
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate for RL')
    parser.add_argument('--epsilon_decay', type=float, default=0.9, help='Decay rate for exploration')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for RL')
    parser.add_argument('--action_scale', type=float, default=0.1, 
                       help='Scale factor for RL actions (lower = smaller adjustments)')
    parser.add_argument('--efficiency_weight', type=float, default=0.0001,
                       help='Weight for reward efficiency vs diversity (0.0-1.0, higher = more emphasis on minimizing rewards)')
    
    # PID specific parameters
    parser.add_argument('--p_param', type=float, default=1000, help='P parameter for PID controller')
    parser.add_argument('--i_param', type=float, default=100, help='I parameter for PID controller')
    parser.add_argument('--d_param', type=float, default=100, help='D parameter for PID controller')
    parser.add_argument('--tune_pid', action='store_true', default=True, help='Use Ziegler-Nichols to tune PID parameters')
    
    # Initial PID parameters for RL mode
    parser.add_argument('--initial_p', type=float, default=10.0, help='Initial P parameter for RL-tuned PID')
    parser.add_argument('--initial_i', type=float, default=1.0, help='Initial I parameter for RL-tuned PID')
    parser.add_argument('--initial_d', type=float, default=1.0, help='Initial D parameter for RL-tuned PID')
    
    # New PID scale factor parameters (configurable instead of hardcoded)
    parser.add_argument('--p_scale_factor', type=float, default=5.0, 
                        help='Scale factor for P parameter relative to reward_scale')
    parser.add_argument('--i_scale_factor', type=float, default=1.0, 
                        help='Scale factor for I parameter relative to reward_scale')
    parser.add_argument('--d_scale_factor', type=float, default=1.0, 
                        help='Scale factor for D parameter relative to reward_scale')
    
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
    
    print(f"\n{'='*60}")
    print(f"Running {num_experiments} experiments with {controller_type.upper()} controller")
    print(f"{'='*60}\n")
    
    # Initialize arrays for storing metrics across experiments
    # Each array will store one value per attribute per experiment
    final_diversity_all_experiments = [[] for _ in range(NUM_ATTRIBUTES)]
    average_diversity_by_experiment = [[] for _ in range(NUM_ATTRIBUTES)]
    ideal_diversity_all_experiments = [[] for _ in range(NUM_ATTRIBUTES)]
    largest_state_all_experiments = [[] for _ in range(NUM_ATTRIBUTES)]
    
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
        
        # Define the possible states an agent can be in
        # INITIAL_STATES = ['NO_STATE', 'State_A', 'State_B', 'State_C', 'State_D']
        INITIAL_STATES = ['NO_STATE', 'State_A', 'State_B', 'State_C', 'State_D', 'State_E', 'State_F', 'State_G', 'State_H']
        NEW_STATE_NAME = 'State_NEW'  # The new state to be added later
        NEW_STATE_EPOCH = round(0.5 * 1 * epochs)  # Epoch at which the new state is added
        
        POSSIBLE_STATES = INITIAL_STATES.copy()  # Start with initial states
        POSSIBLE_STATES.append(NEW_STATE_NAME)
        NUM_STATES = len(POSSIBLE_STATES)
        PER_NODE_REWARD_BASE = 1000
        PER_NODE_REWARD = PER_NODE_REWARD_BASE  # Per node reward per epoch in an evenly distributed system
        
        # Define fixed rewards for each state
        BASE_REWARDS = PER_NODE_REWARD_BASE * N_AGENTS / (NUM_STATES-1)  # -1 for NO_STATE
        
        # PID parameters (will be replaced by tuning if tune_pid is True)
        REWARDS_ADAPTIVE_PARAM = args.p_param
        REWARDS_INTEGRAL_PARAM = args.i_param
        REWARDS_DERIVATIVE_PARAM = args.d_param
        
        BASE_RUN_COST = 100 + 0
        RUN_COST_CEILING = BASE_RUN_COST + 2*BASE_RUN_COST + BASE_RUN_COST*0
        BASE_SWITCH_COST = BASE_RUN_COST

        # Generate evenly spaced values for initial states
        state_run_costs = np.linspace(BASE_RUN_COST, RUN_COST_CEILING, NUM_STATES)
        STATE_RUN_COSTS = {
            state: round(value, 4)
            for state, value in zip(POSSIBLE_STATES, state_run_costs)
        }
        
        # Add run cost for the new state that will be added later
        # Set a high initial run cost to discourage early adoption
        STATE_RUN_COSTS[NEW_STATE_NAME] = round(BASE_RUN_COST + 1.5*BASE_RUN_COST, 4)
        
        STATE_SWITCH_COSTS = {state: BASE_SWITCH_COST for state in POSSIBLE_STATES}
        # Set a high initial switch cost for the new state to ensure zero initial nodes
        STATE_SWITCH_COSTS[NEW_STATE_NAME] = BASE_SWITCH_COST * 100  # Very high switch cost initially
        SWITCH_FREQUENCY_PARAM = 2.0
        
        x_axis.append(i+1)  # Store experiment number (starting from 1 for better readability)
        
        # Initialize state rewards and accumulated error
        state_rewards = [{state: 0 for state in POSSIBLE_STATES} for _ in range(NUM_ATTRIBUTES)]
        accumulated_error = [{state: 0 for state in POSSIBLE_STATES} for _ in range(NUM_ATTRIBUTES)]
        last_error = [{state: 0 for state in POSSIBLE_STATES} for _ in range(NUM_ATTRIBUTES)]
        
        # Pre-add the new state to rewards and error tracking, but make it unattractive
        for k in range(NUM_ATTRIBUTES):
            # Set initial reward very low to ensure zero initial adoption
            state_rewards[k][NEW_STATE_NAME] = 0  # No reward initially
            accumulated_error[k][NEW_STATE_NAME] = 0
            last_error[k][NEW_STATE_NAME] = 0
        
        # Initialize the appropriate controller based on the controller_type
        if controller_type == "pid":
            # Only run Ziegler-Nichols tuning for the first experiment if requested
            if i == 0 and args.tune_pid and tuned_pid_params is None:
                print("\n=== Running Ziegler-Nichols tuning for experiment", i+1, "===")
                REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM = ziegler_nichols_tuning(
                    None, INITIAL_STATES, BASE_REWARDS, epochs=epochs, n_agents=N_AGENTS)
                
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
                POSSIBLE_STATES,
                REWARDS_ADAPTIVE_PARAM,
                REWARDS_INTEGRAL_PARAM,
                REWARDS_DERIVATIVE_PARAM
            )
        
        elif controller_type == "rl":
            print_rl_parameters(args.epsilon, args.epsilon_decay, args.learning_rate, args.action_scale, 
                              args.efficiency_weight, args.initial_p, args.initial_i, args.initial_d,
                              args.p_scale_factor, args.i_scale_factor, args.d_scale_factor)
            
            # Initialize the RL-tuned PID controller without reward_scale parameter
            reward_controller = RLPIDController(
                POSSIBLE_STATES,
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
                reward_efficiency_weight=args.efficiency_weight,
                p_scale_factor=args.p_scale_factor,
                i_scale_factor=args.i_scale_factor,
                d_scale_factor=args.d_scale_factor
            )
            
            # Initialize with zeros for PID params in the tracking array (will be updated during run)
            all_pid_params.append((args.initial_p, args.initial_i, args.initial_d))
        
        # Create initial agents with only initial states
        agents = generate_agents(N_AGENTS, INITIAL_STATES, state_rewards, STATE_RUN_COSTS, STATE_SWITCH_COSTS, SWITCH_FREQUENCY_PARAM)
        agents_real_history = [[] for _ in range(NUM_ATTRIBUTES)]
        agents_declared_history = [[] for _ in range(NUM_ATTRIBUTES)]
        
        # Track states available at each epoch for proper diversity calculation
        states_available_at_epoch = []
        
        # Track PID parameters over time (for RL mode)
        pid_params_history = []
        
        # Track total rewards allocated over time
        total_rewards_history = []
        reward_per_state_history = []
        
        # Run the main experiment
        print("\n=== Starting main experiment ===")
        for epoch in range(epochs):
            # Add the new state at the specified epoch
            if epoch == NEW_STATE_EPOCH:
                # Reset the switch cost for the new state to normal level
                STATE_SWITCH_COSTS[NEW_STATE_NAME] = BASE_SWITCH_COST
                
                # Initialize the new state with normal reward based on new state count
                BASE_REWARDS_NEW = PER_NODE_REWARD_BASE * N_AGENTS / len(POSSIBLE_STATES)
                
                # Update agents to know about the new state
                for agent in agents:
                    # Update agent's possible states list
                    agent.possible_states = POSSIBLE_STATES.copy()
                    
                    # Set normal switch cost for the new state
                    agent.switch_cost[NEW_STATE_NAME] = BASE_SWITCH_COST
                
                # Preserve existing reward values for existing states
                preserved_rewards = {}
                for k in range(NUM_ATTRIBUTES):
                    preserved_rewards[k] = {state: state_rewards[k][state] for state in INITIAL_STATES}
                
                # Update reward controller with the new state
                if controller_type == "rl":
                    # For RL controller, we need to reinitialize with the new state list
                    # but keep the same PID parameters that have been learned
                    current_p = reward_controller.p_param
                    current_i = reward_controller.i_param
                    current_d = reward_controller.d_param
                    current_reward_scale = reward_controller.reward_scale
                    current_epsilon = reward_controller.epsilon
                    current_steps_done = reward_controller.steps_done  # Preserve steps_done counter
                    
                    # Store current PID errors for proper scaling
                    old_accumulated_errors = {state: reward_controller.accumulated_error[state] 
                                        for state in INITIAL_STATES if state in reward_controller.accumulated_error}
                    old_last_errors = {state: reward_controller.last_error[state] 
                                    for state in INITIAL_STATES if state in reward_controller.last_error}
                    
                    # Calculate old and new ideal shares
                    old_ideal_share = 1.0 / (len(INITIAL_STATES) - 1)  # -1 for NO_STATE
                    new_ideal_share = 1.0 / (len(POSSIBLE_STATES) - 1)  # -1 for NO_STATE
                    # Scale factor for error terms (how much the ideal distribution changed)
                    error_scale_factor = new_ideal_share / old_ideal_share
                    
                    # Create new controller
                    new_controller = RLPIDController(
                        POSSIBLE_STATES,
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
                        d_scale_factor=args.d_scale_factor,
                        reward_efficiency_weight=args.efficiency_weight
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
                    
                    # CRITICAL: Scale accumulated errors for existing states to prevent integral term disruption
                    for state in INITIAL_STATES:
                        if state in old_accumulated_errors:
                            # Scale the errors by the ratio of new/old ideal shares
                            new_controller.accumulated_error[state] = old_accumulated_errors[state] * error_scale_factor
                        if state in old_last_errors:
                            new_controller.last_error[state] = old_last_errors[state] * error_scale_factor
                            
                    # Add a reference to previous rewards for initialization
                    new_controller.prev_rewards = {state: preserved_rewards[0][state] for state in INITIAL_STATES 
                                                if state in preserved_rewards[0]}
                    
                    # Add a flag for transition mode
                    new_controller.in_transition = True
                    new_controller.transition_epochs = 1000  # Number of epochs for smooth transition
                    new_controller.transition_progress = 0  # Current progress (0-1)
                    
                    # Add prev_rewards for each attribute
                    new_controller.prev_rewards_by_attr = {}
                    for k in range(NUM_ATTRIBUTES):
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
                    # For PID controller, we need to update states and initialize new state error tracking
                    reward_controller.states = POSSIBLE_STATES.copy()
                    reward_controller.accumulated_error[NEW_STATE_NAME] = 0
                    reward_controller.last_error[NEW_STATE_NAME] = 0
                
                # Restore preserved reward values after controller initialization
                for k in range(NUM_ATTRIBUTES):
                    for state, value in preserved_rewards[k].items():
                        state_rewards[k][state] = value
                    
                    # Set new state's reward to zero initially
                    state_rewards[k][NEW_STATE_NAME] = 0
                
                # Print info about the new state
                print_new_state_added(NEW_STATE_NAME, epoch, BASE_REWARDS_NEW, len(POSSIBLE_STATES), 
                                    STATE_SWITCH_COSTS[NEW_STATE_NAME])
            
            # Keep track of which states were available at this epoch (for proper diversity calculation)
            current_states = INITIAL_STATES.copy() if epoch < NEW_STATE_EPOCH else POSSIBLE_STATES.copy()
            states_available_at_epoch.append(current_states)
            
            # Calculate current diversity for RL reward calculation
            current_diversity = None
            if controller_type == "rl" and epoch > 0:
                diversity_result = calculate_diversity(
                    [agents_declared_history[0]], [epoch-1], 
                    states_available_at_epoch[epoch-1], N_AGENTS)
                current_diversity = diversity_result[0][epoch-1][0]
                
                # Update agent counts in RL controller for accurate agent state calculations
                agents_in_states = {state: 0 for state in current_states}
                for agent in agents:
                    agent_state = agent.declared_state[0]  # Using attribute 0
                    agents_in_states[agent_state] = agents_in_states.get(agent_state, 0) + 1
                reward_controller.agent_counts = agents_in_states
                
            # Get ideal diversity for current set of states
            ideal_diversity = get_ideal_diversity(current_states)
                
            # Update rewards based on current state distribution using the appropriate controller
            if controller_type == "pid":
                # Use PID controller to update rewards
                state_rewards = reward_controller.update_rewards(agents, current_states, state_rewards)
                
                # Calculate total rewards allocated
                total_rewards = 0
                for state in current_states:
                    total_rewards += state_rewards[0][state]
                total_rewards_history.append(total_rewards)
                reward_per_state_history.append(state_rewards[0].copy())
                
                # Track PID parameters (constant for fixed PID controller)
                pid_params_history.append((
                    reward_controller.p_param,
                    reward_controller.i_param,
                    reward_controller.d_param
                ))
                
            elif controller_type == "rl":
                # Use RL-tuned PID controller to update rewards
                state_rewards = reward_controller.update_rewards(
                    agents, current_states, state_rewards,
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
                for state in current_states:
                    total_rewards += state_rewards[0][state]
                
                # Store the calculated total rewards
                total_rewards_history.append(total_rewards)
                reward_per_state_history.append(state_rewards[0].copy())
            
            # Distribute rewards among agents
            state_rewards_last_epoch = distribute_rewards(agents, current_states, state_rewards)
            
            # Record states of all agents
            for k in range(NUM_ATTRIBUTES):
                agents_real_history[k].append([agent.real_state[k] for agent in agents])
                agents_declared_history[k].append([agent.declared_state[k] for agent in agents])
            
            # Update agent decisions
            for agent in agents:
                agent.decision(state_rewards_last_epoch, malicious=False)
                
            # Print status at key epochs
            if epoch % 1000 == 0 or epoch == NEW_STATE_EPOCH or epoch == NEW_STATE_EPOCH + 1:
                # Count agents in each state for the first attribute
                state_counts = {state: 0 for state in current_states}
                for agent in agents:
                    state_counts[agent.declared_state[0]] += 1
                
                # Calculate and print current diversity
                current_epoch_diversity = None
                if epoch > 0:  # Skip first epoch
                    temp_diversity = calculate_diversity([agents_declared_history[0]], [epoch-1], 
                                                       states_available_at_epoch[epoch-1], N_AGENTS)
                    current_epoch_diversity = temp_diversity[0][epoch-1][0]
                
                # Print status for this epoch
                print_epoch_status(epoch, state_counts, state_rewards_last_epoch, 
                                 total_rewards_history[-1], controller_type, 
                                 reward_controller, current_epoch_diversity)
                
                # Specifically print new state count when it matters
                print_new_state_status(epoch, NEW_STATE_EPOCH, NEW_STATE_NAME, 
                                     state_counts, state_rewards_last_epoch)
        
        # Calculate final diversity with awareness of dynamic state set
        final_diversity = [{}] * NUM_ATTRIBUTES
        for k in range(NUM_ATTRIBUTES):
            final_diversity[k] = {}
            for epoch in range(epochs):
                # Use the correct set of possible states for each epoch
                epoch_diversity = calculate_diversity(
                    [agents_declared_history[k]], [epoch], 
                    states_available_at_epoch[epoch], N_AGENTS)
                final_diversity[k][epoch] = epoch_diversity[0][epoch]
        
        # Calculate the number of epochs in the last 25%
        last_25_percent = int(epochs * 0.25)
        
        # Calculate average diversity for each attribute for this experiment
        for k in range(NUM_ATTRIBUTES):
            # Extract the last 25% of diversity values - these are already floats!
            last_25_percent_diversity = [final_diversity[k][epoch][0] for epoch in range(epochs - last_25_percent, epochs)]
            
            # Calculate the average directly (no need to flatten)
            avg_diversity = sum(last_25_percent_diversity) / len(last_25_percent_diversity)
            
            # Store this experiment's average diversity
            average_diversity_by_experiment[k].append(avg_diversity)
            
            # Calculate ideal diversity for final set of states 
            ideal_diversity = get_ideal_diversity(POSSIBLE_STATES)
            ideal_diversity_all_experiments[k].append(ideal_diversity)
            
            # Calculate largest state metric - same issue here
            largest_state = get_largest_state(agents_declared_history, range(epochs), POSSIBLE_STATES, N_AGENTS)
            last_25_percent_largest_state = [largest_state[k][epoch][0] for epoch in range(epochs - last_25_percent, epochs)]
            
            # Calculate the average directly (no need to flatten)
            avg_largest_state = sum(last_25_percent_largest_state) / len(last_25_percent_largest_state)
            largest_state_all_experiments[k].append(avg_largest_state)
        
        # Get diversity values and ideal values for convergence analysis
        diversity_values = [final_diversity[0][epoch][0] for epoch in range(epochs)]
        ideal_before = get_ideal_diversity(INITIAL_STATES)
        ideal_after = get_ideal_diversity(POSSIBLE_STATES)
        
        # Analyze convergence in both phases using new enhanced metrics
        from measurement_functions import calculate_convergence_metrics, calculate_resilience_metrics
        
        # Calculate convergence metrics for both phases
        phase1_values = diversity_values[:NEW_STATE_EPOCH]
        phase1_convergence = calculate_convergence_metrics(phase1_values, ideal_before)
        
        phase2_values = diversity_values[NEW_STATE_EPOCH:]  
        phase2_convergence = calculate_convergence_metrics(phase2_values, ideal_after)
        
        # Calculate resilience metrics
        resilience_metrics = calculate_resilience_metrics(
            diversity_values, NEW_STATE_EPOCH, ideal_before, ideal_after)
        
        # Store convergence metrics for this experiment
        phase1_convergence_time = phase1_convergence['time_to_convergence']
        phase2_convergence_time = phase2_convergence['time_to_convergence']
        
        phase1_convergence_times.append(phase1_convergence_time if phase1_convergence_time is not None else epochs)
        phase2_convergence_times.append(phase2_convergence_time if phase2_convergence_time is not None else epochs)
        
        # Store adaptation quality and recovery time
        if "final_adaptation_quality" in resilience_metrics and resilience_metrics["final_adaptation_quality"] is not None:
            adaptation_metrics.append(resilience_metrics["final_adaptation_quality"])
        else:
            adaptation_metrics.append(0.0)
            
        if "recovery_time_epochs" in resilience_metrics and resilience_metrics["recovery_time_epochs"] is not None:
            recovery_times.append(resilience_metrics["recovery_time_epochs"])
        else:
            recovery_times.append(epochs)  # Use max epochs if no recovery
    
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
                pid_evolution = analyze_pid_parameter_evolution(pid_params_history, diversity_values, NEW_STATE_EPOCH)
            
            # Compile all metrics
            performance_metrics = compile_system_performance_metrics(
                diversity_values, total_rewards_history, pid_params_history, 
                NEW_STATE_EPOCH, ideal_before, ideal_after
            )
            
            # Print detailed metrics table
            from plotting_functions import create_metrics_summary_table
            
            # For RL mode, create a dummy PID metrics to print
            if controller_type == "rl":
                # Create metrics for an ideal PID controller for comparison
                # (this is just for display purposes)
                pid_metrics = {
                    'phase1_convergence': {'time_to_convergence': None, 'convergence_rate': None, 
                                         'steady_state_error': None, 'final_diversity_quality': None},
                    'phase2_convergence': {'time_to_convergence': None, 'convergence_rate': None, 
                                         'steady_state_error': None, 'final_diversity_quality': None},
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
                print(f"  Time to re-convergence (Phase 2): {phase2_convergence_time if phase2_convergence_time is not None else 'N/A'}")
                
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
            plot_diversity_over_time(final_diversity, epochs_list, NUM_ATTRIBUTES, NEW_STATE_EPOCH, 
                                  NEW_STATE_NAME, INITIAL_STATES, POSSIBLE_STATES, controller_type,
                                  REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM,
                                  pid_params_history)
            
            # Plot total rewards over time
            from plotting_functions import plot_total_rewards
            plot_total_rewards(epochs_list, total_rewards_history, NEW_STATE_EPOCH, NEW_STATE_NAME, controller_type,
                            REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM,
                            pid_params_history)
            
            # Plot rewards per state over time
            # Plot rewards per state over time
            from plotting_functions import plot_rewards_per_state
            plot_rewards_per_state(epochs_list, reward_per_state_history, POSSIBLE_STATES, NEW_STATE_EPOCH, 
                                NEW_STATE_NAME, controller_type, REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, 
                                REWARDS_DERIVATIVE_PARAM, pid_params_history)
            
            # For RL, also plot PID parameter evolution
            if controller_type == "rl":
                from plotting_functions import plot_pid_parameters
                plot_pid_parameters(epochs_list, pid_params_history, NEW_STATE_EPOCH, NEW_STATE_NAME)
            
            # Print final results and analysis using new metrics
            from plotting_functions import print_single_experiment_results
            print_single_experiment_results(POSSIBLE_STATES, state_rewards_last_epoch, BASE_REWARDS, 
                                          controller_type, total_rewards_history, pid_params_history,
                                          reward_controller)
            
            # Print enhanced convergence analysis
            from plotting_functions import print_convergence_analysis
            print_convergence_analysis(diversity_values, NEW_STATE_EPOCH, INITIAL_STATES, POSSIBLE_STATES, 
                                     {"phase1": phase1_convergence, "phase2": phase2_convergence})

            # Print enhanced adaptability metrics
            from plotting_functions import print_adaptability_metrics
            print_adaptability_metrics(resilience_metrics)
            
            # Print enhanced reward allocation metrics
            from plotting_functions import print_reward_allocation_metrics
            print_reward_allocation_metrics(total_rewards_history, NEW_STATE_EPOCH)
            
            # Plot agent distribution
            from plotting_functions import plot_agent_distribution
            for k in range(NUM_ATTRIBUTES):
                plot_agent_distribution(agents_declared_history, epochs_list, POSSIBLE_STATES, 
                                      NEW_STATE_EPOCH, NEW_STATE_NAME, controller_type, k)
                
    # For multiple experiments, create summary plots with experiment number as x-axis
    if num_experiments > 1:
        from plotting_functions import print_multi_experiment_header
        print_multi_experiment_header(num_experiments, controller_type)
        print("="*60)
        
        # Print average metrics across all experiments
        for k in range(NUM_ATTRIBUTES):
            avg_diversity = sum(average_diversity_by_experiment[k]) / num_experiments
            avg_largest_state = sum(largest_state_all_experiments[k]) / num_experiments
            
            print(f"\nAttribute {k} Results:")
            print(f"  • Average diversity (last 25% of epochs): {avg_diversity:.4f} bits")
            print(f"  • Average largest state share: {avg_largest_state:.2%}")
            
            ideal_div = sum(ideal_diversity_all_experiments[k]) / num_experiments
            print(f"  • Average ideal diversity: {ideal_div:.4f} bits")
            print(f"  • Average diversity achieved: {(avg_diversity/ideal_div)*100:.1f}% of ideal")
        
        # Print adaptation metrics
        avg_adaptation = sum(adaptation_metrics) / num_experiments
        avg_recovery = sum(recovery_times) / num_experiments
        avg_p1_conv = sum(phase1_convergence_times) / num_experiments
        avg_p2_conv = sum(phase2_convergence_times) / num_experiments
        
        print("\nSystem Adaptation Metrics:")
        print(f"  • Average adaptation quality: {avg_adaptation*100:.1f}% of ideal")
        print(f"  • Average recovery time: {avg_recovery:.1f} epochs")
        print(f"  • Average Phase 1 convergence time: {avg_p1_conv:.1f} epochs")
        print(f"  • Average Phase 2 convergence time: {avg_p2_conv:.1f} epochs")
        
        # Create and display plots
        print("\nGenerating summary plots...")
        
        # Plot average diversity across experiments
        from plotting_functions import plot_average_diversity
        plot_average_diversity(x_axis, average_diversity_by_experiment, NUM_ATTRIBUTES, controller_type, ideal_div)
        
        # Plot adaptation quality
        from plotting_functions import plot_adaptation_quality
        plot_adaptation_quality(x_axis, adaptation_metrics, controller_type)
        
        # Plot recovery times
        from plotting_functions import plot_recovery_times
        plot_recovery_times(x_axis, recovery_times)
        
        # Plot convergence times
        from plotting_functions import plot_convergence_times
        plot_convergence_times(x_axis, phase1_convergence_times, phase2_convergence_times)
        
        # Plot PID parameters
        from plotting_functions import plot_pid_parameters_across_experiments
        plot_pid_parameters_across_experiments(x_axis, all_pid_params)
        
        # Plot largest state share
        from plotting_functions import plot_largest_state_share
        plot_largest_state_share(x_axis, largest_state_all_experiments, NUM_ATTRIBUTES)


if __name__ == "__main__":
    main()