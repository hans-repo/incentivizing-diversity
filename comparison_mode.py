import random
import numpy as np
import torch
from agent import generate_agents, distribute_rewards
from measurement_functions import (calculate_diversity, get_ideal_diversity, 
                                    compile_system_performance_metrics,
                                    calculate_convergence_metrics, calculate_resilience_metrics)
from ziegler_nichols_tuning import ziegler_nichols_tuning
from reinforcement_learning import PIDController, RLPIDController
from reinforcement_learning_nopid import RLDirectRewardController
from plotting_functions import (print_experiment_header, print_pid_parameters, 
                                print_rl_parameters, print_new_version_added, 
                                print_epoch_status, print_new_version_status)
from reinforcement_learning import RLPIDController
from plotting_functions import compare_and_visualize_all_controllers
from plotting_functions import print_new_version_added
from measurement_functions import calculate_diversity
from plotting_functions import print_epoch_status, print_new_version_status

    
def run_comparison_mode(args):
    """
    Run all specified controllers multiple times and compare their performance.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    print("\n" + "="*80)
    print(f"{'':^10}RUNNING COMPARISON MODE: {' vs '.join(args.compare_modes).upper()}{'':^10}")
    print(f"{'':^10}Number of experiments per controller: {args.num_experiments}{'':^10}")
    if hasattr(args, 'no_new_version') and args.no_new_version:
        print(f"{'':^10}Mode: NO NEW version ADDITION{'':^10}")
    else:
        print(f"{'':^10}Mode: WITH NEW version ADDITION{'':^10}")
    print("="*80)
    
    # Save original controller type
    original_controller = args.controller
    
    # Store results for all controllers
    comparison_results = {}
    
    # Determine which controllers to run based on compare_modes argument
    controllers_to_run = args.compare_modes if hasattr(args, 'compare_modes') else ['pid', 'rl', 'rlnopid']
    
    # Run each controller experiment multiple times
    for controller in controllers_to_run:
        print("\n\n" + "="*60)
        print(f"RUNNING {args.num_experiments} EXPERIMENTS WITH {controller.upper()} CONTROLLER")
        print("="*60)
        
        args.controller = controller
        
        # Run multiple experiments for this controller
        controller_all_results = []
        controller_metrics = {
            'average_diversity_by_experiment': [],
            'adaptation_metrics': [],
            'recovery_times': [],
            'phase1_convergence_times': [],
            'phase2_convergence_times': [],
            'final_diversity_values': [],
            'ideal_diversity': None,
            'pid_params_final': []
        }
        
        # Store tuned PID parameters to reuse across experiments
        tuned_pid_params = None
        
        for exp_num in range(args.num_experiments):
            print(f"\n--- Running {controller.upper()} Experiment {exp_num + 1}/{args.num_experiments} ---")
            
            # Run single experiment
            single_result = run_single_controller_experiment(args, exp_num, tuned_pid_params)
            controller_all_results.append(single_result)
            
            # # Store tuned PID params from first experiment if using PID controller
            if controller == 'pid' and exp_num == 0 and args.tune_pid and args.tune_once:
                tuned_pid_params = single_result['pid_params']
                        
            # Extract metrics from this experiment
            diversity_values = [single_result['final_diversity'][0][epoch][0] 
                              for epoch in range(single_result['epochs'])]
            
            # Calculate average diversity for last 25% of epochs
            last_25_percent = int(single_result['epochs'] * 0.25)
            last_25_percent_diversity = diversity_values[-last_25_percent:]
            avg_diversity = sum(last_25_percent_diversity) / len(last_25_percent_diversity)
            controller_metrics['average_diversity_by_experiment'].append(avg_diversity)
            
            # Store ideal diversity (same for all experiments)
            if controller_metrics['ideal_diversity'] is None:
                controller_metrics['ideal_diversity'] = get_ideal_diversity(single_result['POSSIBLE_versionS'])
            
            # Calculate convergence and adaptation metrics
            ideal_before = get_ideal_diversity(single_result['INITIAL_versionS'])
            ideal_after = get_ideal_diversity(single_result['POSSIBLE_versionS'])
            NEW_version_EPOCH = single_result['NEW_version_EPOCH']
            
            # Handle phase convergence analysis based on whether new version was added
            if NEW_version_EPOCH is None:
                # Single-phase analysis (no new version)
                phase1_values = diversity_values
                phase2_values = []
                
                phase1_convergence = calculate_convergence_metrics(phase1_values, ideal_before)
                phase2_convergence = {'time_to_convergence': None}
                
                # No resilience metrics for single-phase
                resilience_metrics = calculate_resilience_metrics(
                    diversity_values, NEW_version_EPOCH, ideal_before, ideal_after)
                
            else:
                # Dual-phase analysis (with new version)
                phase1_values = diversity_values[:NEW_version_EPOCH]
                phase2_values = diversity_values[NEW_version_EPOCH:]
                
                phase1_convergence = calculate_convergence_metrics(phase1_values, ideal_before)
                phase2_convergence = calculate_convergence_metrics(phase2_values, ideal_after)
                
                # Resilience metrics
                resilience_metrics = calculate_resilience_metrics(
                    diversity_values, NEW_version_EPOCH, ideal_before, ideal_after)
            
            # Store metrics
            phase1_time = phase1_convergence['time_to_convergence']
            phase2_time = phase2_convergence['time_to_convergence']
            
            controller_metrics['phase1_convergence_times'].append(
                phase1_time if phase1_time is not None else single_result['epochs'])
            controller_metrics['phase2_convergence_times'].append(
                phase2_time if phase2_time is not None else single_result['epochs'])
            
            # Adaptation quality
            adaptation_quality = resilience_metrics.get("final_adaptation_quality", 1.0 if NEW_version_EPOCH is None else 0.0)
            controller_metrics['adaptation_metrics'].append(
                adaptation_quality if adaptation_quality is not None else (1.0 if NEW_version_EPOCH is None else 0.0))
            
            # Recovery time
            recovery_time = resilience_metrics.get("recovery_time_epochs", 0 if NEW_version_EPOCH is None else single_result['epochs'])
            controller_metrics['recovery_times'].append(
                recovery_time if recovery_time is not None else (0 if NEW_version_EPOCH is None else single_result['epochs']))
            
            # Final diversity value
            controller_metrics['final_diversity_values'].append(diversity_values[-1])
            
            # Final PID parameters
            controller_metrics['pid_params_final'].append(single_result['final_pid_params'])
        
        # Store results for this controller
        comparison_results[controller] = {
            'all_experiments': controller_all_results,
            'metrics': controller_metrics,
            'controller_type': controller,
            # Include a representative experiment for plotting (first one)
            'representative_experiment': controller_all_results[0]
        }
        
        # Print summary for this controller
        print(f"\n{controller.upper()} Controller Summary ({args.num_experiments} experiments):")
        print(f"  • Average diversity: {np.mean(controller_metrics['average_diversity_by_experiment']):.4f} ± {np.std(controller_metrics['average_diversity_by_experiment']):.4f}")
        print(f"  • Average adaptation quality: {np.mean(controller_metrics['adaptation_metrics'])*100:.1f}% ± {np.std(controller_metrics['adaptation_metrics'])*100:.1f}%")
        print(f"  • Average recovery time: {np.mean(controller_metrics['recovery_times']):.1f} ± {np.std(controller_metrics['recovery_times']):.1f} epochs")
        print(f"  • Average Phase 1 convergence: {np.mean(controller_metrics['phase1_convergence_times']):.1f} ± {np.std(controller_metrics['phase1_convergence_times']):.1f} epochs")
        
        # Only print Phase 2 convergence if there was actually a phase 2
        if not (hasattr(args, 'no_new_version') and args.no_new_version):
            print(f"  • Average Phase 2 convergence: {np.mean(controller_metrics['phase2_convergence_times']):.1f} ± {np.std(controller_metrics['phase2_convergence_times']):.1f} epochs")
    
    # Restore original controller type
    args.controller = original_controller
    
    # Compare and visualize the results
    compare_and_visualize_all_controllers(comparison_results, args)
    
    return comparison_results

def run_single_controller_experiment(args, experiment_number=0, tuned_pid_params=None):
    """
    Run a single experiment with PID, RL, or RL-no-PID controller.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    experiment_number : int
        Current experiment number (for seeding/logging)
    tuned_pid_params : tuple or None
        Pre-tuned PID parameters to reuse (P, I, D)
        
    Returns:
    --------
    dict
        Dictionary containing the experiment results
    """

    # Extract experiment parameters from args
    controller_type = args.controller
    epochs = args.epochs
    N_AGENTS = args.n_agents
    NUM_ATTRIBUTES = 1  # Simplified for clarity
    
    # Set random seeds for reproducibility across experiments
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)
    
    # Use the same version configuration as main experiments
    # INITIAL_versionS = ['NO_version', 'version_A', 'version_B', 'version_C', 'version_D']
    # INITIAL_versionS = ['NO_version', 'version_A', 'version_B', 'version_C', 'version_D', 'version_E', 'version_F', 'version_G']
    INITIAL_versionS = ['NO_version', 'version_A', 'version_B', 'version_C', 'version_D', 'version_E', 'version_F', 'version_G', 'version_H', 'version_I', 'version_J', 'version_K', 'version_L']
    NEW_version_NAME = 'version_NEW'
    
    # Check if new version should be added
    if hasattr(args, 'add_new_version') and args.add_new_version:
        NEW_version_EPOCH = round(0.05 * epochs)
        POSSIBLE_versionS = INITIAL_versionS.copy()
        POSSIBLE_versionS.append(NEW_version_NAME)
        print(f"  Running experiment WITH new version addition at epoch {NEW_version_EPOCH}")
    else:
        NEW_version_EPOCH = None
        POSSIBLE_versionS = INITIAL_versionS.copy()
        print(f"  Running experiment WITHOUT new version addition")
    
    NUM_versionS = len(POSSIBLE_versionS)

    # PID parameters
    REWARDS_ADAPTIVE_PARAM = args.p_param
    REWARDS_INTEGRAL_PARAM = args.i_param
    REWARDS_DERIVATIVE_PARAM = args.d_param
    
    # Cost configuration - add some variation per experiment
    BASE_RUN_COST = 100 + 0
    RUN_COST_CEILING = BASE_RUN_COST + round(2*BASE_RUN_COST)
    # RUN_COST_CEILING = BASE_RUN_COST + round(experiment_number*20*BASE_RUN_COST)
    BASE_SWITCH_COST = BASE_RUN_COST

    # Generate evenly spaced values for initial versions
    version_run_costs = np.linspace(BASE_RUN_COST, RUN_COST_CEILING, NUM_versionS)
    version_RUN_COSTS = {
        version: round(value, 4)
        for version, value in zip(POSSIBLE_versionS, version_run_costs)
    }
    
    # Add run cost for the new version if it will be added later
    if NEW_version_EPOCH is not None:
        version_RUN_COSTS[NEW_version_NAME] = round(BASE_RUN_COST + RUN_COST_CEILING, 4)
    
    version_SWITCH_COSTS = {version: BASE_SWITCH_COST for version in POSSIBLE_versionS}
    
    # Set initial switch cost for new version if it will be added
    if NEW_version_EPOCH is not None:
        # Set a high initial switch cost for the new version to ensure zero initial nodes
        version_SWITCH_COSTS[NEW_version_NAME] = BASE_SWITCH_COST * 100
    
    # SWITCH_FREQUENCY_PARAM = 2
    SWITCH_FREQUENCY_PARAM = 0.4*experiment_number  # Add variation per experiment
    
    # Initialize version rewards and tracking
    version_rewards = [{version: 0 for version in POSSIBLE_versionS} for _ in range(NUM_ATTRIBUTES)]
    
    # Pre-add the new version to rewards tracking if it will be used, but make it unattractive
    if NEW_version_EPOCH is not None:
        for k in range(NUM_ATTRIBUTES):
            version_rewards[k][NEW_version_NAME] = 0  # No reward initially
    
    # Initialize the appropriate controller based on the controller_type
    if controller_type == "pid":
        # Use tuned parameters if available, otherwise tune or use defaults
        if tuned_pid_params is not None:
            REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM = tuned_pid_params
            print(f"\n=== Reusing PID parameters from first experiment ===")
        elif args.tune_pid:
            print(f"\n=== Running Ziegler-Nichols tuning for experiment {experiment_number + 1} ===")
            REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM = ziegler_nichols_tuning(
                None, INITIAL_versionS, epochs=epochs, n_agents=N_AGENTS)
        
        print_pid_parameters(REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM)
        
        reward_controller = PIDController(
            POSSIBLE_versionS,
            REWARDS_ADAPTIVE_PARAM,
            REWARDS_INTEGRAL_PARAM,
            REWARDS_DERIVATIVE_PARAM
        )
    
    elif controller_type == "rl":
        print_rl_parameters(args.epsilon, args.epsilon_decay, args.learning_rate, args.action_scale, 
                        args.initial_p, args.initial_i, args.initial_d,
                          args.p_scale_factor, args.i_scale_factor, args.d_scale_factor)
        
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
        
    elif controller_type == "rlnopid":
        print("\n=== Using RL Direct Reward Control (no PID) ===")
        print(f"  • Epsilon: {args.epsilon}")
        print(f"  • Epsilon decay: {args.epsilon_decay}")
        print(f"  • Learning rate: {args.learning_rate}")
        print(f"  • Action scale: {args.action_scale}")

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
    
    # Create initial agents - use only INITIAL_versionS for agent generation
    agents = generate_agents(N_AGENTS, INITIAL_versionS, version_rewards, version_RUN_COSTS, version_SWITCH_COSTS, SWITCH_FREQUENCY_PARAM)
    agents_real_history = [[] for _ in range(NUM_ATTRIBUTES)]
    agents_declared_history = [[] for _ in range(NUM_ATTRIBUTES)]
    
    # Track versions available at each epoch
    versions_available_at_epoch = []
    
    # Track metrics over time
    pid_params_history = []
    total_rewards_history = []
    reward_per_version_history = []
    
    # Run the main experiment loop
    print(f"=== Starting main experiment {experiment_number + 1} ===")
    for epoch in range(epochs):
        # Handle new version addition only if enabled
        if NEW_version_EPOCH is not None and epoch == NEW_version_EPOCH:
            # Use existing helper function for version addition logic
            _handle_new_version_addition(epoch, NEW_version_NAME, NEW_version_EPOCH, 
                                     POSSIBLE_versionS, agents, 
                                     version_rewards, version_SWITCH_COSTS, BASE_SWITCH_COST,
                                     reward_controller, controller_type, args, 
                                     INITIAL_versionS, N_AGENTS, NUM_ATTRIBUTES)
        
        # Keep track of available versions
        if NEW_version_EPOCH is None:
            current_versions = INITIAL_versionS.copy()
        else:
            current_versions = INITIAL_versionS.copy() if epoch < NEW_version_EPOCH else POSSIBLE_versionS.copy()
        versions_available_at_epoch.append(current_versions)
        
        # Calculate current diversity for RL controllers
        current_diversity = None
        if (controller_type == "rl" or controller_type == "rlnopid") and epoch > 0:
            diversity_result = calculate_diversity(
                [agents_declared_history[0]], [epoch-1], 
                versions_available_at_epoch[epoch-1], N_AGENTS)
            current_diversity = diversity_result[0][epoch-1][0]
            
            # Update agent counts in RL controller
            agents_in_versions = {version: 0 for version in current_versions}
            for agent in agents:
                agent_version = agent.declared_version[0]
                agents_in_versions[agent_version] = agents_in_versions.get(agent_version, 0) + 1
            reward_controller.agent_counts = agents_in_versions
        
        ideal_diversity = get_ideal_diversity(current_versions)
        
        # Update rewards using appropriate controller
        total_rewards = _update_rewards_for_controller(
            controller_type, reward_controller, agents, current_versions, 
            version_rewards, current_diversity, ideal_diversity, epoch, args)
        
        total_rewards_history.append(total_rewards)
        reward_per_version_history.append(version_rewards[0].copy())
        
        # Track PID parameters
        if controller_type == "pid":
            pid_params_history.append((
                reward_controller.p_param,
                reward_controller.i_param,
                reward_controller.d_param
            ))
        elif controller_type == "rl":
            pid_params_history.append((
                reward_controller.p_param,
                reward_controller.i_param,
                reward_controller.d_param
            ))
        elif controller_type == "rlnopid":
            pid_params_history.append((0, 0, 0))  # No PID parameters
        
        # Distribute rewards and update agents
        version_rewards_last_epoch = distribute_rewards(agents, current_versions, version_rewards)
        
        # Record agent versions
        for k in range(NUM_ATTRIBUTES):
            agents_real_history[k].append([agent.real_version[k] for agent in agents])
            agents_declared_history[k].append([agent.declared_version[k] for agent in agents])
        
        # Update agent decisions
        for agent in agents:
            agent.decision(version_rewards_last_epoch, malicious=False)
        
        # Print status at key epochs (only for first experiment to avoid spam)
        if experiment_number == 0:
            # Print status for key epochs and new version addition
            should_print = (epoch % 1000 == 0 or 
                          (NEW_version_EPOCH is not None and (epoch == NEW_version_EPOCH or epoch == NEW_version_EPOCH + 1)))
            if should_print:
                _print_epoch_status(epoch, current_versions, agents, version_rewards_last_epoch,
                                  total_rewards, controller_type, reward_controller, 
                                  agents_declared_history, versions_available_at_epoch, N_AGENTS,
                                  NEW_version_EPOCH, NEW_version_NAME)
    
    # Calculate final diversity
    final_diversity = [{}] * NUM_ATTRIBUTES
    for k in range(NUM_ATTRIBUTES):
        final_diversity[k] = {}
        for epoch in range(epochs):
            epoch_diversity = calculate_diversity(
                [agents_declared_history[k]], [epoch], 
                versions_available_at_epoch[epoch], N_AGENTS)
            final_diversity[k][epoch] = epoch_diversity[0][epoch]
    
    # Extract diversity values for metrics
    diversity_values = [final_diversity[0][epoch][0] for epoch in range(epochs)]
    ideal_before = get_ideal_diversity(INITIAL_versionS)
    ideal_after = get_ideal_diversity(POSSIBLE_versionS)
    
    # Compile performance metrics
    performance_metrics = compile_system_performance_metrics(
        diversity_values, total_rewards_history, pid_params_history,
        NEW_version_EPOCH, ideal_before, ideal_after
    )
    
    # Return results
    return {
        'controller_type': controller_type,
        'final_diversity': final_diversity,
        'agents_declared_history': agents_declared_history,
        'total_rewards_history': total_rewards_history,
        'reward_per_version_history': reward_per_version_history,
        'pid_params_history': pid_params_history,
        'pid_params': (REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM),
        'final_pid_params': pid_params_history[-1] if pid_params_history else (REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM),
        'NEW_version_EPOCH': NEW_version_EPOCH,
        'NEW_version_NAME': NEW_version_NAME if NEW_version_EPOCH is not None else None,
        'INITIAL_versionS': INITIAL_versionS,
        'POSSIBLE_versionS': POSSIBLE_versionS,
        'performance_metrics': performance_metrics,
        'epochs': epochs,
        'N_AGENTS': N_AGENTS,
        'experiment_number': experiment_number
    }

def _handle_new_version_addition(epoch, NEW_version_NAME, NEW_version_EPOCH, POSSIBLE_versionS, 
                              agents, version_rewards, version_SWITCH_COSTS, 
                              BASE_SWITCH_COST, reward_controller, controller_type, args, 
                              INITIAL_versionS, N_AGENTS, NUM_ATTRIBUTES):
    """Helper function to handle new version addition logic"""

    # Reset switch cost for new version
    version_SWITCH_COSTS[NEW_version_NAME] = BASE_SWITCH_COST
    

    # Update agents
    for agent in agents:
        agent.possible_versions = POSSIBLE_versionS.copy()
        agent.switch_cost[NEW_version_NAME] = BASE_SWITCH_COST
    
    # Preserve existing rewards
    preserved_rewards = {}
    for k in range(NUM_ATTRIBUTES):
        preserved_rewards[k] = {version: version_rewards[k][version] for version in INITIAL_versionS}
    
    # Update controller for new version
    if controller_type == "rl":
        _update_rl_controller_for_new_version(reward_controller, POSSIBLE_versionS, N_AGENTS, 
                                           args, INITIAL_versionS, preserved_rewards)
    elif controller_type == "pid":
        reward_controller.versions = POSSIBLE_versionS.copy()
        reward_controller.accumulated_error[NEW_version_NAME] = 0
        reward_controller.last_error[NEW_version_NAME] = 0
    # RL-no-PID handles version changes internally
    
    # Restore preserved rewards
    for k in range(NUM_ATTRIBUTES):
        for version, value in preserved_rewards[k].items():
            version_rewards[k][version] = value
        version_rewards[k][NEW_version_NAME] = 0
    
    print_new_version_added(NEW_version_NAME, epoch, len(POSSIBLE_versionS), 
                         version_SWITCH_COSTS[NEW_version_NAME])

def _update_rl_controller_for_new_version(reward_controller, POSSIBLE_versionS, N_AGENTS, 
                                       args, INITIAL_versionS, preserved_rewards):
    """Helper function to update RL controller when new version is added"""

    # Store current version
    current_p = reward_controller.p_param
    current_i = reward_controller.i_param
    current_d = reward_controller.d_param
    current_reward_scale = reward_controller.reward_scale
    current_epsilon = reward_controller.epsilon
    current_steps_done = reward_controller.steps_done
    
    # Store accumulated errors
    old_accumulated_errors = {version: reward_controller.accumulated_error[version] 
                            for version in INITIAL_versionS if version in reward_controller.accumulated_error}
    old_last_errors = {version: reward_controller.last_error[version] 
                     for version in INITIAL_versionS if version in reward_controller.last_error}
    
    # Calculate error scaling
    old_ideal_share = 1.0 / (len(INITIAL_versionS) - 1)
    new_ideal_share = 1.0 / (len(POSSIBLE_versionS) - 1)
    error_scale_factor = new_ideal_share / old_ideal_share
    
    # Create new controller
    new_controller = RLPIDController(
        POSSIBLE_versionS, N_AGENTS,
        initial_p=current_p, initial_i=current_i, initial_d=current_d,
        epsilon=current_epsilon, epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate, action_scale=args.action_scale,
        update_frequency=5, batch_size=16,
        p_scale_factor=args.p_scale_factor,
        i_scale_factor=args.i_scale_factor,
        d_scale_factor=args.d_scale_factor
    )
    
    # Transfer version
    new_controller.reward_scale = current_reward_scale
    new_controller.steps_done = current_steps_done
    new_controller.best_pid_params = reward_controller.best_pid_params
    new_controller.best_diversity_ratio = reward_controller.best_diversity_ratio
    new_controller.best_reward_scale = reward_controller.best_reward_scale
    
    if hasattr(reward_controller, 'replay_buffer'):
        new_controller.replay_buffer = reward_controller.replay_buffer
    
    # Scale accumulated errors
    for version in INITIAL_versionS:
        if version in old_accumulated_errors:
            new_controller.accumulated_error[version] = old_accumulated_errors[version] * error_scale_factor
        if version in old_last_errors:
            new_controller.last_error[version] = old_last_errors[version] * error_scale_factor
    
    # Set transition mode
    new_controller.prev_rewards = {version: preserved_rewards[0][version] for version in INITIAL_versionS 
                                 if version in preserved_rewards[0]}
    new_controller.in_transition = True
    new_controller.transition_epochs = 1000
    new_controller.transition_progress = 0
    
    # Update the original controller reference
    reward_controller.__dict__.update(new_controller.__dict__)

def _update_rewards_for_controller(controller_type, reward_controller, agents, current_versions, 
                                 version_rewards, current_diversity, ideal_diversity, epoch, args):
    """Helper function to update rewards based on controller type"""
    if controller_type == "pid":
        version_rewards = reward_controller.update_rewards(agents, current_versions, version_rewards)
        total_rewards = sum(version_rewards[0][version] for version in current_versions)
        
    elif controller_type == "rl":
        try:
            version_rewards = reward_controller.update_rewards(
                agents, current_versions, version_rewards,
                current_diversity=current_diversity, 
                ideal_diversity=ideal_diversity,
                epoch=epoch
            )
            total_rewards = sum(version_rewards[0][version] for version in current_versions)
        except Exception as e:
            print(f"Error in RL controller: {e}")
            total_rewards = 0
            
    elif controller_type == "rlnopid":
        try:
            version_rewards = reward_controller.update_rewards(
                agents, current_versions, version_rewards,
                current_diversity=current_diversity, 
                ideal_diversity=ideal_diversity,
                epoch=epoch
            )
            total_rewards = sum(version_rewards[0][version] for version in current_versions)
        except Exception as e:
            print(f"Error in RL-no-PID controller: {e}")
            total_rewards = 0
    
    return total_rewards

def _print_epoch_status(epoch, current_versions, agents, version_rewards_last_epoch,
                       total_rewards, controller_type, reward_controller, 
                       agents_declared_history, versions_available_at_epoch, N_AGENTS,
                       NEW_version_EPOCH, NEW_version_NAME):
    """Helper function to print epoch status"""

    # Count agents in each version
    version_counts = {version: 0 for version in current_versions}
    for agent in agents:
        version_counts[agent.declared_version[0]] += 1
    
    # Calculate current diversity
    current_epoch_diversity = None
    if epoch > 0:
        temp_diversity = calculate_diversity([agents_declared_history[0]], [epoch-1], 
                                           versions_available_at_epoch[epoch-1], N_AGENTS)
        current_epoch_diversity = temp_diversity[0][epoch-1][0]
    
    print_epoch_status(epoch, version_counts, version_rewards_last_epoch, 
                     total_rewards, controller_type, 
                     reward_controller, current_epoch_diversity)
    
    # Only print new version status if new version was actually added
    if (NEW_version_EPOCH is not None and NEW_version_NAME is not None and 
        epoch >= NEW_version_EPOCH and epoch <= NEW_version_EPOCH + 100 and epoch % 10 == 0):
        print_new_version_status(epoch, NEW_version_EPOCH, NEW_version_NAME, 
                             version_counts, version_rewards_last_epoch)