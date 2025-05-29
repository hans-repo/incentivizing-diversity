"""
Implementation of comparison mode for multiple_experiments.py
"""

def run_comparison_mode(args):
    """
    Run both PID and RL controllers with the same parameters and compare their performance.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    print("\n" + "="*80)
    print(f"{'':^10}RUNNING COMPARISON MODE: PID vs RL-TUNED PID{'':^10}")
    print("="*80)
    
    # Save original controller type
    original_controller = args.controller
    
    # Store results for both controllers
    comparison_results = {
        'pid': {},
        'rl': {}
    }
    
    # Run PID controller experiment first
    print("\n\n" + "="*60)
    print(f"RUNNING FIXED PID CONTROLLER EXPERIMENT")
    print("="*60)
    
    args.controller = 'pid'
    pid_results = run_experiment(args)
    comparison_results['pid'] = pid_results
    
    # Then run RL controller experiment
    print("\n\n" + "="*60)
    print(f"RUNNING RL-TUNED PID CONTROLLER EXPERIMENT")
    print("="*60)
    
    args.controller = 'rl'
    rl_results = run_experiment(args)
    comparison_results['rl'] = rl_results
    
    # Restore original controller type
    args.controller = original_controller
    
    # Compare and visualize the results
    compare_and_visualize(comparison_results, args)
    
    return comparison_results

"""
Continuation of comparison mode implementation for multiple_experiments.py
"""

def run_experiment(args):
    """
    Run a single experiment with either PID or RL controller.
    This is a modified version of the main function that returns the results.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    dict
        Dictionary containing the experiment results
    """
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from agent import generate_agents, distribute_rewards
    from measurement_functions import calculate_diversity, get_ideal_diversity, get_largest_state
    from ziegler_nichols_tuning import ziegler_nichols_tuning
    from reinforcement_learning import PIDController, RLPIDController
    from plotting_functions import print_experiment_header, print_pid_parameters, print_rl_parameters
    from plotting_functions import print_new_state_added, print_epoch_status, print_new_state_status
    
    # Extract experiment parameters from args
    controller_type = args.controller
    epochs = args.epochs
    N_AGENTS = args.n_agents
    NUM_ATTRIBUTES = 1  # Simplified for clarity
    
    # Define the possible states an agent can be in
    INITIAL_STATES = ['NO_STATE', 'State_A', 'State_B', 'State_C', 'State_D']
    NEW_STATE_NAME = 'State_NEW'  # The new state to be added later
    NEW_STATE_EPOCH = round(0.5 * epochs)  # Epoch at which the new state is added
    
    POSSIBLE_STATES = INITIAL_STATES.copy()  # Start with initial states
    POSSIBLE_STATES.append(NEW_STATE_NAME)
    NUM_STATES = len(POSSIBLE_STATES)
    PER_NODE_REWARD_BASE = 1000
    PER_NODE_REWARD = PER_NODE_REWARD_BASE  # Per node reward per epoch in an evenly distributed system
    
    # Define fixed rewards for each state
    BASE_REWARDS = PER_NODE_REWARD_BASE * N_AGENTS / (NUM_STATES-1)  # -1 for NO_STATE
    
    # PID parameters
    REWARDS_ADAPTIVE_PARAM = args.p_param
    REWARDS_INTEGRAL_PARAM = args.i_param
    REWARDS_DERIVATIVE_PARAM = args.d_param
    
    BASE_RUN_COST = 100
    RUN_COST_CEILING = BASE_RUN_COST * 3
    BASE_SWITCH_COST = BASE_RUN_COST

    # Generate evenly spaced values for initial states
    state_run_costs = np.linspace(BASE_RUN_COST, RUN_COST_CEILING, NUM_STATES)
    STATE_RUN_COSTS = {
        state: round(value, 4)
        for state, value in zip(POSSIBLE_STATES, state_run_costs)
    }
    
    # Add run cost for the new state that will be added later
    STATE_RUN_COSTS[NEW_STATE_NAME] = round(BASE_RUN_COST + 1.5*BASE_RUN_COST, 4)
    
    STATE_SWITCH_COSTS = {state: BASE_SWITCH_COST for state in POSSIBLE_STATES}
    # Set a high initial switch cost for the new state to ensure zero initial nodes
    STATE_SWITCH_COSTS[NEW_STATE_NAME] = BASE_SWITCH_COST * 100  # Very high switch cost initially
    SWITCH_FREQUENCY_PARAM = 0.5
    
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
        # Check if tuning is requested
        if args.tune_pid:
            print("\n=== Running Ziegler-Nichols tuning ===")
            REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM = ziegler_nichols_tuning(
                None, INITIAL_STATES, BASE_REWARDS, epochs=epochs, n_agents=N_AGENTS)
        
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
        
        # Initialize the RL-tuned PID controller
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
            
            # Debug: print preserved rewards
            # print("Preserved rewards before controller update:", preserved_rewards)
            
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

                # Debug prev_rewards
                # print("Added prev_rewards:", new_controller.prev_rewards)
                
                # Add a flag for transition mode
                new_controller.in_transition = True
                new_controller.transition_epochs = 1000  # Number of epochs for smooth transition
                new_controller.transition_progress = 0  # Current progress (0-1)
                
                # Store the old controller's attribute
                new_controller.prev_rewards_by_attr = {}
                for k in range(NUM_ATTRIBUTES):
                    if k in preserved_rewards:
                        new_controller.prev_rewards_by_attr[k] = preserved_rewards[k]

                # Debug: check if transition flag is set
                # print("RL Controller in transition mode:", new_controller.in_transition)
                
                # Debug: ensure prev_rewards is properly set
                # print("Before assignment: Does new controller have prev_rewards?", hasattr(new_controller, 'prev_rewards'))
                # print("prev_rewards keys:", list(new_controller.prev_rewards.keys()) if hasattr(new_controller, 'prev_rewards') else None)
                
                # Make a backup of prev_rewards
                prev_rewards_backup = new_controller.prev_rewards.copy() if hasattr(new_controller, 'prev_rewards') else {}
                
                # Replace the old controller with the new one
                reward_controller = new_controller
                
                # Debug: check if prev_rewards survived the assignment
                # print("After assignment: Does controller have prev_rewards?", hasattr(reward_controller, 'prev_rewards'))
                # print("prev_rewards keys:", list(reward_controller.prev_rewards.keys()) if hasattr(reward_controller, 'prev_rewards') else None)
                
                # CRITICAL: Double-check that prev_rewards survived and restore if needed
                if not hasattr(reward_controller, 'prev_rewards') or len(reward_controller.prev_rewards) == 0:
                    print("WARNING: prev_rewards missing or empty after controller assignment, restoring from backup")
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
            
            # Debug: check restored state_rewards
            # print("Restored state_rewards:", state_rewards)
            
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
            
            # Track PID parameters (constant for PID controller)
            pid_params_history.append((
                reward_controller.p_param,
                reward_controller.i_param,
                reward_controller.d_param
            ))
            
        elif controller_type == "rl":
            # Use RL-tuned PID controller to update rewards
            try:
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
                
                # Calculate total rewards allocated correctly
                total_rewards = 0
                for state in current_states:
                    total_rewards += state_rewards[0][state]
                
                # Store the calculated total rewards
                total_rewards_history.append(total_rewards)
                reward_per_state_history.append(state_rewards[0].copy())
                
            except Exception as e:
                print(f"Error in RL controller update_rewards: {e}")
                import traceback
                traceback.print_exc()
                # Continue with previous rewards to avoid breaking the experiment
                total_rewards_history.append(total_rewards_history[-1] if total_rewards_history else 0)
                reward_per_state_history.append(reward_per_state_history[-1].copy() if reward_per_state_history else {state: 0 for state in current_states})
                pid_params_history.append(pid_params_history[-1] if pid_params_history else (args.initial_p, args.initial_i, args.initial_d))
        
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
            if epoch >= NEW_STATE_EPOCH and epoch <= NEW_STATE_EPOCH + 100 and epoch % 10 == 0:
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
    
    # Extract diversity values and ideal values for metrics
    diversity_values = [final_diversity[0][epoch][0] for epoch in range(epochs)]
    ideal_before = get_ideal_diversity(INITIAL_STATES)
    ideal_after = get_ideal_diversity(POSSIBLE_STATES)
    
    # Compile performance metrics
    from measurement_functions import compile_system_performance_metrics
    
    performance_metrics = compile_system_performance_metrics(
        diversity_values, total_rewards_history, pid_params_history,
        NEW_STATE_EPOCH, ideal_before, ideal_after
    )
    
    # Return the results
    return {
        'controller_type': controller_type,
        'final_diversity': final_diversity,
        'agents_declared_history': agents_declared_history,
        'total_rewards_history': total_rewards_history,
        'reward_per_state_history': reward_per_state_history,
        'pid_params_history': pid_params_history,
        'pid_params': (REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM),
        'final_pid_params': pid_params_history[-1] if pid_params_history else (REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM),
        'NEW_STATE_EPOCH': NEW_STATE_EPOCH,
        'NEW_STATE_NAME': NEW_STATE_NAME,
        'INITIAL_STATES': INITIAL_STATES,
        'POSSIBLE_STATES': POSSIBLE_STATES,
        'performance_metrics': performance_metrics,
        'epochs': epochs,
        'N_AGENTS': N_AGENTS
    }

def compare_and_visualize(comparison_results, args):
    """
    Compare and visualize the results from both controllers.
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary containing results from both PID and RL controllers
    args : argparse.Namespace
        Command line arguments
    """
    import matplotlib.pyplot as plt
    from plotting_functions import (
        plot_side_by_side_diversity, 
        plot_rewards_comparison,
        plot_rl_pid_parameter_evolution,
        plot_agent_distribution_comparison,
        plot_performance_metrics_comparison,
        create_metrics_summary_table
    )
    
    pid_results = comparison_results['pid']
    rl_results = comparison_results['rl']
    
    # Extract common parameters
    epochs = pid_results['epochs']
    epochs_list = list(range(epochs))
    NEW_STATE_EPOCH = pid_results['NEW_STATE_EPOCH']
    NEW_STATE_NAME = pid_results['NEW_STATE_NAME']
    INITIAL_STATES = pid_results['INITIAL_STATES']
    POSSIBLE_STATES = pid_results['POSSIBLE_STATES']
    
    print("\n\n" + "="*80)
    print(f"{'':^10}COMPARISON RESULTS: PID vs RL-TUNED PID{'':^10}")
    print("="*80)
    
    # 1. Compare diversity over time
    plt = plot_side_by_side_diversity(
        pid_results['final_diversity'],
        rl_results['final_diversity'],
        epochs_list,
        1,  # NUM_ATTRIBUTES
        NEW_STATE_EPOCH,
        NEW_STATE_NAME,
        INITIAL_STATES,
        POSSIBLE_STATES,
        pid_results['pid_params'],
        rl_results['final_pid_params']
    )
    plt.show()
    
    # 2. Compare rewards over time
    plt = plot_rewards_comparison(
        pid_results['total_rewards_history'],
        rl_results['total_rewards_history'],
        epochs_list,
        NEW_STATE_EPOCH,
        NEW_STATE_NAME
    )
    plt.show()
    
    # 3. Show RL parameter evolution
    plt = plot_rl_pid_parameter_evolution(
        rl_results['pid_params_history'],
        epochs_list,
        NEW_STATE_EPOCH,
        NEW_STATE_NAME,
        pid_results['pid_params']
    )
    plt.show()
    
    # 4. Compare agent distributions
    plt = plot_agent_distribution_comparison(
        pid_results['agents_declared_history'],
        rl_results['agents_declared_history'],
        epochs_list,
        POSSIBLE_STATES,
        NEW_STATE_EPOCH,
        NEW_STATE_NAME
    )
    plt.show()
    
    # 5. Compare performance metrics
    plt = plot_performance_metrics_comparison(
        pid_results['performance_metrics'],
        rl_results['performance_metrics']
    )
    plt.show()
    
    # 6. Print detailed metrics summary table
    metrics_table = create_metrics_summary_table(
        pid_results['performance_metrics'],
        rl_results['performance_metrics']
    )
    print("\n")
    print(metrics_table)