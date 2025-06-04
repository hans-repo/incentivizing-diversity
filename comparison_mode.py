import random
import numpy as np
import torch
from agent import generate_agents, distribute_rewards
from measurement_functions import (calculate_diversity, get_ideal_diversity, 
                                    compile_system_performance_metrics)
from ziegler_nichols_tuning import ziegler_nichols_tuning
from reinforcement_learning import PIDController, RLPIDController
from reinforcement_learning_nopid import RLDirectRewardController
from plotting_functions import (print_experiment_header, print_pid_parameters, 
                                print_rl_parameters, print_new_state_added, 
                                print_epoch_status, print_new_state_status)
from reinforcement_learning import RLPIDController
from plotting_functions import compare_and_visualize_all_controllers
from plotting_functions import print_new_state_added
from measurement_functions import calculate_diversity
from plotting_functions import print_epoch_status, print_new_state_status

    
def run_comparison_mode(args):
    """
    Run all three controllers (PID, RL, and RL-no-PID) with the same parameters and compare their performance.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    print("\n" + "="*80)
    print(f"{'':^10}RUNNING COMPARISON MODE: PID vs RL vs RL-NO-PID{'':^10}")
    print("="*80)
    
    # Save original controller type
    original_controller = args.controller
    
    # Store results for all controllers
    comparison_results = {}
    
    # Determine which controllers to run based on compare_modes argument
    controllers_to_run = args.compare_modes if hasattr(args, 'compare_modes') else ['pid', 'rl', 'rlnopid']
    
    # Run each controller experiment
    for controller in controllers_to_run:
        print("\n\n" + "="*60)
        print(f"RUNNING {controller.upper()} CONTROLLER EXPERIMENT")
        print("="*60)
        
        args.controller = controller
        controller_results = run_single_controller_experiment(args)
        comparison_results[controller] = controller_results
    
    # Restore original controller type
    args.controller = original_controller
    
    # Compare and visualize the results

    compare_and_visualize_all_controllers(comparison_results, args)
    
    return comparison_results

def run_single_controller_experiment(args):
    """
    Run a single experiment with PID, RL, or RL-no-PID controller.
    This uses the existing experiment infrastructure from multiple_experiments.py
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
        
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
    
    # Use the same state configuration as main experiments
    # INITIAL_STATES = ['NO_STATE', 'State_A', 'State_B', 'State_C', 'State_D']
    # INITIAL_STATES = ['NO_STATE', 'State_A', 'State_B', 'State_C', 'State_D', 'State_E','State_F','State_G']
    # INITIAL_STATES = ['NO_STATE', 'State_A', 'State_B', 'State_C', 'State_D', 'State_E', 'State_F', 'State_G', 'State_H']
    INITIAL_STATES = ['NO_STATE', 'State_A', 'State_B', 'State_C', 'State_D', 'State_E', 'State_F', 'State_G', 'State_H', 'State_I', 'State_J', 'State_K', 'State_L']
    NEW_STATE_NAME = 'State_NEW'
    NEW_STATE_EPOCH = round(0.5 * epochs)
    
    POSSIBLE_STATES = INITIAL_STATES.copy()
    POSSIBLE_STATES.append(NEW_STATE_NAME)
    NUM_STATES = len(POSSIBLE_STATES)

    # PID parameters
    REWARDS_ADAPTIVE_PARAM = args.p_param
    REWARDS_INTEGRAL_PARAM = args.i_param
    REWARDS_DERIVATIVE_PARAM = args.d_param
    
    # Cost configuration
    BASE_RUN_COST = 100 + 0
    RUN_COST_CEILING = BASE_RUN_COST + 2*BASE_RUN_COST
    BASE_SWITCH_COST = BASE_RUN_COST

    # Generate evenly spaced values for initial states
    state_run_costs = np.linspace(BASE_RUN_COST, RUN_COST_CEILING, NUM_STATES)
    STATE_RUN_COSTS = {
        state: round(value, 4)
        for state, value in zip(POSSIBLE_STATES, state_run_costs)
    }
    
    # Add run cost for the new state that will be added later
    STATE_RUN_COSTS[NEW_STATE_NAME] = round(BASE_RUN_COST + RUN_COST_CEILING, 4)
    
    STATE_SWITCH_COSTS = {state: BASE_SWITCH_COST for state in POSSIBLE_STATES}
    # Set a high initial switch cost for the new state to ensure zero initial nodes
    STATE_SWITCH_COSTS[NEW_STATE_NAME] = BASE_SWITCH_COST * 100
    SWITCH_FREQUENCY_PARAM = 3
    
    # Initialize state rewards and tracking
    state_rewards = [{state: 0 for state in POSSIBLE_STATES} for _ in range(NUM_ATTRIBUTES)]
    
    # Pre-add the new state to rewards tracking, but make it unattractive
    for k in range(NUM_ATTRIBUTES):
        state_rewards[k][NEW_STATE_NAME] = 0  # No reward initially
    
    # Initialize the appropriate controller based on the controller_type
    if controller_type == "pid":
        if args.tune_pid:
            print("\n=== Running Ziegler-Nichols tuning ===")
            REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM = ziegler_nichols_tuning(
                None, INITIAL_STATES, epochs=epochs, n_agents=N_AGENTS)
        
        print_pid_parameters(REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM)
        
        reward_controller = PIDController(
            POSSIBLE_STATES,
            REWARDS_ADAPTIVE_PARAM,
            REWARDS_INTEGRAL_PARAM,
            REWARDS_DERIVATIVE_PARAM
        )
    
    elif controller_type == "rl":
        print_rl_parameters(args.epsilon, args.epsilon_decay, args.learning_rate, args.action_scale, 
                        args.initial_p, args.initial_i, args.initial_d,
                          args.p_scale_factor, args.i_scale_factor, args.d_scale_factor)
        
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
            POSSIBLE_STATES,
            N_AGENTS,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            learning_rate=args.learning_rate,
            action_scale=args.action_scale,
            update_frequency=5,
            batch_size=16
        )
    
    # Create initial agents
    agents = generate_agents(N_AGENTS, INITIAL_STATES, state_rewards, STATE_RUN_COSTS, STATE_SWITCH_COSTS, SWITCH_FREQUENCY_PARAM)
    agents_real_history = [[] for _ in range(NUM_ATTRIBUTES)]
    agents_declared_history = [[] for _ in range(NUM_ATTRIBUTES)]
    
    # Track states available at each epoch
    states_available_at_epoch = []
    
    # Track metrics over time
    pid_params_history = []
    total_rewards_history = []
    reward_per_state_history = []
    
    # Run the main experiment loop
    print("\n=== Starting main experiment ===")
    for epoch in range(epochs):
        # Handle new state addition
        if epoch == NEW_STATE_EPOCH:
            # Use existing helper function for state addition logic
            _handle_new_state_addition(epoch, NEW_STATE_NAME, NEW_STATE_EPOCH, 
                                     POSSIBLE_STATES, agents, 
                                     state_rewards, STATE_SWITCH_COSTS, BASE_SWITCH_COST,
                                     reward_controller, controller_type, args, 
                                     INITIAL_STATES, N_AGENTS, NUM_ATTRIBUTES)
        
        # Keep track of available states
        current_states = INITIAL_STATES.copy() if epoch < NEW_STATE_EPOCH else POSSIBLE_STATES.copy()
        states_available_at_epoch.append(current_states)
        
        # Calculate current diversity for RL controllers
        current_diversity = None
        if (controller_type == "rl" or controller_type == "rlnopid") and epoch > 0:
            diversity_result = calculate_diversity(
                [agents_declared_history[0]], [epoch-1], 
                states_available_at_epoch[epoch-1], N_AGENTS)
            current_diversity = diversity_result[0][epoch-1][0]
            
            # Update agent counts in RL controller
            agents_in_states = {state: 0 for state in current_states}
            for agent in agents:
                agent_state = agent.declared_state[0]
                agents_in_states[agent_state] = agents_in_states.get(agent_state, 0) + 1
            reward_controller.agent_counts = agents_in_states
        
        ideal_diversity = get_ideal_diversity(current_states)
        
        # Update rewards using appropriate controller
        total_rewards = _update_rewards_for_controller(
            controller_type, reward_controller, agents, current_states, 
            state_rewards, current_diversity, ideal_diversity, epoch, args)
        
        total_rewards_history.append(total_rewards)
        reward_per_state_history.append(state_rewards[0].copy())
        
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
        state_rewards_last_epoch = distribute_rewards(agents, current_states, state_rewards)
        
        # Record agent states
        for k in range(NUM_ATTRIBUTES):
            agents_real_history[k].append([agent.real_state[k] for agent in agents])
            agents_declared_history[k].append([agent.declared_state[k] for agent in agents])
        
        # Update agent decisions
        for agent in agents:
            agent.decision(state_rewards_last_epoch, malicious=False)
        
        # Print status at key epochs
        if epoch % 1000 == 0 or epoch == NEW_STATE_EPOCH or epoch == NEW_STATE_EPOCH + 1:
            _print_epoch_status(epoch, current_states, agents, state_rewards_last_epoch,
                              total_rewards, controller_type, reward_controller, 
                              agents_declared_history, states_available_at_epoch, N_AGENTS,
                              NEW_STATE_EPOCH, NEW_STATE_NAME)
    
    # Calculate final diversity
    final_diversity = [{}] * NUM_ATTRIBUTES
    for k in range(NUM_ATTRIBUTES):
        final_diversity[k] = {}
        for epoch in range(epochs):
            epoch_diversity = calculate_diversity(
                [agents_declared_history[k]], [epoch], 
                states_available_at_epoch[epoch], N_AGENTS)
            final_diversity[k][epoch] = epoch_diversity[0][epoch]
    
    # Extract diversity values for metrics
    diversity_values = [final_diversity[0][epoch][0] for epoch in range(epochs)]
    ideal_before = get_ideal_diversity(INITIAL_STATES)
    ideal_after = get_ideal_diversity(POSSIBLE_STATES)
    
    # Compile performance metrics
    performance_metrics = compile_system_performance_metrics(
        diversity_values, total_rewards_history, pid_params_history,
        NEW_STATE_EPOCH, ideal_before, ideal_after
    )
    
    # Return results
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

def _handle_new_state_addition(epoch, NEW_STATE_NAME, NEW_STATE_EPOCH, POSSIBLE_STATES, 
                              agents, state_rewards, STATE_SWITCH_COSTS, 
                              BASE_SWITCH_COST, reward_controller, controller_type, args, 
                              INITIAL_STATES, N_AGENTS, NUM_ATTRIBUTES):
    """Helper function to handle new state addition logic"""

    # Reset switch cost for new state
    STATE_SWITCH_COSTS[NEW_STATE_NAME] = BASE_SWITCH_COST
    

    # Update agents
    for agent in agents:
        agent.possible_states = POSSIBLE_STATES.copy()
        agent.switch_cost[NEW_STATE_NAME] = BASE_SWITCH_COST
    
    # Preserve existing rewards
    preserved_rewards = {}
    for k in range(NUM_ATTRIBUTES):
        preserved_rewards[k] = {state: state_rewards[k][state] for state in INITIAL_STATES}
    
    # Update controller for new state
    if controller_type == "rl":
        _update_rl_controller_for_new_state(reward_controller, POSSIBLE_STATES, N_AGENTS, 
                                           args, INITIAL_STATES, preserved_rewards)
    elif controller_type == "pid":
        reward_controller.states = POSSIBLE_STATES.copy()
        reward_controller.accumulated_error[NEW_STATE_NAME] = 0
        reward_controller.last_error[NEW_STATE_NAME] = 0
    # RL-no-PID handles state changes internally
    
    # Restore preserved rewards
    for k in range(NUM_ATTRIBUTES):
        for state, value in preserved_rewards[k].items():
            state_rewards[k][state] = value
        state_rewards[k][NEW_STATE_NAME] = 0
    
    print_new_state_added(NEW_STATE_NAME, epoch, len(POSSIBLE_STATES), 
                         STATE_SWITCH_COSTS[NEW_STATE_NAME])

def _update_rl_controller_for_new_state(reward_controller, POSSIBLE_STATES, N_AGENTS, 
                                       args, INITIAL_STATES, preserved_rewards):
    """Helper function to update RL controller when new state is added"""

    # Store current state
    current_p = reward_controller.p_param
    current_i = reward_controller.i_param
    current_d = reward_controller.d_param
    current_reward_scale = reward_controller.reward_scale
    current_epsilon = reward_controller.epsilon
    current_steps_done = reward_controller.steps_done
    
    # Store accumulated errors
    old_accumulated_errors = {state: reward_controller.accumulated_error[state] 
                            for state in INITIAL_STATES if state in reward_controller.accumulated_error}
    old_last_errors = {state: reward_controller.last_error[state] 
                     for state in INITIAL_STATES if state in reward_controller.last_error}
    
    # Calculate error scaling
    old_ideal_share = 1.0 / (len(INITIAL_STATES) - 1)
    new_ideal_share = 1.0 / (len(POSSIBLE_STATES) - 1)
    error_scale_factor = new_ideal_share / old_ideal_share
    
    # Create new controller
    new_controller = RLPIDController(
        POSSIBLE_STATES, N_AGENTS,
        initial_p=current_p, initial_i=current_i, initial_d=current_d,
        epsilon=current_epsilon, epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate, action_scale=args.action_scale,
        update_frequency=5, batch_size=16,
        p_scale_factor=args.p_scale_factor,
        i_scale_factor=args.i_scale_factor,
        d_scale_factor=args.d_scale_factor
    )
    
    # Transfer state
    new_controller.reward_scale = current_reward_scale
    new_controller.steps_done = current_steps_done
    new_controller.best_pid_params = reward_controller.best_pid_params
    new_controller.best_diversity_ratio = reward_controller.best_diversity_ratio
    new_controller.best_reward_scale = reward_controller.best_reward_scale
    
    if hasattr(reward_controller, 'replay_buffer'):
        new_controller.replay_buffer = reward_controller.replay_buffer
    
    # Scale accumulated errors
    for state in INITIAL_STATES:
        if state in old_accumulated_errors:
            new_controller.accumulated_error[state] = old_accumulated_errors[state] * error_scale_factor
        if state in old_last_errors:
            new_controller.last_error[state] = old_last_errors[state] * error_scale_factor
    
    # Set transition mode
    new_controller.prev_rewards = {state: preserved_rewards[0][state] for state in INITIAL_STATES 
                                 if state in preserved_rewards[0]}
    new_controller.in_transition = True
    new_controller.transition_epochs = 1000
    new_controller.transition_progress = 0
    
    # Update the original controller reference
    reward_controller.__dict__.update(new_controller.__dict__)

def _update_rewards_for_controller(controller_type, reward_controller, agents, current_states, 
                                 state_rewards, current_diversity, ideal_diversity, epoch, args):
    """Helper function to update rewards based on controller type"""
    if controller_type == "pid":
        state_rewards = reward_controller.update_rewards(agents, current_states, state_rewards)
        total_rewards = sum(state_rewards[0][state] for state in current_states)
        
    elif controller_type == "rl":
        try:
            state_rewards = reward_controller.update_rewards(
                agents, current_states, state_rewards,
                current_diversity=current_diversity, 
                ideal_diversity=ideal_diversity,
                epoch=epoch
            )
            total_rewards = sum(state_rewards[0][state] for state in current_states)
        except Exception as e:
            print(f"Error in RL controller: {e}")
            total_rewards = 0
            
    elif controller_type == "rlnopid":
        try:
            state_rewards = reward_controller.update_rewards(
                agents, current_states, state_rewards,
                current_diversity=current_diversity, 
                ideal_diversity=ideal_diversity,
                epoch=epoch
            )
            total_rewards = sum(state_rewards[0][state] for state in current_states)
        except Exception as e:
            print(f"Error in RL-no-PID controller: {e}")
            total_rewards = 0
    
    return total_rewards

def _print_epoch_status(epoch, current_states, agents, state_rewards_last_epoch,
                       total_rewards, controller_type, reward_controller, 
                       agents_declared_history, states_available_at_epoch, N_AGENTS,
                       NEW_STATE_EPOCH, NEW_STATE_NAME):
    """Helper function to print epoch status"""

    # Count agents in each state
    state_counts = {state: 0 for state in current_states}
    for agent in agents:
        state_counts[agent.declared_state[0]] += 1
    
    # Calculate current diversity
    current_epoch_diversity = None
    if epoch > 0:
        temp_diversity = calculate_diversity([agents_declared_history[0]], [epoch-1], 
                                           states_available_at_epoch[epoch-1], N_AGENTS)
        current_epoch_diversity = temp_diversity[0][epoch-1][0]
    
    print_epoch_status(epoch, state_counts, state_rewards_last_epoch, 
                     total_rewards, controller_type, 
                     reward_controller, current_epoch_diversity)
    
    if epoch >= NEW_STATE_EPOCH and epoch <= NEW_STATE_EPOCH + 100 and epoch % 10 == 0:
        print_new_state_status(epoch, NEW_STATE_EPOCH, NEW_STATE_NAME, 
                             state_counts, state_rewards_last_epoch)