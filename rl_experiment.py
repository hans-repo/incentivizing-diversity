import random
import numpy as np
import matplotlib.pyplot as plt
from agent import *
from measurement_functions import *
from reinforcement_learning import *

# Configuration for the experiment
num_experiments = 1  # Change this value to run multiple experiments
NUM_ATTRIBUTES = 1   # Simplified for clarity

def main():
    # Initialize arrays for storing metrics across experiments
    # Each array will store one value per attribute per experiment
    final_diversity_all_experiments = [[] for _ in range(NUM_ATTRIBUTES)]
    average_diversity_by_experiment = [[] for _ in range(NUM_ATTRIBUTES)]
    ideal_diversity_all_experiments = [[] for _ in range(NUM_ATTRIBUTES)]
    largest_state_all_experiments = [[] for _ in range(NUM_ATTRIBUTES)]
    
    # Track experiment-specific metrics
    x_axis = []  # Experiment numbers for x-axis
    adaptation_metrics = []  # Store adaptation quality across experiments
    recovery_times = []  # Store recovery times across experiments
    phase1_convergence_times = []  # Convergence times for phase 1
    phase2_convergence_times = []  # Convergence times for phase 2
    
    for i in range(num_experiments):
        print(f"\n\n{'='*50}")
        print(f"STARTING EXPERIMENT {i+1}/{num_experiments}")
        print(f"{'='*50}\n")
        epochs = 3000  # Total epochs
        
        # Define the possible states an agent can be in
        INITIAL_STATES = ['NO_STATE', 'State_A', 'State_B', 'State_C', 'State_D']
        NEW_STATE_NAME = 'State_E'  # The new state to be added later
        NEW_STATE_EPOCH = round(0.2 * 1 * epochs)  # Epoch at which the new state is added
        
        POSSIBLE_STATES = INITIAL_STATES.copy()  # Start with initial states
        NUM_STATES = len(POSSIBLE_STATES)
        N_AGENTS = 100  # Number of agents
        PER_NODE_REWARD_BASE = 1000
        PER_NODE_REWARD = PER_NODE_REWARD_BASE  # Per node reward per epoch in an evenly distributed system
        
        # Define fixed rewards for each state
        BASE_REWARDS = PER_NODE_REWARD_BASE * N_AGENTS / (NUM_STATES-1)  # -1 for NO_STATE
        
        # Setup for agent costs
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
        SWITCH_FREQUENCY_PARAM = 0.5
        
        x_axis.append(i+1)  # Store experiment number (starting from 1 for better readability)
        
        # Initialize RL agent for learning reward allocation
        reward_learner = RewardLearner(INITIAL_STATES, N_AGENTS, reward_scale=PER_NODE_REWARD_BASE)
        
        # Initialize state rewards
        state_rewards = [{state: 0 for state in POSSIBLE_STATES} for _ in range(NUM_ATTRIBUTES)]
        
        # Pre-add the new state to rewards tracking, but make it unattractive
        for k in range(NUM_ATTRIBUTES):
            # Set initial reward very low to ensure zero initial adoption
            state_rewards[k][NEW_STATE_NAME] = 0  # No reward initially
        
        # Create initial agents with only initial states
        agents = generate_agents(N_AGENTS, INITIAL_STATES, state_rewards, STATE_RUN_COSTS, STATE_SWITCH_COSTS, SWITCH_FREQUENCY_PARAM)
        agents_real_history = [[] for _ in range(NUM_ATTRIBUTES)]
        agents_declared_history = [[] for _ in range(NUM_ATTRIBUTES)]
        
        # Track states available at each epoch for proper diversity calculation
        states_available_at_epoch = []
        
        # Track RL metrics for analysis
        exploration_rates = []  # Average exploration rate across all states
        rewards_history = []    # Environment rewards from reward calculation
        training_losses = []    # Training losses from DQN agents
        
        # Run the main experiment
        print("\n=== Starting main RL experiment ===")
        for epoch in range(epochs):
            # Add the new state at the specified epoch
            if epoch == NEW_STATE_EPOCH:
                print(f"\nAdding new state {NEW_STATE_NAME} at epoch {epoch}")
                POSSIBLE_STATES.append(NEW_STATE_NAME)
                
                # Reset the switch cost for the new state to normal level
                STATE_SWITCH_COSTS[NEW_STATE_NAME] = BASE_SWITCH_COST
                
                # Update reward learner to include the new state
                reward_learner.update_possible_states(POSSIBLE_STATES)
                
                # Update agents to know about the new state
                for agent in agents:
                    # Update agent's possible states list
                    agent.possible_states = POSSIBLE_STATES.copy()
                    
                    # Set normal switch cost for the new state
                    agent.switch_cost[NEW_STATE_NAME] = random.uniform(0.0, 1.0) * agent.personal_switch_cost
                
                # Print info about the new state
                print(f"New state added: {NEW_STATE_NAME}")
                print(f"Total states now: {len(POSSIBLE_STATES)}")
                print(f"Switch cost for {NEW_STATE_NAME}: {STATE_SWITCH_COSTS[NEW_STATE_NAME]}")
            
            # Keep track of which states were available at this epoch (for proper diversity calculation)
            current_states = INITIAL_STATES.copy() if epoch < NEW_STATE_EPOCH else POSSIBLE_STATES.copy()
            states_available_at_epoch.append(current_states)
                
            # Get rewards from RL agent based on current state distribution
            # Use the new update_state_rewards function similar to PID version in agent.py
            current_rewards, accumulated_error, last_error = reward_learner.update_state_rewards(agents)
            
            # Format the rewards to match the expected format in the simulation
            state_rewards = reward_learner.format_state_rewards(current_rewards, NUM_ATTRIBUTES)
            
            # Distribute rewards to agents
            state_rewards_last_epoch = distribute_rewards_rl(agents, current_states, state_rewards)
            
            # Record states of all agents
            for k in range(NUM_ATTRIBUTES):
                agents_real_history[k].append([agent.real_state[k] for agent in agents])
                agents_declared_history[k].append([agent.declared_state[k] for agent in agents])
            
            # Update agent decisions
            for agent in agents:
                agent.decision(state_rewards_last_epoch, malicious=False)
            
            # Record RL metrics
            # Get average exploration rate across all state agents
            state_epsilons = reward_learner.get_exploration_rates()
            avg_epsilon = sum(state_epsilons.values()) / len(state_epsilons) if state_epsilons else 0
            exploration_rates.append(avg_epsilon)
            
            # Calculate and store environment reward
            env_reward, _ = reward_learner.env.calculate_reward()
            rewards_history.append(env_reward)
            
            # Track training losses if available
            if hasattr(reward_learner, 'training_losses') and reward_learner.training_losses:
                if len(reward_learner.training_losses) > len(training_losses):
                    training_losses.append(reward_learner.training_losses[-1])
                
            # Print status at key epochs
            if epoch % 100 == 0 or epoch == NEW_STATE_EPOCH or epoch == NEW_STATE_EPOCH + 1:
                # Count agents in each state for the first attribute
                state_counts = {state: 0 for state in current_states}
                for agent in agents:
                    state_counts[agent.declared_state[0]] += 1
                
                print(f"Epoch {epoch} - Agents per state: {state_counts}")
                print(f"Epoch {epoch} - Rewards per state: {state_rewards_last_epoch}")
                print(f"Average exploration rate: {avg_epsilon:.4f}")
                print(f"Environment reward: {env_reward:.4f}")
                
                # Print some reward values for key states
                reward_samples = {state: current_rewards[state] for state in list(current_states)[:3] if state != 'NO_STATE'}
                print(f"Sample rewards: {reward_samples}")
                
                # Print training progress
                if training_losses:
                    recent_losses = training_losses[-min(10, len(training_losses)):]
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    print(f"Recent training loss avg: {avg_loss:.6f}")
                
                # Calculate and print current diversity
                if epoch > 0:  # Skip first epoch
                    temp_diversity = calculate_diversity([agents_declared_history[0]], [epoch-1], 
                                                         states_available_at_epoch[epoch-1], N_AGENTS)
                    print(f"Current diversity: {temp_diversity[0][epoch-1][0]}")
                    
                # Specifically print new state count when it matters
                if epoch >= NEW_STATE_EPOCH and epoch <= NEW_STATE_EPOCH + 10:
                    print(f"Agents in new state {NEW_STATE_NAME}: {state_counts.get(NEW_STATE_NAME, 0)}")
                    print(f"Current reward for {NEW_STATE_NAME}: {state_rewards_last_epoch[0].get(NEW_STATE_NAME, 0)}")
        
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
            # Extract the last 25% of diversity values
            last_25_percent_diversity = [final_diversity[k][epoch] for epoch in range(epochs - last_25_percent, epochs)]
            
            # Flatten and calculate average
            flattened_diversity = [item for sublist in last_25_percent_diversity for item in sublist]
            avg_diversity = sum(flattened_diversity) / len(flattened_diversity)
            
            # Store this experiment's average diversity
            average_diversity_by_experiment[k].append(avg_diversity)
            
            # Calculate ideal diversity for final set of states 
            ideal_diversity = get_ideal_diversity(POSSIBLE_STATES)
            ideal_diversity_all_experiments[k].append(ideal_diversity)
            
            # Calculate largest state metric
            largest_state = get_largest_state(agents_declared_history, range(epochs), POSSIBLE_STATES, N_AGENTS)
            last_25_percent_largest_state = [largest_state[k][epoch] for epoch in range(epochs - last_25_percent, epochs)]
            flattened_largest_state = [item for sublist in last_25_percent_largest_state for item in sublist]
            avg_largest_state = sum(flattened_largest_state) / len(flattened_largest_state)
            largest_state_all_experiments[k].append(avg_largest_state)
        
        # Get diversity values and ideal values for convergence analysis
        diversity_values = [final_diversity[0][epoch][0] for epoch in range(epochs)]
        ideal_before = get_ideal_diversity(INITIAL_STATES)
        ideal_after = get_ideal_diversity(POSSIBLE_STATES)
        
        # Analyze convergence in both phases
        convergence_results = measure_dual_phase_convergence(
            diversity_values, NEW_STATE_EPOCH, ideal_before, ideal_after, 0.9)
        
        # Store convergence metrics for this experiment
        phase1_convergence_time = convergence_results['phase1']['epochs_to_converge']
        phase2_convergence_time = convergence_results['phase2']['re_convergence_time']
        
        phase1_convergence_times.append(phase1_convergence_time if phase1_convergence_time is not None else epochs)
        phase2_convergence_times.append(phase2_convergence_time if phase2_convergence_time is not None else epochs)
        
        # Get adaptation metrics
        adapt_metrics = analyze_system_adaptability(
            diversity_values, NEW_STATE_EPOCH, ideal_before, ideal_after)
        
        # Store adaptation quality
        if "final_adaptation_quality" in adapt_metrics and adapt_metrics["final_adaptation_quality"] is not None:
            adaptation_metrics.append(adapt_metrics["final_adaptation_quality"])
        else:
            adaptation_metrics.append(0.0)
            
        # Store recovery time
        if "recovery_time_epochs" in adapt_metrics and adapt_metrics["recovery_time_epochs"] is not None:
            recovery_times.append(adapt_metrics["recovery_time_epochs"])
        else:
            recovery_times.append(epochs)  # Use max epochs if no recovery
    
    # Display results for single experiment case
    if num_experiments == 1:
        epochs_list = list(range(epochs))
        
        # Print summary statistics
        print("\n=== REINFORCEMENT LEARNING EXPERIMENT RESULTS ===")
        print(f"Final average diversity (last 25%): {average_diversity_by_experiment[0][0]:.4f}")
        print(f"Final largest state share: {largest_state_all_experiments[0][0]:.2%}")
        print(f"Phase 1 convergence time: {phase1_convergence_times[0]} epochs")
        print(f"Phase 2 convergence time: {phase2_convergence_times[0]} epochs")
        print(f"Recovery time: {recovery_times[0]} epochs")
        print(f"Final adaptation quality: {adaptation_metrics[0]*100:.1f}% of ideal")
        print(f"Final average exploration rate (epsilon): {exploration_rates[-1]:.4f}")
        print(f"Final RL reward: {rewards_history[-1]:.4f}")
        
        # Print training performance
        if training_losses:
            print(f"Final training loss: {training_losses[-1]:.6f}")
            print(f"Training loss improvement: {(training_losses[0] - training_losses[-1]) / training_losses[0]:.2%}")
        
        # Plot training loss over time
        if training_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(training_losses, linewidth=1, alpha=0.8)
            plt.title('DQN Training Loss Over Time')
            plt.xlabel('Training Updates')
            plt.ylabel('Loss')
            plt.yscale('log')  # Log scale often better for visualizing loss
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
        
        # Plot diversity over time
        plt.figure(figsize=(12, 7))
        for k in range(NUM_ATTRIBUTES):
            # Extract diversity values from dictionary
            diversity_values = [final_diversity[k][epoch][0] for epoch in epochs_list]  # [0] to get the first (and only) value
            plt.plot(epochs_list, diversity_values, label=f"Diversity - Attribute {k}")
        
        # Add vertical line to mark when new state was added
        plt.axvline(x=NEW_STATE_EPOCH, color='r', linestyle='--', 
                    label=f"New state ({NEW_STATE_NAME}) added")
        
        # Add horizontal line for ideal diversity before and after new state
        ideal_before = get_ideal_diversity(INITIAL_STATES)
        ideal_after = get_ideal_diversity(POSSIBLE_STATES)
        plt.axhline(y=ideal_before, color='g', linestyle=':', 
                    label=f"Ideal diversity - {len(INITIAL_STATES)-1} states")
        plt.axhline(y=ideal_after, color='g', linestyle='-', 
                    label=f"Ideal diversity - {len(POSSIBLE_STATES)-1} states")
        
        plt.xlabel('Epochs')
        plt.ylabel('Diversity')
        plt.title('Diversity with Reinforcement Learning')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        # Plot RL metrics
        plot_exploration_rate(exploration_rates, epochs)
        plt.show()
        
        plot_rl_rewards(rewards_history, epochs)
        plt.show()
        
        print("\n=== Final Results ===")
        print("Final states in system:", POSSIBLE_STATES)
        print("Initial base rewards:", BASE_REWARDS)
        print(f"Final ideal diversity ({len(POSSIBLE_STATES)-1} states):", ideal_after)
        
        # After your experiment runs:
        diversity_values = [final_diversity[0][epoch][0] for epoch in range(epochs)]

        # Print convergence analysis with better formatting
        print("\n" + "="*50)
        print("           SYSTEM CONVERGENCE ANALYSIS           ")
        print("="*50)

        print(f"\n💡 DIVERSITY TARGETS:")
        print(f"   • Initial phase (4 states): {ideal_before:.4f} bits")
        print(f"   • Final phase (5 states):   {ideal_after:.4f} bits")
        print(f"   • State transition occurred at epoch {NEW_STATE_EPOCH}")

        # Format Phase 1 results
        print("\n📈 PHASE 1 CONVERGENCE (EPOCHS 0-{})".format(NEW_STATE_EPOCH-1))
        p1_conv = convergence_results['phase1']['epochs_to_converge']
        if p1_conv is not None:
            print(f"   • Time to reach 90% of ideal:   {p1_conv} epochs")
        else:
            print("   • System did not reach 90% of ideal diversity")
            
        p1_pct = convergence_results['phase1']['percentage_reached']*100
        print(f"   • Maximum diversity achieved:    {p1_pct:.1f}% of ideal")

        # Format Phase 2 results
        print("\n📉 PHASE 2 CONVERGENCE (EPOCHS {}-END)".format(NEW_STATE_EPOCH))
        p2_conv = convergence_results['phase2']['re_convergence_time']
        if p2_conv is not None:
            print(f"   • Time to reach 90% of new ideal: {p2_conv} epochs after transition")
        else:
            print("   • System did not reach 90% of new ideal diversity")
            
        p2_pct = convergence_results['phase2']['percentage_reached']*100
        print(f"   • Maximum diversity achieved:    {p2_pct:.1f}% of ideal")

        # Format adaptation metrics
        print("\n🔄 ADAPTATION METRICS")
        shock = convergence_results['overall']['adaptation_shock']
        if shock is not None:
            print(f"   • Diversity drop at transition:  {shock*100:.1f}% of pre-change value")
        else:
            print("   • Could not calculate diversity drop")
            
        recovery = convergence_results['overall']['recovery_time']
        if recovery is not None:
            print(f"   • Recovery time:                 {recovery} epochs after transition")
        else:
            print("   • System did not recover to pre-change diversity levels")

        # More detailed adaptability analysis
        adapt_metrics = analyze_system_adaptability(
            diversity_values, NEW_STATE_EPOCH, ideal_before, ideal_after)

        print("\n" + "="*50)
        print("           DETAILED ADAPTABILITY METRICS           ")
        print("="*50)

        if "error" in adapt_metrics:
            print(f"\n⚠️ ANALYSIS ERROR: {adapt_metrics['error']}")
        else:
            print(f"\n📊 PRE-CHANGE STABILITY")
            print(f"   • Average diversity before change: {adapt_metrics['pre_change_diversity_avg']:.4f} bits")

            print(f"\n📊 ADAPTATION SHOCK")
            if adapt_metrics['initial_drop_pct'] is not None:
                print(f"   • Initial diversity drop:         {adapt_metrics['initial_drop_pct']*100:.1f}%")
            
            print(f"\n📊 RECOVERY SPEED")
            if adapt_metrics['recovery_time_epochs'] is not None:
                print(f"   • Time to return to pre-change level: {adapt_metrics['recovery_time_epochs']} epochs")
            else:
                print(f"   • System did not return to pre-change level")
                
            if adapt_metrics['time_to_90pct_new_ideal'] is not None:
                print(f"   • Time to reach 90% of new ideal:    {adapt_metrics['time_to_90pct_new_ideal']} epochs")
            else:
                print(f"   • System did not reach 90% of new ideal")
            
            print(f"\n📊 FINAL PERFORMANCE")
            if adapt_metrics['phase2_settling_time'] is not None:
                print(f"   • System settled after:              {adapt_metrics['phase2_settling_time']} epochs")
            else:
                print(f"   • System did not fully settle")
                
            if adapt_metrics['final_adaptation_quality'] is not None:
                print(f"   • Final adaptation quality:          {adapt_metrics['final_adaptation_quality']*100:.1f}% of ideal")
            
        print("\n" + "="*50)

        # Plot stacked area chart showing agent distribution
        for k in range(NUM_ATTRIBUTES):
            plt.figure(figsize=(12, 7))
            agents_history = [agents_declared_history[k][epoch] for epoch in epochs_list]
            
            # Use our updated plotting function
            plot_stacked_area(agents_history, epochs, POSSIBLE_STATES)
            
            # Add vertical line for when new state was added
            plt.axvline(x=NEW_STATE_EPOCH, color='r', linestyle='--', 
                      label=f"New state ({NEW_STATE_NAME}) added")
            
            plt.title(f"Agent Distribution Over Time with RL Controller")
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()

        
if __name__ == "__main__":
    main()