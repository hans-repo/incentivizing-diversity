import random
import numpy as np
import matplotlib.pyplot as plt
from agent import *
from measurement_functions import *

# Configuration for the experiment
num_experiments = 1
NUM_ATTRIBUTES = 1  # Simplified for clarity

def ziegler_nichols_tuning(agents, possible_states, base_rewards, epochs=1000, n_agents=100):
    """
    Implement Ziegler-Nichols PID tuning method by finding critical gain and period.
    Returns tuned P, I, D parameters based on critical gain (Ku) and critical period (Tu).
    """
    print("Starting Ziegler-Nichols tuning...")
    
    # We'll start with P control only and increase Kp until oscillation
    test_epochs = 500  # Number of epochs per test
    num_attributes = 1  # For simplicity, tune for one attribute
    
    # Storage for state distribution history to detect oscillations
    state_history = []
    
    # Start with low gain and increase until we see sustained oscillations
    Kp_values = np.logspace(4.8, 5, 100)  # Test range from 1 to 10000
    Ku = 0  # Critical gain
    Tu = 0  # Critical period
    
    for Kp in Kp_values:
        print(f"Testing Kp = {Kp}")
        
        # Reset for this test
        state_rewards = [{state: base_rewards for state in possible_states} for _ in range(num_attributes)]
        accumulated_error = [{state: 0 for state in possible_states} for _ in range(num_attributes)]
        last_error = 0
        
        # Create fresh agents for each test
        test_run_costs = {state: 100 for state in possible_states}  # Simplified run costs
        test_switch_costs = {state: 100 for state in possible_states}  # Simplified switch costs
        test_agents = generate_agents(n_agents, possible_states, state_rewards, test_run_costs, test_switch_costs)
        
        # For detecting oscillations
        state_counts_history = []
        
        # Run simulation with current Kp (P only controller)
        for epoch in range(test_epochs):
            # P-only control (Ki=0, Kd=0)
            state_rewards, accumulated_error, last_error = update_state_rewards(
                test_agents, possible_states, state_rewards, accumulated_error, last_error, 
                Kp, 0, 0)  # P only, I=0, D=0
                
            state_rewards_last_epoch = distribute_rewards(test_agents, possible_states, state_rewards)
            
            # Record state distribution to detect oscillations
            state_counts = {state: 0 for state in possible_states}
            for agent in test_agents:
                state_counts[agent.declared_state[0]] += 1
            state_counts_history.append(state_counts)
            
            # Update agent decisions
            for agent in test_agents:
                agent.decision(state_rewards_last_epoch, malicious=False)
        
        # Check for sustained oscillations in the second half of the simulation
        analysis_window = state_counts_history[test_epochs//2:]
        
        # Extract counts for one state (any non-NO_STATE) to analyze oscillations
        oscillation_state = [state for state in possible_states if state != 'NO_STATE'][0]
        counts = [d[oscillation_state] for d in analysis_window]
        
        # Detect oscillations using FFT
        if len(counts) > 10:  # Ensure enough data points
            fft_result = np.fft.fft(counts - np.mean(counts))
            freqs = np.fft.fftfreq(len(counts))
            
            # Get dominant frequency (exclude DC component)
            pos_freqs = freqs[1:len(freqs)//2]
            pos_amps = np.abs(fft_result)[1:len(freqs)//2]
            
            if len(pos_amps) > 0 and max(pos_amps) > n_agents * 0.1:  # Significant oscillation
                dominant_idx = np.argmax(pos_amps)
                dominant_freq = pos_freqs[dominant_idx]
                
                if dominant_freq > 0:  # Valid frequency found
                    period = 1.0 / abs(dominant_freq)
                    amplitude = max(counts) - min(counts)
                    
                    # Sustained oscillation criteria: amplitude > 10% of agents and stable period
                    if amplitude > n_agents * 0.1 and 2 < period < test_epochs//2:
                        print(f"Sustained oscillation detected at Kp={Kp}, period={period:.2f}")
                        Ku = Kp
                        Tu = period
                        
                        # Plot the oscillation
                        plt.figure(figsize=(12, 6))
                        plt.plot(counts)
                        plt.title(f"Oscillation at Critical Gain Ku={Ku:.2f}, Period Tu={Tu:.2f}")
                        plt.xlabel("Epoch")
                        plt.ylabel(f"Agents in {oscillation_state}")
                        plt.grid(True)
                        plt.show()
                        
                        break  # We found the critical values
    
    if Ku == 0 or Tu == 0:
        print("Could not find critical values. Using defaults.")
        return 20000, 10, 10  # Default values
    
    # Calculate PID parameters using Ziegler-Nichols method
    # Classic PID formula
    # Kp = 0.6 * Ku
    # Ki = 1.2 * Ku / Tu
    # Kd = 0.075 * Ku * Tu
    
    # Alternative: Try less aggressive tuning if system is unstable
    Kp = 0.2 * Ku
    Ki = 0.4 * Ku / Tu
    Kd = 0.025 * Ku * Tu
    
    print(f"Ziegler-Nichols tuning complete:")
    print(f"Critical Gain (Ku): {Ku}")
    print(f"Critical Period (Tu): {Tu}")
    print(f"Tuned parameters: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}")
    
    return Kp, Ki, Kd

def main():
    final_diversity_all_experiments = [[] for _ in range(NUM_ATTRIBUTES)]
    last_25_percent_diversity = [[] for _ in range(NUM_ATTRIBUTES)]
    flattened_diversity = [[] for _ in range(NUM_ATTRIBUTES)]
    average_last_25_percent = [[] for _ in range(NUM_ATTRIBUTES)]
    ideal_diversity_all_experiments = [[] for _ in range(NUM_ATTRIBUTES)]
    largest_state_all_experiments = [[] for _ in range(NUM_ATTRIBUTES)]

    last_25_percent_largest_state = [[] for _ in range(NUM_ATTRIBUTES)]
    flattened_largest_state = [[] for _ in range(NUM_ATTRIBUTES)]
    average_last_25_percent_largest_state = [[] for _ in range(NUM_ATTRIBUTES)]
    x_axis = []
    
    for i in range(num_experiments):
        # Define the possible states an agent can be in
        INITIAL_STATES = ['NO_STATE', 'State_A', 'State_B', 'State_C', 'State_D']
        NEW_STATE_NAME = 'State_E'  # The new state to be added later
        NEW_STATE_EPOCH = 800  # Epoch at which the new state is added
        
        POSSIBLE_STATES = INITIAL_STATES.copy()  # Start with initial states
        NUM_STATES = len(POSSIBLE_STATES)
        N_AGENTS = 100  # Number of agents
        PER_NODE_REWARD_BASE = 1000
        PER_NODE_REWARD = PER_NODE_REWARD_BASE  # Per node reward per epoch in an evenly distributed system
        
        # Define fixed rewards for each state
        BASE_REWARDS = PER_NODE_REWARD_BASE * N_AGENTS / (NUM_STATES-1)  # -1 for NO_STATE
        
        # Will be replaced by Ziegler-Nichols tuning
        REWARDS_ADAPTIVE_PARAM = 1000  # Initial P value before tuning
        REWARDS_INTEGRAL_PARAM = 100   # Initial I value before tuning
        REWARDS_DERIVATIVE_PARAM = 100 # Initial D value before tuning
        
        BASE_RUN_COST = 0 + 100
        RUN_COST_CEILING = BASE_RUN_COST + 2*BASE_RUN_COST
        BASE_SWITCH_COST = BASE_RUN_COST
        epochs = 1000  # Total epochs
        
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
        
        x_axis.append(i*0.01*100)
        
        # Initialize state rewards and accumulated error
        state_rewards = [{state: BASE_REWARDS for state in POSSIBLE_STATES} for _ in range(NUM_ATTRIBUTES)]
        accumulated_error = [{state: 0 for state in POSSIBLE_STATES} for _ in range(NUM_ATTRIBUTES)]
        
        # Pre-add the new state to rewards and error tracking, but make it unattractive
        for k in range(NUM_ATTRIBUTES):
            # Set initial reward very low to ensure zero initial adoption
            state_rewards[k][NEW_STATE_NAME] = 0  # No reward initially
            accumulated_error[k][NEW_STATE_NAME] = 0
            
        last_error = 0
        
        # Run Ziegler-Nichols tuning to get PID parameters
        print("\n=== Running Ziegler-Nichols tuning ===")
        REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM = ziegler_nichols_tuning(
            None, INITIAL_STATES, BASE_REWARDS, epochs=500, n_agents=N_AGENTS)  # Only tune with initial states
        
        print("\n=== Using PID parameters: ===")
        print(f"P: {REWARDS_ADAPTIVE_PARAM}")
        print(f"I: {REWARDS_INTEGRAL_PARAM}")
        print(f"D: {REWARDS_DERIVATIVE_PARAM}")
        
        # Option to override with default values if Ziegler-Nichols produces unsuitable parameters
        # REWARDS_ADAPTIVE_PARAM = 50000  # Override with default if needed
        # REWARDS_INTEGRAL_PARAM = 0
        # REWARDS_DERIVATIVE_PARAM = 0
        
        # Create initial agents with only initial states
        agents = generate_agents(N_AGENTS, INITIAL_STATES, state_rewards, STATE_RUN_COSTS, STATE_SWITCH_COSTS)
        agents_real_history = [[] for _ in range(NUM_ATTRIBUTES)]
        agents_declared_history = [[] for _ in range(NUM_ATTRIBUTES)]
        
        # Track states available at each epoch for proper diversity calculation
        states_available_at_epoch = []
        
        # Run the main experiment
        print("\n=== Starting main experiment ===")
        for epoch in range(epochs):
            # Add the new state at the specified epoch
            if epoch == NEW_STATE_EPOCH:
                print(f"\nAdding new state {NEW_STATE_NAME} at epoch {epoch}")
                POSSIBLE_STATES.append(NEW_STATE_NAME)
                
                # Reset the switch cost for the new state to normal level
                STATE_SWITCH_COSTS[NEW_STATE_NAME] = BASE_SWITCH_COST
                
                # Initialize the new state with normal reward based on new state count
                BASE_REWARDS_NEW = PER_NODE_REWARD_BASE * N_AGENTS / (len(POSSIBLE_STATES)-1)
                
                # Update agents to know about the new state
                for agent in agents:
                    # Update agent's possible states list
                    agent.possible_states = POSSIBLE_STATES.copy()
                    
                    # Set normal switch cost for the new state
                    agent.switch_cost[NEW_STATE_NAME] = random.uniform(0.0, 1.0) * agent.personal_switch_cost
                
                # Update state rewards to account for the new state
                for k in range(NUM_ATTRIBUTES):
                    # Set new state's reward to normal value
                    state_rewards[k][NEW_STATE_NAME] = BASE_REWARDS_NEW
                    
                    # Adjust all states' rewards proportionally
                    for state in POSSIBLE_STATES:
                        if state != 'NO_STATE':
                            # We've already set NEW_STATE_NAME appropriately above
                            if state != NEW_STATE_NAME:
                                state_rewards[k][state] = state_rewards[k][state] * (NUM_STATES-1) / (len(POSSIBLE_STATES)-1)
                
                # Print info about the new state
                print(f"New state added: {NEW_STATE_NAME}")
                print(f"Updated BASE_REWARDS: {BASE_REWARDS_NEW}")
                print(f"Total states now: {len(POSSIBLE_STATES)}")
                print(f"Switch cost for {NEW_STATE_NAME}: {STATE_SWITCH_COSTS[NEW_STATE_NAME]}")
            
            # Keep track of which states were available at this epoch (for proper diversity calculation)
            current_states = INITIAL_STATES.copy() if epoch < NEW_STATE_EPOCH else POSSIBLE_STATES.copy()
            states_available_at_epoch.append(current_states)
                
            # Update rewards based on current state distribution
            state_rewards, accumulated_error, last_error = update_state_rewards(
                agents, current_states, state_rewards, accumulated_error, last_error, 
                REWARDS_ADAPTIVE_PARAM, REWARDS_INTEGRAL_PARAM, REWARDS_DERIVATIVE_PARAM)
            
            state_rewards_last_epoch = distribute_rewards(agents, current_states, state_rewards)
            
            # Record states of all agents
            for k in range(NUM_ATTRIBUTES):
                agents_real_history[k].append([agent.real_state[k] for agent in agents])
                agents_declared_history[k].append([agent.declared_state[k] for agent in agents])
            
            # Update agent decisions
            for agent in agents:
                agent.decision(state_rewards_last_epoch, malicious=False)
                
            # Print status at key epochs
            if epoch % 100 == 0 or epoch == NEW_STATE_EPOCH or epoch == NEW_STATE_EPOCH + 1:
                # Count agents in each state for the first attribute
                state_counts = {state: 0 for state in current_states}
                for agent in agents:
                    state_counts[agent.declared_state[0]] += 1
                
                print(f"Epoch {epoch} - Agents per state: {state_counts}")
                
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
        
        # Extract the last 25% of diversity values
        for k in range(len(final_diversity)):
            last_25_percent_diversity[k] = [final_diversity[k][epoch] for epoch in range(epochs - last_25_percent, epochs)]
            
            # Flatten and calculate average
            flattened_diversity[k] = [item for sublist in last_25_percent_diversity[k] for item in sublist]
            average_last_25_percent[k] = sum(flattened_diversity[k]) / len(flattened_diversity[k])
            
            # Calculate ideal diversity for final set of states 
            ideal_diversity = get_ideal_diversity(POSSIBLE_STATES)
            ideal_diversity_all_experiments[k].append(ideal_diversity)
            
            # Calculate largest state metric
            largest_state = get_largest_state(agents_declared_history, range(epochs), POSSIBLE_STATES, N_AGENTS)
            last_25_percent_largest_state[k] = [largest_state[k][epoch] for epoch in range(epochs - last_25_percent, epochs)]
            flattened_largest_state[k] = [item for sublist in last_25_percent_largest_state[k] for item in sublist]
            average_last_25_percent_largest_state[k] = sum(flattened_largest_state[k]) / len(flattened_largest_state[k])
            largest_state_all_experiments[k].append(average_last_25_percent_largest_state[k])
    
    # Display results for single experiment case
    if num_experiments == 1:
        epochs_list = list(range(epochs))
        
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
        plt.title(f'Diversity with PID (P={REWARDS_ADAPTIVE_PARAM:.1f}, I={REWARDS_INTEGRAL_PARAM:.1f}, D={REWARDS_DERIVATIVE_PARAM:.1f})')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
        
        print("\n=== Final Results ===")
        print("Final states in system:", POSSIBLE_STATES)
        print("Rewards per state at the end:", state_rewards_last_epoch)
        print("Initial base rewards:", BASE_REWARDS)
        print(f"Final ideal diversity ({len(POSSIBLE_STATES)-1} states):", ideal_after)
        print(f"PID parameters used: P={REWARDS_ADAPTIVE_PARAM:.2f}, I={REWARDS_INTEGRAL_PARAM:.2f}, D={REWARDS_DERIVATIVE_PARAM:.2f}")
        
        # After your experiment runs:
        diversity_values = [final_diversity[0][epoch][0] for epoch in range(epochs)]
        ideal_before = get_ideal_diversity(INITIAL_STATES)
        ideal_after = get_ideal_diversity(POSSIBLE_STATES)

        # Print convergence analysis with better formatting
        print("\n" + "="*50)
        print("           SYSTEM CONVERGENCE ANALYSIS           ")
        print("="*50)

        print(f"\n💡 DIVERSITY TARGETS:")
        print(f"   • Initial phase (4 states): {ideal_before:.4f} bits")
        print(f"   • Final phase (5 states):   {ideal_after:.4f} bits")
        print(f"   • State transition occurred at epoch {NEW_STATE_EPOCH}")

        # Analyze convergence in both phases
        convergence_results = measure_dual_phase_convergence(
            diversity_values, NEW_STATE_EPOCH, ideal_before, ideal_after, 0.9)

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
            
            plt.title(f"Agent Distribution Over Time with PID Controller")
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()
        
if __name__ == "__main__":
    main()