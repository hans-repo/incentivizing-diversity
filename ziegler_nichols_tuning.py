import numpy as np
import matplotlib.pyplot as plt
from agent import *

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
    Kp_values = np.logspace(2.8, 5, 1000)  # Test range from 1 to 10000
    Ku = 0  # Critical gain
    Tu = 0  # Critical period
    
    for Kp in Kp_values:
        print(f"Testing Kp = {Kp}")
        
        # Reset for this test
        state_rewards = [{state: base_rewards for state in possible_states} for _ in range(num_attributes)]
        accumulated_error = [{state: 0 for state in possible_states} for _ in range(num_attributes)]
        last_error = [{state: 0 for state in possible_states} for _ in range(num_attributes)]

        
        # Create fresh agents for each test
        test_run_costs = {state: 100 for state in possible_states}  # Simplified run costs
        test_switch_costs = {state: 100 for state in possible_states}  # Simplified switch costs
        switch_frequency_param = 5  # Default value for tuning phase
        test_agents = generate_agents(n_agents, possible_states, state_rewards, test_run_costs, test_switch_costs, switch_frequency_param)
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
    Kp = 0.6 * Ku
    Ki = 1.2 * Ku / Tu
    Kd = 0.075 * Ku * Tu
    
    # Alternative: Try less aggressive tuning if system is unstable
    # Kp = 0.2 * Ku
    # Ki = 0.4 * Ku / Tu
    # Kd = 0.025 * Ku * Tu
    
    print(f"Ziegler-Nichols tuning complete:")
    print(f"Critical Gain (Ku): {Ku}")
    print(f"Critical Period (Tu): {Tu}")
    print(f"Tuned parameters: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}")
    
    return Kp, Ki, Kd