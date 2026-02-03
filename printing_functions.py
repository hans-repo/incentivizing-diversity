# printing_functions.py - All printing/console output functions

import numpy as np
from measurement_functions import get_ideal_diversity

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

def print_new_version_added(new_version_name, epoch, num_versions, switch_cost):
    """Print information when a new version is added."""
    print(f"\nAdding new version {new_version_name} at epoch {epoch}")
    print(f"New version added")
    print(f"Total versions now: {num_versions}")
    print(f"Switch cost for {new_version_name}: {switch_cost}")

def print_epoch_status(epoch, version_counts, version_rewards_last_epoch, total_rewards, 
                     controller_type, reward_controller=None, diversity=None):
    """Print status at key epochs."""
    print(f"Epoch {epoch} - Agents per version: {version_counts}")
    print(f"Epoch {epoch} - Rewards per node per version: {version_rewards_last_epoch}")
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

def print_new_version_status(epoch, new_version_epoch, new_version_name, version_counts, version_rewards_last_epoch):
    """Print new version status for specific epochs after adding a new version."""
    if epoch >= new_version_epoch and epoch <= new_version_epoch + 10:
        print(f"Agents in new version {new_version_name}: {version_counts.get(new_version_name, 0)}")
        print(f"Current reward for {new_version_name}: {version_rewards_last_epoch[0].get(new_version_name, 0)}")

def print_single_experiment_results(POSSIBLE_versionS, version_rewards_last_epoch,
                                  controller_type, total_rewards_history, pid_params_history=None,
                                  reward_controller=None):
    """Print final results for a single experiment."""
    print("\n=== Final Results ===")
    print("Final versions in system:", POSSIBLE_versionS)
    print("Rewards per version at the end:", version_rewards_last_epoch)
    print(f"Final ideal diversity ({len(POSSIBLE_versionS)-1} versions):", get_ideal_diversity(POSSIBLE_versionS))
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

def print_convergence_analysis(diversity_values, NEW_version_EPOCH, INITIAL_versionS, POSSIBLE_versionS, convergence_results, add_new_version=True):
    """Print detailed convergence analysis."""
    ideal_before = get_ideal_diversity(INITIAL_versionS)
    ideal_after = get_ideal_diversity(POSSIBLE_versionS)

    # Print convergence analysis with better formatting
    print("\n" + "="*50)
    print("           SYSTEM CONVERGENCE ANALYSIS           ")
    print("="*50)

    print(f"\n💡 DIVERSITY TARGETS:")
    if add_new_version:
        print(f"   • Initial phase ({len(INITIAL_versionS)-1} versions): {ideal_before:.4f} bits")
        print(f"   • Final phase ({len(POSSIBLE_versionS)-1} versions):   {ideal_after:.4f} bits")
        print(f"   • version transition occurred at epoch {NEW_version_EPOCH}")
    else:
        print(f"   • Target diversity ({len(INITIAL_versionS)-1} versions): {ideal_before:.4f} bits")

    # Format Phase 1 results
    if add_new_version:
        print("\n📈 PHASE 1 CONVERGENCE (EPOCHS 0-{})".format(NEW_version_EPOCH-1))
    else:
        print("\n📈 CONVERGENCE ANALYSIS")
    
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

    # Format Phase 2 results only if new version was added
    if add_new_version:
        print("\n📉 PHASE 2 CONVERGENCE (EPOCHS {}-END)".format(NEW_version_EPOCH))
        
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

        # Format adaptation metrics only if new version was added
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

def print_reward_allocation_metrics(total_rewards_history, new_version_epoch=None):
    """Print reward allocation metrics."""
    print("\n" + "="*50)
    print("           REWARD ALLOCATION METRICS           ")
    print("="*50)
    print(f"   • Initial total rewards: {total_rewards_history[0]:.2f}")
    print(f"   • Final total rewards: {total_rewards_history[-1]:.2f}")
    print(f"   • Maximum total rewards: {max(total_rewards_history):.2f} at epoch {total_rewards_history.index(max(total_rewards_history))}")
    print(f"   • Minimum total rewards: {min(total_rewards_history):.2f} at epoch {total_rewards_history.index(min(total_rewards_history))}")
    print(f"   • Average total rewards: {sum(total_rewards_history)/len(total_rewards_history):.2f}")
    
    # Calculate the change in reward efficiency only if new version was added
    if new_version_epoch is not None:
        pre_change_rewards = total_rewards_history[:new_version_epoch]
        post_change_rewards = total_rewards_history[new_version_epoch:]
        avg_pre_change = sum(pre_change_rewards)/len(pre_change_rewards) if pre_change_rewards else 0
        avg_post_change = sum(post_change_rewards)/len(post_change_rewards) if post_change_rewards else 0
        
        print(f"   • Average pre-change rewards: {avg_pre_change:.2f}")
        print(f"   • Average post-change rewards: {avg_post_change:.2f}")
        if avg_pre_change > 0:
            print(f"   • Change in reward allocation: {((avg_post_change-avg_pre_change)/avg_pre_change)*100:.1f}%")
    print("\n" + "="*50)

def print_multi_experiment_header(num_experiments, controller_type):
    """Print header for multiple experiments summary."""
    print("\n\n" + "="*60)
    print(f"{'':^10}SUMMARY RESULTS FOR {num_experiments} EXPERIMENTS{'':^10}")
    print(f"{'':^10}CONTROLLER TYPE: {controller_type.upper()}{'':^10}")

def print_single_phase_convergence_times(x_axis, convergence_times):
    """Print convergence times for single phase experiments."""
    print("\nConvergence Time Results:")
    for i, time in enumerate(convergence_times):
        print(f"  Experiment {x_axis[i]}: {time} epochs")
    
    avg_time = sum(convergence_times) / len(convergence_times)
    print(f"  Average convergence time: {avg_time:.1f} epochs")

def print_comparison_summary_statistics(comparison_results, epochs, INITIAL_versionS, POSSIBLE_versionS):
    """
    Print summary statistics for all controllers and relative performance comparison.
    """
    ideal_after = get_ideal_diversity(POSSIBLE_versionS)
    
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
        
        # Access data from representative_experiment
        rep_exp = results['representative_experiment']
        
        # Calculate final diversity metrics
        diversity_values = [rep_exp['final_diversity'][0][epoch][0] for epoch in range(epochs)]
        final_diversity = diversity_values[-1]
        avg_diversity_last_25pct = np.mean(diversity_values[-int(epochs*0.25):])
        
        # Calculate reward metrics
        total_rewards = rep_exp['total_rewards_history']
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
            pid_params = rep_exp['pid_params']
            print(f"  • PID parameters: P={pid_params[0]:.1f}, I={pid_params[1]:.1f}, D={pid_params[2]:.1f}")
        elif controller_type == 'rl':
            final_pid_params = rep_exp['final_pid_params']
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
    print_convergence_adaptation_metrics(comparison_results, epochs, INITIAL_versionS, POSSIBLE_versionS, labels)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

def print_convergence_adaptation_metrics(comparison_results, epochs, INITIAL_versionS, POSSIBLE_versionS, labels):
    """
    Print convergence and adaptation metrics for all controllers.
    """
    ideal_before = get_ideal_diversity(INITIAL_versionS)
    ideal_after = get_ideal_diversity(POSSIBLE_versionS)
    
    # Extract NEW_version_EPOCH from results
    first_result = list(comparison_results.values())[0]
    rep_exp = first_result['representative_experiment']
    NEW_version_EPOCH = rep_exp.get('NEW_version_EPOCH')  # Use .get() in case it's None
    
    print("\n" + "="*80)
    print("CONVERGENCE AND ADAPTATION METRICS")
    print("="*80)
    
    for controller_type, results in comparison_results.items():
        print(f"\n{labels[controller_type]}:")
        print("-" * 40)
        
        # Access data from representative_experiment
        rep_exp = results['representative_experiment']
        diversity_values = [rep_exp['final_diversity'][0][epoch][0] for epoch in range(epochs)]
        
        # Phase 1 convergence (before new version or entire run if no new version)
        if NEW_version_EPOCH is not None:
            phase1_values = diversity_values[:NEW_version_EPOCH]
        else:
            phase1_values = diversity_values
            
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
        
        # Phase 2 adaptation (after new version) - only if new version was added
        if NEW_version_EPOCH is not None:
            phase2_values = diversity_values[NEW_version_EPOCH:]
            if len(phase2_values) > 0:
                phase2_target = ideal_after * 0.9  # 90% of new ideal
                phase2_convergence = None
                for i, val in enumerate(phase2_values):
                    if val >= phase2_target:
                        phase2_convergence = i
                        break
                
                if phase2_convergence is not None:
                    print(f"  • Phase 2 convergence time: {phase2_convergence} epochs after version addition")
                else:
                    print(f"  • Phase 2: Did not reach 90% of new ideal diversity")
                
                # Calculate adaptation shock (immediate impact of new version)
                if len(phase1_values) > 10 and len(phase2_values) > 10:
                    pre_change_avg = np.mean(phase1_values[-10:])  # Last 10 epochs before change
                    immediate_impact = phase2_values[0]  # First epoch after change
                    adaptation_shock = ((pre_change_avg - immediate_impact) / pre_change_avg) * 100
                    print(f"  • Adaptation shock: {adaptation_shock:.1f}% diversity drop")

def print_statistical_comparison_summary(comparison_results, controllers, num_experiments):
    """Print statistical summary comparing all controllers."""
    
    print("\n" + "="*80)
    print(f"{'':^20}STATISTICAL COMPARISON SUMMARY{'':^20}")
    print("="*80)
    
    print(f"\nNumber of experiments per controller: {num_experiments}")
    print(f"Controllers compared: {', '.join([c.upper() for c in controllers])}")
    
    # Diversity comparison
    print(f"\n{'DIVERSITY PERFORMANCE':^60}")
    print("-" * 60)
    print(f"{'Controller':<12} {'Mean (%)':<10} {'Std (%)':<10} {'Min (%)':<10} {'Max (%)':<10}")
    print("-" * 60)
    
    for controller in controllers:
        metrics = comparison_results[controller]['metrics']
        diversity_values = metrics['average_diversity_by_experiment']
        ideal_diversity = metrics['ideal_diversity']
        
        diversity_percentages = [(d/ideal_diversity)*100 for d in diversity_values]
        
        mean_div = np.mean(diversity_percentages)
        std_div = np.std(diversity_percentages)
        min_div = np.min(diversity_percentages)
        max_div = np.max(diversity_percentages)
        
        print(f"{controller.upper():<12} {mean_div:<10.1f} {std_div:<10.1f} {min_div:<10.1f} {max_div:<10.1f}")
    
    # Adaptation comparison - only if experiments included version addition
    first_controller_metrics = list(comparison_results.values())[0]['metrics']
    if 'adaptation_metrics' in first_controller_metrics and any(a > 0 for a in first_controller_metrics['adaptation_metrics']):
        print(f"\n{'ADAPTATION PERFORMANCE':^60}")
        print("-" * 60)
        print(f"{'Controller':<12} {'Mean (%)':<10} {'Std (%)':<10} {'Min (%)':<10} {'Max (%)':<10}")
        print("-" * 60)
        
        for controller in controllers:
            metrics = comparison_results[controller]['metrics']
            adaptation_values = [a*100 for a in metrics['adaptation_metrics']]
            
            mean_adapt = np.mean(adaptation_values)
            std_adapt = np.std(adaptation_values)
            min_adapt = np.min(adaptation_values)
            max_adapt = np.max(adaptation_values)
            
            print(f"{controller.upper():<12} {mean_adapt:<10.1f} {std_adapt:<10.1f} {min_adapt:<10.1f} {max_adapt:<10.1f}")
        
        # Recovery time comparison
        print(f"\n{'RECOVERY TIME PERFORMANCE':^60}")
        print("-" * 60)
        print(f"{'Controller':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 60)
        
        for controller in controllers:
            metrics = comparison_results[controller]['metrics']
            recovery_values = metrics['recovery_times']
            
            mean_recovery = np.mean(recovery_values)
            std_recovery = np.std(recovery_values)
            min_recovery = np.min(recovery_values)
            max_recovery = np.max(recovery_values)
            
            print(f"{controller.upper():<12} {mean_recovery:<10.1f} {std_recovery:<10.1f} {min_recovery:<10.1f} {max_recovery:<10.1f}")
    
    # Convergence comparison
    print(f"\n{'CONVERGENCE TIME PERFORMANCE':^80}")
    print("-" * 80)
    
    # Check if we have phase 2 data
    has_phase2 = 'phase2_convergence_times' in first_controller_metrics
    
    if has_phase2:
        print(f"{'Controller':<12} {'Phase 1 Mean':<15} {'Phase 1 Std':<15} {'Phase 2 Mean':<15} {'Phase 2 Std':<15}")
        print("-" * 80)
        
        for controller in controllers:
            metrics = comparison_results[controller]['metrics']
            phase1_values = metrics['phase1_convergence_times']
            phase2_values = metrics['phase2_convergence_times']
            
            mean_p1 = np.mean(phase1_values)
            std_p1 = np.std(phase1_values)
            mean_p2 = np.mean(phase2_values)
            std_p2 = np.std(phase2_values)
            
            print(f"{controller.upper():<12} {mean_p1:<15.1f} {std_p1:<15.1f} {mean_p2:<15.1f} {std_p2:<15.1f}")
    else:
        print(f"{'Controller':<12} {'Mean Time':<15} {'Std Time':<15}")
        print("-" * 45)
        
        for controller in controllers:
            metrics = comparison_results[controller]['metrics']
            phase1_values = metrics['phase1_convergence_times']
            
            mean_p1 = np.mean(phase1_values)
            std_p1 = np.std(phase1_values)
            
            print(f"{controller.upper():<12} {mean_p1:<15.1f} {std_p1:<15.1f}")
    
    # Best performer analysis
    print(f"\n{'BEST PERFORMER ANALYSIS':^60}")
    print("-" * 60)
    
    # Find best diversity performer
    best_diversity_controller = None
    best_diversity_score = 0
    
    for controller in controllers:
        metrics = comparison_results[controller]['metrics']
        diversity_values = metrics['average_diversity_by_experiment']
        ideal_diversity = metrics['ideal_diversity']
        
        mean_diversity_percentage = np.mean([(d/ideal_diversity)*100 for d in diversity_values])
        
        if mean_diversity_percentage > best_diversity_score:
            best_diversity_score = mean_diversity_percentage
            best_diversity_controller = controller
    
    print(f"Best Diversity Performance: {best_diversity_controller.upper()} ({best_diversity_score:.1f}% of ideal)")
    
    # Find best adaptation performer (only if adaptation metrics exist)
    if 'adaptation_metrics' in first_controller_metrics and any(a > 0 for a in first_controller_metrics['adaptation_metrics']):
        best_adaptation_controller = None
        best_adaptation_score = 0
        
        for controller in controllers:
            metrics = comparison_results[controller]['metrics']
            adaptation_values = [a*100 for a in metrics['adaptation_metrics']]
            mean_adaptation = np.mean(adaptation_values)
            
            if mean_adaptation > best_adaptation_score:
                best_adaptation_score = mean_adaptation
                best_adaptation_controller = controller
        
        print(f"Best Adaptation Performance: {best_adaptation_controller.upper()} ({best_adaptation_score:.1f}% adaptation quality)")
        
        # Find fastest recovery performer
        best_recovery_controller = None
        best_recovery_score = float('inf')
        
        for controller in controllers:
            metrics = comparison_results[controller]['metrics']
            recovery_values = metrics['recovery_times']
            mean_recovery = np.mean(recovery_values)
            
            if mean_recovery < best_recovery_score:
                best_recovery_score = mean_recovery
                best_recovery_controller = controller
        
        print(f"Fastest Recovery Performance: {best_recovery_controller.upper()} ({best_recovery_score:.1f} epochs)")
    
    print("\n" + "="*80)

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
        ("  Steady-version error (Phase 1)", 
         pid_metrics['phase1_convergence']['steady_version_error'],
         rl_metrics['phase1_convergence']['steady_version_error']),
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