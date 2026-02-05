# Blockchain Fault Independence Simulation

Simulation code for adaptive reward mechanisms that incentivize client diversity in blockchain systems.

## Installation

```bash
pip install numpy matplotlib torch
```

## Running Experiments

### Basic Usage

Run a single controller experiment:

```bash
python multiple_experiments.py --controller pid --epochs 10000
python multiple_experiments.py --controller rl --epochs 10000
python multiple_experiments.py --controller rlnopid --epochs 10000
```

Compare controllers side-by-side:

```bash
python multiple_experiments.py --controller compare --num_experiments 5
python multiple_experiments.py --controller compare --compare_modes pid rl
```

Test dynamic version addition (a new client appears at 5% of total epochs):

```bash
python multiple_experiments.py --controller pid --add_new_version --epochs 10000
```

### Command Line Parameters

**Simulation settings:**
- `--n_agents`: Number of validators (default: 544)
- `--epochs`: Simulation length (default: 10000)
- `--num_experiments`: Number of runs for averaging (default: 20)

**Controller selection:**
- `--controller`: Choose `pid`, `rl`, `rlnopid`, or `compare`
- `--compare_modes`: When using compare, specify which controllers (e.g., `--compare_modes pid rl`)

**PID tuning:**
- `--tune_pid`: Enable Ziegler-Nichols auto-tuning (default: True)
- `--tune_once`: Tune only on first experiment and reuse (default: True)
- `--p_param`, `--i_param`, `--d_param`: Manual PID gains if not using auto-tuning

**RL parameters:**
- `--epsilon`: Initial exploration rate (default: 1.0)
- `--epsilon_decay`: Exploration decay (default: 0.9995)
- `--learning_rate`: Neural network learning rate (default: 0.1)
- `--action_scale`: How much RL adjusts PID parameters (default: 10)

**Dynamic scenarios:**
- `--add_new_version`: Add a new client version mid-simulation

## Configuring Parameters in Code

Several parameters must be set by editing the source files directly.

### Number of Versions

Edit the `INITIAL_versionS` list in `multiple_experiments.py` (around line 131) or `comparison_mode.py` (around line 214):

```python
# 5 versions (4 clients + NO_version for opt-out)
INITIAL_versionS = ['NO_version', 'version_A', 'version_B', 'version_C', 'version_D']

# 8 versions
INITIAL_versionS = ['NO_version', 'version_A', 'version_B', 'version_C', 'version_D', 
                    'version_E', 'version_F', 'version_G']
```

The `NO_version` entry represents validators who opt out of the diversity rewards.

### Cost Configuration

Set running costs in `multiple_experiments.py` (lines 153-155) or `comparison_mode.py` (lines 236-239):

```python
BASE_RUN_COST = 100
RUN_COST_CEILING = BASE_RUN_COST + round(2*BASE_RUN_COST)  # Creates 3× cost range
BASE_SWITCH_COST = BASE_RUN_COST
```

Costs are distributed linearly across versions from `BASE_RUN_COST` to `RUN_COST_CEILING`.

### Agent Switching Behavior

The `SWITCH_FREQUENCY_PARAM` controls how reluctant agents are to change versions. Set it in `multiple_experiments.py` (line 174) or `comparison_mode.py` (line 260):

```python
SWITCH_FREQUENCY_PARAM = 2  # Higher = more reluctant to switch
```

Each agent also has a random switch cooldown (1-10 epochs) set in `agent.py` line 27.

## Varying Parameters Across Experiments

When running multiple experiments, you can vary parameters by incorporating the experiment index into calculations.

In `multiple_experiments.py`, the loop variable is `i` (ranging from 0 to `num_experiments - 1`).
In `comparison_mode.py`, it's `experiment_number`.

### Example: Increasing Cost Gap Per Experiment

To test how controllers handle growing cost heterogeneity, modify line 154 in `multiple_experiments.py`:

```python
# Original (fixed 3× cost range)
RUN_COST_CEILING = BASE_RUN_COST + round(2*BASE_RUN_COST)

# Modified (cost gap grows by 20× each experiment)
RUN_COST_CEILING = BASE_RUN_COST + round(i*20*BASE_RUN_COST)
```

With `--num_experiments 5`, this creates experiments with cost gaps of 1×, 21×, 41×, 61×, and 81×.

### Example: Varying Switch Frequency

Modify line 174 in `multiple_experiments.py`:

```python
# Original
SWITCH_FREQUENCY_PARAM = 2

# Modified (agents become more reluctant each experiment)
SWITCH_FREQUENCY_PARAM = 0.5 * i
```

### Example: Combined Iteration

You can combine multiple parameter variations:

```python
RUN_COST_CEILING = BASE_RUN_COST + round(i*10*BASE_RUN_COST)
SWITCH_FREQUENCY_PARAM = 0.5 + 0.25*i
```

Remember that experiment index `i` starts at 0, so the first experiment uses `i=0`.

## Customizing Plot Labels for Parameter Iteration

When you iterate parameters across experiments, the default plot labels will say "Experiment Number" on the x-axis, which doesn't reflect what you're actually varying. You need to update both the x-axis values and the labels in the plotting functions.

### Setting X-Axis Values

The x-axis values are set in `multiple_experiments.py` at line 176:

```python
x_axis.append(i+1)  # Default: experiment numbers 1, 2, 3, ...
```

To show the actual parameter values, change this to match your iteration. For example, if you're varying cost gap by 20× per experiment:

```python
x_axis.append(i*20)  # Shows: 0, 20, 40, 60, ...
```

Note: The `plot_average_diversity` function (line 452 in `plotting_functions.py`) overrides x_axis internally with its own calculation. If you're using that plot, modify line 452 instead:

```python
# Original
x_axis = np.arange(0, len(average_diversity_by_experiment[k])) * 10

# Modified for 20× cost gap increments
x_axis = np.arange(0, len(average_diversity_by_experiment[k])) * 20
```

### Updating Axis Labels and Titles

The plotting functions in `plotting_functions.py` have hardcoded labels. Find and update the `plt.xlabel()` and `plt.title()` calls in the functions you use.

**Key functions and their label locations:**

`plot_average_diversity` (line 448):
- Line 454: `plt.xlabel('Cost factor between cheapest and most expensive version ')`
- Line 456: `plt.title(f'Diversity Across Experiments with {controller_type.upper()}')`

`plot_recovery_times` (line 313):
- Line 318: `plt.xlabel('Experiment Number')`
- Line 320: `plt.title('System Recovery Time Across Experiments')`

`plot_convergence_times` (line 333):
- Line 340: `plt.xlabel('Experiment Number')`
- Line 342: `plt.title('System Convergence Times Across Experiments')`

`plot_single_phase_convergence_times` (line 359):
- Line 363: `plt.xlabel('Experiment Number')`
- Line 365: `plt.title('System Convergence Times Across Experiments')`

`plot_adaptation_quality` (line 427):
- Line 432: `plt.xlabel('Experiment Number')`
- Line 434: `plt.title(f'System Adaptation Quality Across Experiments...')`

`plot_pid_parameters_across_experiments` (line 378):
- Line 388: `plt.xlabel('Experiment Number')`
- Line 407: `plt.suptitle('PID Parameters Across Experiments')`

`plot_largest_version_share` (line 411):
- Line 417: `plt.xlabel('Experiment Number')`
- Line 419: `plt.title('Largest version Metric Across Experiments')`

### Example: Labeling for Cost Gap Iteration

If you're iterating cost gap by 20× per experiment, update `plotting_functions.py`:

```python
# In plot_recovery_times (around line 318)
plt.xlabel('Cost Gap (×)')
plt.title('System Recovery Time vs Cost Heterogeneity')

# In plot_convergence_times (around line 340)
plt.xlabel('Cost Gap (×)')
plt.title('Convergence Time vs Cost Heterogeneity')
```

### Example: Labeling for Switch Frequency Iteration

If you're iterating switch frequency:

```python
# In multiple_experiments.py line 176
x_axis.append(0.5 * i)  # Match your SWITCH_FREQUENCY_PARAM = 0.5 * i

# In plotting_functions.py
plt.xlabel('Switch Frequency Parameter')
plt.title('Convergence Time vs Agent Switching Reluctance')
```

## Project Structure

**Main entry points:**
- `multiple_experiments.py` - Primary experiment runner
- `comparison_mode.py` - Side-by-side controller comparison
- `single_experiment.py` - Simple single-run script

**Controllers:**
- `reinforcement_learning.py` - PID controller and RL-adjusted PID (Double DQN)
- `reinforcement_learning_nopid.py` - Pure RL controller without PID structure
- `ziegler_nichols_tuning.py` - Automated PID parameter tuning

**Supporting modules:**
- `agent.py` - Rational validator behavior model
- `measurement_functions.py` - Shannon entropy diversity calculation
- `plotting_functions.py` - Visualization
- `printing_functions.py` - Console output

## Understanding the Output

The simulation measures diversity using Shannon Entropy. For a system with `|V|` versions (excluding NO_version), ideal diversity is `log₂(|V|)` bits, achieved when validators distribute evenly.

Key metrics printed during runs:
- **Diversity**: Current Shannon entropy vs. ideal
- **Convergence time**: Epochs to reach 99% of ideal diversity
- **Total rewards**: Cumulative rewards allocated (lower is more efficient)
- **Version distribution**: Percentage of validators on each client

Plots generated include diversity over time, reward allocation per version, and agent distribution.
