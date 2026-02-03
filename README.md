# Blockchain Fault Independence Simulation

A simulation framework for incentivizing client diversity in blockchain systems using adaptive reward mechanisms.

## Overview

This codebase accompanies the research paper "Incentivizing Fault Independence in Blockchains." It simulates diversity-aware incentive mechanisms that encourage validators to distribute evenly across client implementations, preventing single points of failure.

**Problem:** Blockchain networks like Ethereum suffer from client monoculture—if >33% of validators run the same client and it has a bug, the network can halt.

**Solution:** Adaptive reward mechanisms that increase rewards for underrepresented clients and decrease them for overrepresented ones, achieving even distribution without knowing operational costs.

## Controllers Implemented

| Controller | Description | Convergence | Reward Efficiency |
|------------|-------------|-------------|-------------------|
| **Ziegler-Nichols PID** | Classical control with automated tuning | ~400 epochs | Best (6-7× fewer rewards) |
| **RL-Adjusted PID** | DQN learns optimal PID parameters | ~1500 epochs | Poor |
| **Pure RL** | Direct reward allocation via DQN | ~1800 epochs | Poor |

## Installation

```bash
# Requirements
pip install numpy matplotlib torch
```

## Quick Start

```bash
# Run single experiment with PID controller
python multiple_experiments.py --controller pid --epochs 10000

# Run with RL-adjusted PID
python multiple_experiments.py --controller rl --epochs 10000

# Run with pure RL (no PID structure)
python multiple_experiments.py --controller rlnopid --epochs 10000

# Compare all controllers
python multiple_experiments.py --controller compare
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_agents` | 100 | Number of simulated validators |
| `--epochs` | 10000 | Simulation duration |
| `--num_experiments` | 20 | Runs for statistical averaging |
| `--num_states` | 4 | Number of client implementations |
| `--switch_frequency` | 0.5 | Agent switching reluctance (0-1) |
| `--tune_pid` | False | Enable Ziegler-Nichols auto-tuning |
| `--add_new_state` | False | Add new client mid-simulation |
| `--cost_factor` | 2 | Max/min running cost ratio |

## Project Structure

```
├── agent.py                    # Rational agent behavior model
├── multiple_experiments.py     # Main entry point
├── comparison_mode.py          # Side-by-side controller comparison
├── reinforcement_learning.py   # RL-adjusted PID (Double DQN)
├── reinforcement_learning_nopid.py  # Pure RL controller
├── ziegler_nichols_tuning.py   # Automated PID parameter tuning
├── measurement_functions.py    # Shannon entropy diversity metrics
├── plotting_functions.py       # Visualization utilities
└── printing_functions.py       # Console output helpers
```

## How It Works

1. **Agents** represent rational validators choosing clients to maximize `reward - cost`
2. **Diversity** is measured using Shannon Entropy: `H = -Σ(p_i × log₂(p_i))`
3. **Controllers** adjust per-client reward pools based on current distribution
4. **Target:** Even distribution (25% each for 4 clients) = maximum fault tolerance

## Key Results

- ZN-PID achieves **99% of ideal diversity** within 400 epochs
- ZN-PID uses **6-7× fewer rewards** than RL methods
- Pure RL handles extreme cost heterogeneity (>20×) better than PID
- All methods recover from dynamic state addition

## Citation

```bibtex
@article{faultindependence2025,
  title={Incentivizing Fault Independence in Blockchains},
  year={2025}
}
```

## License

Research code for academic purposes.