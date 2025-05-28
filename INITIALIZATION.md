# Initialization Configuration Guide

This document explains the initialization parameters in the federated learning system and how to configure them according to your needs.

## Client Configuration

### Poisoned Clients
- `poisoned_clients_data`: List of client indices that will have poisoned data (e.g., `[1, 3, 5]`)
- `poisoned_clients_model`: List of client indices that will have poisoned models
- `poisoned_clients_byzantine`: List of client indices that will perform Byzantine attacks

Example:
```python
poisoned_clients_data = [1, 2]  # Clients 1 and 2 will have poisoned data
poisoned_clients_model = [3]    # Client 3 will have a poisoned model
poisoned_clients_byzantine = [4] # Client 4 will perform Byzantine attacks
```

## Aggregation Settings

### Aggregation Algorithm
- `Aggregation_algorithm`: Choose from the following options:
  - `"geometric median"`: Robust against outliers
  - `"fed_avg"`: Standard Federated Averaging
  - `"multi_krum"`: Krum algorithm for Byzantine-robust aggregation
  - `"bulyan"`: Bulyan algorithm for enhanced security
  - `"coordinate_median"`: Coordinate-wise median aggregation
  - `"adaptive_trimmed_mean"`: Trimmed mean with adaptive threshold

Example:
```python
Aggregation_algorithm = "geometric median"  # Uses geometric median for aggregation
```

### Weighting Options
- `weighted_with_history`: Boolean flag to enable/disable historical accuracy weighting
  - `True`: Weights clients based on their historical performance
  - `False`: Uses equal weighting for all clients

## Attack Configuration

### Zero Attack Mode
- `zero_attack`: Boolean flag to enable/disable zero-attack mode
  - `True`: Attack type information is removed during training
  - `False`: Attack type information is available during training

### Attack Type
- `attack_type`: Specifies which attack type to focus on (e.g., "neptune", "ipsweep", "back")

Example:
```python
zero_attack = True
attack_type = "neptune"  # Focus on neptune attack type
```

## Best Practices

1. **Security vs. Performance**: More robust aggregation methods (like Bulyan) are more secure but may reduce model performance.
2. **Poisoning**: When testing defense mechanisms, start with a small number of poisoned clients.
3. **Attack Types**: Ensure the specified attack type exists in your dataset.
4. **Weighting**: Use historical weighting for more stable training but be aware it may introduce bias.

## Default Configuration

```python
# Initialize poisoned clients
poisoned_clients_data = []
poisoned_clients_model = []
poisoned_clients_byzantine = []

# Aggregation settings
Aggregation_algorithm = "geometric median"
weighted_with_history = False

# Attack configuration
zero_attack = True
attack_type = "neptune"
```

For more information about specific algorithms or attack types, please refer to the project documentation or relevant research papers.
