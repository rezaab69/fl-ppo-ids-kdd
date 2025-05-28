# fl-ppo-ids-kdd
Secure intrusion detection system using federated learning and reinforcement learning.

# Federated Learning for IDS using PPO with Geometric Median Aggregation

## Project Initialization Guide

### 1. Environment Setup

#### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

#### Recommended: Create a Virtual Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

### 2. Install Required Packages
Run the following command to install all necessary dependencies:

```bash
# On Windows or MacOS:
pip install torch numpy pandas scikit-learn
# On Unix:
pip install numpy pandas scikit-learn
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Project Structure
```
fl-ppo-ids-kdd/
├── main.py       # Python script version
└── KDDTrain+.txt              # Dataset file
```

### 4. Configuration Parameters

#### Federated Learning Parameters
- `num_clients`: Number of clients in the federation (default: 5)
- `federated_rounds`: Number of federated training rounds (default: 10)
- `local_iterations`: Number of local training iterations per round (default: 5)
- `global_lr`: Global learning rate (default: 3e-4)
- `Aggregation_algorithm`: Options: "geometric median", "fed_avg", "multi_krum", "bulyan", "coordinate_median", "adaptive_trimmed_mean", "best client" (default: "geometric median")

#### PPO Parameters
- `clip_epsilon`: PPO clipping parameter (default: 0.2)
- `ppo_epochs`: Number of PPO epochs (default: 4)
- `batch_size`: Training batch size (default: 64)
- `rollout_length`: Length of rollout buffer (default: 512)

#### Security Parameters
- `poisoned_clients_data`: List of clients with poisoned data
- `poisoned_clients_model`: List of clients with poisoned models
- `poisoned_clients_byzantine`: List of byzantine clients
- `weighted_with_history`: Whether to use historical accuracy for weighting (default: True)


#### Attack Configuration

##### Zero Attack Mode
- `zero_attack`: Boolean flag to enable/disable zero-attack mode
  - `True`: Attack type information is removed during training
  - `False`: Attack type information is available during training

##### Attack Type
- `attack_type`: Specifies which attack type to focus on (e.g., "neptune", "ipsweep", "back")

### 5. Running the Code

#### Option 1: Python Script
```bash
python main.py
```

### 6. Expected Output
- Training progress will be displayed in the console
- Model accuracies will be saved to CSV files
- Performance metrics will be logged for each federated round

### 7. Notes
- The code implements a federated learning framework with PPO for intrusion detection
- The geometric median aggregation is used to defend against poisoning attacks
- Model weights and training history are tracked throughout the federated learning process

### 8. Troubleshooting
- Ensure all required packages are installed with compatible versions
- Check that the dataset file `KDDTrain+.txt` is in the correct directory
