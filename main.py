import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict
import os
import glob

# *********************************************** #
# *********************************************** #
# initialize poisoned clients
poisoned_clients_data = []
poisoned_clients_model = []
poisoned_clients_byzantine = []
# initialize Aggregation algorithm
# fed_avg geometric median multi_krum bulyan coordinate_median adaptive_trimmed_mean best client
Aggregation_algorithm = "geometric median"
# True or False
weighted_with_history = True
# initialize zero attack
zero_attack = True
attack_type = "ipsweep"
# *********************************************** #
# *********************************************** #

# initialize federated parameters
global_lr = 3e-4
num_clients = 5
federated_rounds = 10
local_iterations = 5

# Initialize PPO parameters
# gamma = 0.99
clip_epsilon = 0.2
ppo_epochs = 4
batch_size = 64
rollout_length = 512

# Initialize weighting based on accuracy
clients_history = [0.0 for _ in range(num_clients)]
alpha = 0.5

# Initialize lists to store accuracies
accuracy_records = []

# Load and preprocess data
data = pd.read_csv("KDDTrain+.txt")

columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'attack', 'level'])

data.columns = columns

attack_n = []
for i in data.attack:
    if i == 'normal':
        attack_n.append("normal")
    else:
        attack_n.append("attack")

le1 = LabelEncoder()
for x in ['protocol_type', 'service', 'flag']:
    data[x] = le1.fit_transform(data[x])

le2 = LabelEncoder()
features = data.drop(columns=['attack'])
labels = le2.fit_transform(attack_n)

# Save original attack labels for evaluation
original_attack_labels = data['attack'].copy()

# If zero_attack is True, remove/mask 'attack' info from data before training
if zero_attack:
    features = data.drop(columns=['attack'])
else:
    features = data.drop(columns=['attack'])  # Default behavior (could be extended)

labels = le2.fit_transform(attack_n)

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(features, labels, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.long)
# For evaluation with attack_type data
original_attack_labels_train, original_attack_labels_test = train_test_split(original_attack_labels, test_size=0.2, random_state=42)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value


def partition_data(X, y, num_clients):
    dataset = torch.utils.data.TensorDataset(X, y)
    total_size = len(dataset)
    split_sizes = [total_size // num_clients for _ in range(num_clients)]
    remainder = total_size - sum(split_sizes)
    for i in range(remainder):
        split_sizes[i] += 1
    partitions = torch.utils.data.random_split(dataset, split_sizes)
    return partitions


def collect_trajectories(model, dataset, rollout_length):
    model.eval()
    states, actions, rewards, log_probs, values = [], [], [], [], []
    dataset_size = len(dataset)
    indices = np.random.choice(dataset_size, rollout_length, replace=True)
    for idx in indices:
        state, true_label = dataset[idx]
        state = state.to(device)
        true_label = true_label.item()
        with torch.no_grad():
            probs, value = model(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        reward = 1.0 if action.item() == true_label else -1.0
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value.item())
    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    log_probs = torch.stack(log_probs).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    returns = rewards
    advantages = returns - values
    return states, actions, log_probs, returns, advantages


def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages):
    model.train()
    dataset_size = states.size(0)
    for _ in range(ppo_epochs):
        indices = torch.randperm(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            mb_idx = indices[start:end]
            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_returns = returns[mb_idx]
            mb_advantages = advantages[mb_idx]
            probs, values = model(mb_states)
            m = torch.distributions.Categorical(probs)
            new_log_probs = m.log_prob(mb_actions)
            entropy = m.entropy().mean()
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), mb_returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def local_training(client_dataset, global_model, local_iterations, rollout_length):
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    optimizer_local = optim.Adam(local_model.parameters(), lr=global_lr)
    for _ in range(local_iterations):
        states, actions, log_probs, returns, advantages = collect_trajectories(local_model, client_dataset,
                                                                               rollout_length)
        ppo_update(local_model, optimizer_local, states, actions, log_probs, returns, advantages)

    # Calculate parameter updates (deltas) instead of returning full parameters
    global_params = global_model.state_dict()
    local_params = local_model.state_dict()

    # Compute the difference between local and global parameters
    param_updates = {}
    for key in global_params:
        param_updates[key] = local_params[key] - global_params[key]

    return param_updates


def apply_updates_to_global_model(global_model, updates):
    global_params = global_model.state_dict()
    updated_params = {}

    for key in global_params:
        updated_params[key] = global_params[key] + updates[key]

    global_model.load_state_dict(updated_params)
    return global_model


def poison_data(data, labels, target_label=0, poison_factor=0.1):
    poisoned_data = data.clone()
    poisoned_labels = labels.clone()
    num_poison = int(poison_factor * len(data))
    for i in range(num_poison):
        poisoned_data[i] = poisoned_data[i] + torch.randn_like(poisoned_data[i]) * 0.1
        poisoned_labels[i] = target_label
    return poisoned_data, poisoned_labels


def poison_model(model, poison_factor=0.1):
    model_params = model.state_dict()
    for key in model_params:
        noise = torch.randn_like(model_params[key]) * poison_factor
        model_params[key] = model_params[key] + noise
    model.load_state_dict(model_params)
    return model


def filter_clients_by_effective_accuracy(_clients_updates, _clients_accuracy, _clients_history, threshold):
    effective_weights = []
    filtered_clients_updates = []
    for idx, (current_acc, history_acc) in enumerate(zip(_clients_accuracy, _clients_history)):
        effective = current_acc * history_acc
        if effective >= threshold:
            effective_weights.append(effective)
            filtered_clients_updates.append(_clients_updates[idx])
    return filtered_clients_updates, effective_weights


def weighted_geometric_median(_global_state, filtered_clients_updates, effective_weights, eps=1e-10, max_iters=100):
    updates = {}
    for key in _global_state.keys():
        weights_tensor = torch.stack([client[key].float() for client in filtered_clients_updates])
        median = weights_tensor.mean(dim=0)
        for _ in range(max_iters):
            distances = torch.linalg.norm(weights_tensor - median, dim=tuple(range(1, weights_tensor.ndim))) + eps
            weights_sum = torch.sum(1.0 / distances)
            weighted_sum = torch.sum(weights_tensor / distances.view(-1, *[1] * (weights_tensor.ndim - 1)), dim=0)
            new_median = weighted_sum / weights_sum
            if torch.norm(new_median - median) < eps:
                break
            median = new_median

        total_effective = sum(effective_weights)
        updates[key] = sum(
            client[key] * effective / total_effective
            for client, effective in zip(filtered_clients_updates, effective_weights)
        )
    return updates


def geometric_median_custom_weighted_with_history(_global_state, _clients_updates, _clients_accuracy, _clients_history,
                                                  threshold=0.8, eps=1e-10, max_iters=100):
    with torch.no_grad():
        filtered_clients_updates, effective_weights = filter_clients_by_effective_accuracy(
            _clients_updates, _clients_accuracy, _clients_history, threshold
        )
        if not filtered_clients_updates:
            # Return zero updates if no clients pass the threshold
            return {key: torch.zeros_like(value) for key, value in _global_state.items()}
        return weighted_geometric_median(_global_state, filtered_clients_updates, effective_weights, eps, max_iters)


def fed_avg(param_updates_list):
    avg_updates = {}
    n = len(param_updates_list)
    for key in param_updates_list[0].keys():
        avg_updates[key] = sum([updates[key] for updates in param_updates_list]) / n
    return avg_updates


def multi_krum(param_updates_list, f):
    n = len(param_updates_list)
    m = n - f - 2  # number of closest distances to sum
    flat_updates = [torch.cat([v.flatten() for v in updates.values()]) for updates in param_updates_list]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = torch.norm(flat_updates[i] - flat_updates[j]).item() ** 2
            distances[i, j] = dist
            distances[j, i] = dist
    scores = []
    for i in range(n):
        dists = np.sort(distances[i])
        scores.append(np.sum(dists[1:m + 1]))  # skip self-distance at dists[0]
    selected_indices = np.argsort(scores)[:n - f]
    selected_updates = [param_updates_list[i] for i in selected_indices]
    agg_updates = {}
    for key in selected_updates[0].keys():
        agg_updates[key] = torch.stack([updates[key] for updates in selected_updates], dim=0).mean(dim=0)
    return agg_updates


def bulyan(param_updates_list, f):
    n = len(param_updates_list)
    m = n - 2 * f
    candidates = []
    selected = set()
    for _ in range(m):
        scores = []
        flat_updates = [torch.cat([v.flatten() for v in updates.values()]) for updates in param_updates_list]
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(flat_updates[i] - flat_updates[j]).item() ** 2
                distances[i, j] = dist
                distances[j, i] = dist
        for i in range(n):
            dists = np.sort(distances[i])
            scores.append(np.sum(dists[1:(n - f - 1) + 1]))
        for idx in selected:
            scores[idx] = float('inf')
        idx = int(np.argmin(scores))
        selected.add(idx)
        candidates.append(param_updates_list[idx])
    agg_updates = {}
    for key in candidates[0].keys():
        stacked = torch.stack([updates[key] for updates in candidates], dim=0)
        shape = stacked.shape
        flat = stacked.view(stacked.shape[0], -1)
        trimmed = []
        for i in range(flat.shape[1]):
            col = flat[:, i]
            sorted_col, _ = torch.sort(col)
            trimmed_col = sorted_col[f: -f] if flat.shape[0] > 2 * f else sorted_col
            trimmed.append(trimmed_col.mean().item())
        agg_updates[key] = torch.tensor(trimmed, dtype=stacked.dtype, device=stacked.device).view(shape[1:])
    return agg_updates


def coordinate_median(param_updates_list):
    agg_updates = {}
    for key in param_updates_list[0].keys():
        stacked = torch.stack([updates[key] for updates in param_updates_list], dim=0)
        shape = stacked.shape
        flat = stacked.view(stacked.shape[0], -1)
        median = flat.median(dim=0).values
        agg_updates[key] = median.view(shape[1:])
    return agg_updates


def adaptive_trimmed_mean(param_updates_list, trim_ratio=0.2):
    n = len(param_updates_list)
    trim_k = int(n * trim_ratio)
    agg_updates = {}
    for key in param_updates_list[0].keys():
        stacked = torch.stack([updates[key] for updates in param_updates_list], dim=0)
        shape = stacked.shape
        flat = stacked.view(stacked.shape[0], -1)
        trimmed_means = []
        for i in range(flat.shape[1]):
            col = flat[:, i]
            sorted_col, _ = torch.sort(col)
            trimmed_col = sorted_col[trim_k: n - trim_k] if n > 2 * trim_k else sorted_col
            trimmed_means.append(trimmed_col.mean().item())
        agg_updates[key] = torch.tensor(trimmed_means, dtype=stacked.dtype, device=stacked.device).view(shape[1:])
    return agg_updates


def evaluate_model(model):
    model.eval()
    correct = 0
    total = len(X_test)
    with torch.no_grad():
        for i in range(total):
            state = X_test[i].to(device)
            true_label = y_test[i].item()
            probs, _ = model(state)
            action = torch.argmax(probs).item()
            if action == true_label:
                correct += 1
    accuracy = correct / total
    return accuracy


def byzantine_attack(local_updates, byzantine_factor=10.0):
    # Byzantine attack: send completely random updates
    for key in local_updates:
        local_updates[key] = torch.randn_like(local_updates[key]) * byzantine_factor
    return local_updates


# Initialize parameters
input_dim = X_train.shape[1]
hidden_dim = 64
num_actions = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model = ActorCritic(input_dim, hidden_dim, num_actions).to(device)
global_model_raw = ActorCritic(input_dim, hidden_dim, num_actions).to(device)

client_datasets = partition_data(X_train, y_train, num_clients)

start_time = time.time()

for round in range(1, federated_rounds + 1):
    print(f"\n===== Federated Round {round} =====")
    local_updates_list = []
    local_accuracies = []

    # Record for this round
    round_record = {
        'round': round,
        'global_accuracy': 0.0
    }
    for i in range(num_clients):
        round_record[f'client_{i + 1}_accuracy'] = 0.0

    for client_id, client_dataset in enumerate(client_datasets, 1):
        print(f"\n--- Client {client_id} training ---")
        if client_id in poisoned_clients_data:
            indices = client_dataset.indices if hasattr(client_dataset, 'indices') else list(range(len(client_dataset)))
            data_list, labels_list = [], []
            for idx in indices:
                sample, lab = client_dataset.dataset[idx]
                data_list.append(sample)
                labels_list.append(lab)
            data_tensor = torch.stack(data_list)
            labels_tensor = torch.tensor(labels_list)
            poisoned_data, poisoned_labels = poison_data(data_tensor, labels_tensor, poison_factor=0.9)
            client_dataset = torch.utils.data.TensorDataset(poisoned_data, poisoned_labels)

        # Get parameter updates from local training
        local_updates = local_training(client_dataset, global_model, local_iterations, rollout_length)

        # Create a temporary model to evaluate with the local updates
        temp_model = copy.deepcopy(global_model)

        if client_id in poisoned_clients_model:
            # Apply model poisoning
            for key in local_updates:
                noise = torch.randn_like(local_updates[key]) * 0.9
                local_updates[key] = local_updates[key] + noise
        elif client_id in poisoned_clients_byzantine:
            # Apply Byzantine attack
            local_updates = byzantine_attack(local_updates)

        # Apply updates to temporary model for evaluation
        temp_params = {}
        for key in temp_model.state_dict():
            temp_params[key] = temp_model.state_dict()[key] + local_updates[key]
        temp_model.load_state_dict(temp_params)

        client_accuracy = evaluate_model(temp_model)
        print(f"Client {client_id} Test Accuracy: {client_accuracy:.5f}")
        local_updates_list.append(local_updates)
        local_accuracies.append(client_accuracy)

        # Store client accuracy
        round_record[f'client_{client_id}_accuracy'] = client_accuracy

        if clients_history[client_id - 1] == 0.0:
            clients_history[client_id - 1] = client_accuracy
        else:
            clients_history[client_id - 1] = alpha * client_accuracy + (1 - alpha) * clients_history[client_id - 1]

    if Aggregation_algorithm == "fed_avg":
        averaged_updates = fed_avg(local_updates_list)
        # Apply the averaged updates to the global model
        global_model = apply_updates_to_global_model(global_model, averaged_updates)
        global_accuracy = evaluate_model(global_model)
        print(f"\nGlobal fed_avg Model Accuracy after Round {round}: {global_accuracy:.5f}")
    elif Aggregation_algorithm == "geometric median":
        if weighted_with_history:
            threshold = 0.9
        else:
            threshold = 0
            clients_history = [1.0] * len(clients_history)
            local_accuracies = [1.0] * len(local_accuracies)
        global_state = global_model.state_dict()
        aggregated_updates = geometric_median_custom_weighted_with_history(global_state, local_updates_list,
                                                                           local_accuracies,
                                                                           clients_history, threshold=threshold)
        # Apply the aggregated updates to the global model
        global_model = apply_updates_to_global_model(global_model, aggregated_updates)
        global_accuracy = evaluate_model(global_model)
        print(f"\nGlobal geometric median Model Accuracy after Round {round}: {global_accuracy:.5f}")
    elif Aggregation_algorithm == "multi_krum":
        aggregated_updates = multi_krum(local_updates_list, f=1)
        # Apply the aggregated updates to the global model
        global_model = apply_updates_to_global_model(global_model, aggregated_updates)
        global_accuracy = evaluate_model(global_model)
        print(f"\nGlobal multi_krum Model Accuracy after Round {round}: {global_accuracy:.5f}")
    elif Aggregation_algorithm == "bulyan":
        aggregated_updates = bulyan(local_updates_list, f=1)
        # Apply the aggregated updates to the global model
        global_model = apply_updates_to_global_model(global_model, aggregated_updates)
        global_accuracy = evaluate_model(global_model)
        print(f"\nGlobal Bulyan Model Accuracy after Round {round}: {global_accuracy:.5f}")
    elif Aggregation_algorithm == "coordinate_median":
        aggregated_updates = coordinate_median(local_updates_list)
        # Apply the aggregated updates to the global model
        global_model = apply_updates_to_global_model(global_model, aggregated_updates)
        global_accuracy = evaluate_model(global_model)
        print(f"\nGlobal Coordinate-wise Median Model Accuracy after Round {round}: {global_accuracy:.5f}")
    elif Aggregation_algorithm == "adaptive_trimmed_mean":
        aggregated_updates = adaptive_trimmed_mean(local_updates_list, trim_ratio=0.2)
        # Apply the aggregated updates to the global model
        global_model = apply_updates_to_global_model(global_model, aggregated_updates)
        global_accuracy = evaluate_model(global_model)
        print(f"\nGlobal Adaptive Trimmed Mean Model Accuracy after Round {round}: {global_accuracy:.5f}")
    elif Aggregation_algorithm == "best client":
        best_client_index = np.argmax(local_accuracies)
        print(
            f"Best client selected: Client {best_client_index + 1} with accuracy {local_accuracies[best_client_index]:.5f}")
        best_updates = local_updates_list[best_client_index]
        # Apply the best client's updates to the global model
        global_model = apply_updates_to_global_model(global_model, best_updates)
        global_accuracy = evaluate_model(global_model)
        print(f"\nGlobal Best Client Model Accuracy after Round {round}: {global_accuracy:.5f}")
    else:
        global_accuracy = []

    # Store global accuracy
    round_record['global_accuracy'] = global_accuracy
    accuracy_records.append(round_record)

    # After model training, evaluate with attack_type data if zero_attack is True
    if zero_attack:
        # Use the trained model to predict on X_test
        global_model.eval()
        with torch.no_grad():
            outputs = global_model(X_test)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = torch.max(outputs, 1)
        # Calculate accuracy with original attack labels
        attack_label_encoder = LabelEncoder()
        true_attack_labels = attack_label_encoder.fit_transform(original_attack_labels_test)
        accuracy = (predicted.numpy() == true_attack_labels).mean()
        print(f"Model accuracy with attack_type data: {accuracy:.4f}")

# Find existing model_accuracies files
# existing_files = glob.glob('model_accuracies_*.csv')
# Extract numbers from filenames
# file_numbers = [int(f.replace('model_accuracies_', '').replace('.csv', ''))
#                 for f in existing_files if
#                 f.startswith('model_accuracies_') and f.replace('model_accuracies_', '').replace('.csv', '').isdigit()]
# Determine the next file number
# next_number = max(file_numbers + [0]) + 1 if file_numbers else 1
# Save to new file
# output_file = f'model_accuracies_{next_number}.csv'
# accuracy_df = pd.DataFrame(accuracy_records)
# accuracy_df.to_csv(output_file, index=False)
# print(f"Accuracy results saved to '{output_file}'")
