import pandas as pd
import numpy as np
import wfdb
import ast
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold

# Load datasets
path = 'C:/Users/Akhil/ptbxl/'  # Path of the database folder
ptbxl_data = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
scp_statements = pd.read_csv(path + 'scp_statements.csv', index_col=0)

# Prepare BioClinicalBERT model and tokenizer, load onto GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

# ECG Encoder with Residual Blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        # Downsample layer to match dimensions if in_channels != out_channels
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Adjust identity if necessary
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        return self.relu(out)


class ECGEncoder(nn.Module):
    def __init__(self, in_channels=12, num_res_blocks=5, latent_dim=512):
        super(ECGEncoder, self).__init__()
        layers = []
        current_channels = in_channels
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(current_channels, 16))
            current_channels = 16  # Set current_channels to match out_channels
        self.res_blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(16 * 1024, latent_dim)

    def forward(self, x):
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


ecg_encoder = ECGEncoder().to(device)

# Decoder for ECG reconstruction
class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=512, output_channels=12):
        super(SimpleDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 16 * 1024)
        self.deconv1 = nn.ConvTranspose1d(16, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 16, -1)
        return self.deconv1(x)


decoder = SimpleDecoder().to(device)

# Risk Prediction MLP
class RiskPredictionMLP(nn.Module):
    def __init__(self, latent_dim=512):
        super(RiskPredictionMLP, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

risk_model = RiskPredictionMLP().to(device)

# StratifiedECGDataset Class
class StratifiedECGDataset(Dataset):
    def __init__(self, data, times, events):
        """
        Dataset class for ECG data with stratified sampling for censored/uncensored cases.

        Args:
            data: Preprocessed ECG signals (numpy array or tensor).
            times: Follow-up times for patients (numpy array or tensor).
            events: Event indicators (1 = event occurred, 0 = censored) (numpy array or tensor).
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.times = torch.tensor(times, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.times[idx], self.events[idx]


# Fine-Tuning Model
class FineTuningModel(nn.Module):
    def __init__(self, ecg_encoder, decoder, risk_model):
        super(FineTuningModel, self).__init__()
        self.ecg_encoder = ecg_encoder
        self.decoder = decoder
        self.risk_model = risk_model

    def forward(self, x):
        z_ecg = self.ecg_encoder(x)  # Latent ECG representation
        recon_x = self.decoder(z_ecg)  # Reconstructed ECG
        risk_score = self.risk_model(z_ecg)  # Risk score
        return recon_x, risk_score, z_ecg

finetuning_model = FineTuningModel(ecg_encoder, decoder, risk_model).to(device)

# Preprocessing function for ECG signals
def preprocess_ecg_signal(signal):
    """Z-score normalization and zero-padding."""
    signal = (signal - np.mean(signal)) / np.std(signal)  # Z-score normalization
    padded_signal = np.zeros((12, 1024))  # Zero-pad to length 1024
    padded_signal[:, :signal.shape[1]] = signal
    return padded_signal

# Function to load and preprocess raw ECG data
def load_and_preprocess_data(df, sampling_rate, base_path):
    data = []
    for filename in df.filename_lr:
        full_path = base_path + filename
        record = wfdb.rdsamp(full_path)
        signal = preprocess_ecg_signal(record[0].T)
        data.append(signal)
    return np.array(data)

# Loss functions
reconstruction_loss_fn = nn.MSELoss()

def risk_loss_fn(risk_scores, times, events):
    log_likelihood = 0
    num_uncensored = events.sum().item()
    for i in range(len(risk_scores)):
        if events[i] == 1:  # Only consider uncensored events
            risk_score_i = risk_scores[i]
            log_risk = torch.logsumexp(risk_scores[times >= times[i]], dim=0)
            log_likelihood += risk_score_i - log_risk
    return -log_likelihood / num_uncensored

def finetuning_loss(recon_x, x, risk_scores, times, events, alpha=0.5):
    l_recon = reconstruction_loss_fn(recon_x, x)
    l_risk = risk_loss_fn(risk_scores, times, events)
    print(f"Reconstruction Loss: {l_recon.item()}, Risk Loss: {l_risk.item()}")
    return alpha * l_recon + (1 - alpha) * l_risk

# Stratified DataLoader
def get_stratified_dataloader(data, times, events, batch_size):
    """
    Create a DataLoader for stratified sampling of censored and uncensored cases.

    Args:
        data: Preprocessed ECG signals (numpy array).
        times: Follow-up times (numpy array).
        events: Event indicators (1 = event occurred, 0 = censored) (numpy array).
        batch_size: Number of samples per batch.
    """
    uncensored_idx = np.where(events == 1)[0]
    censored_idx = np.where(events == 0)[0]

    uncensored_size = int(batch_size * len(uncensored_idx) / len(events))
    censored_size = batch_size - uncensored_size

    # Ensure stratified sampling
    uncensored_data = data[uncensored_idx]
    censored_data = data[censored_idx]
    uncensored_times = times[uncensored_idx]
    censored_times = times[censored_idx]
    uncensored_events = events[uncensored_idx]
    censored_events = events[censored_idx]

    # Combine sampled censored and uncensored data
    combined_data = np.concatenate((uncensored_data[:uncensored_size], censored_data[:censored_size]))
    combined_times = np.concatenate((uncensored_times[:uncensored_size], censored_times[:censored_size]))
    combined_events = np.concatenate((uncensored_events[:uncensored_size], censored_events[:censored_size]))

    # Create dataset and DataLoader
    dataset = StratifiedECGDataset(combined_data, combined_times, combined_events)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def calculate_c_index(risk_scores, times, events):
    concordant = 0
    discordant = 0
    n = len(risk_scores)

    for i in range(n):
        for j in range(i + 1, n):
            if events[i] == 1 or events[j] == 1:  # At least one is uncensored
                # Check concordant or discordant pairs
                if times[i] < times[j] and risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif times[i] > times[j] and risk_scores[i] < risk_scores[j]:
                    concordant += 1
                elif times[i] != times[j]:  # Only count discordant pairs if times differ
                    discordant += 1

    return concordant / (concordant + discordant) if (concordant + discordant) > 0 else 0.5

# def stratified_cross_validation(data, times, events, n_splits=2, epochs=5, batch_size=128):
#     """
#     Perform two-fold stratified cross-validation with repeated trials.

#     Args:
#         data: Preprocessed ECG signals (numpy array).
#         times: Follow-up times (numpy array).
#         events: Event indicators (numpy array).
#         n_splits: Number of folds (default = 2 for two-fold CV).
#         epochs: Number of epochs for training.
#         batch_size: Batch size for DataLoader.
#     """
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     c_indices = []

#     for fold, (train_idx, test_idx) in enumerate(skf.split(data, events)):
#         print(f"Fold {fold + 1}/{n_splits}")

#         # Split data
#         train_data, test_data = data[train_idx], data[test_idx]
#         train_times, test_times = times[train_idx], times[test_idx]
#         train_events, test_events = events[train_idx], events[test_idx]

#         # Train the model
#         train_finetuning_model(train_data, train_times, train_events, batch_size, epochs)

#         # Evaluate on the test set
#         test_risk_scores = []
#         test_dataloader = get_stratified_dataloader(test_data, test_times, test_events, batch_size)

#         for batch in test_dataloader:
#             x, times_batch, events_batch = batch
#             x = x.to(device)
#             times_batch = times_batch.to(device)
#             events_batch = events_batch.to(device)

#             # Forward pass
#             _, risk_scores, _ = finetuning_model(x)
#             test_risk_scores.append(risk_scores.detach().cpu())

#         test_risk_scores = torch.cat(test_risk_scores)
#         test_c_index = calculate_c_index(test_risk_scores, torch.tensor(test_times), torch.tensor(test_events))
#         c_indices.append(test_c_index)

#         print(f"Test C-Index for Fold {fold + 1}: {test_c_index:.4f}")

#     print(f"Average C-Index across folds: {np.mean(c_indices):.4f} ± {np.std(c_indices):.4f}")

# Fine-Tuning Training Function
def train_finetuning_model(data, times, events, batch_size, epochs=5, alpha=0.5):
    dataloader = get_stratified_dataloader(data, times, events, batch_size)
    optimizer = torch.optim.AdamW(finetuning_model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        total_loss = 0
        all_risk_scores = []
        all_times = []
        all_events = []

        for batch in dataloader:
            x, times_batch, events_batch = batch
            x = x.to(device)
            times_batch = times_batch.to(device)
            events_batch = events_batch.to(device)

            # Forward pass
            recon_x, risk_scores, _ = finetuning_model(x)

            # Compute loss
            loss = finetuning_loss(recon_x, x, risk_scores.squeeze(), times_batch, events_batch, alpha)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Collect risk scores and ground truth for C-index calculation
            all_risk_scores.append(risk_scores.detach().cpu())
            all_times.append(times_batch.cpu())
            all_events.append(events_batch.cpu())

        # Concatenate all batches for C-index calculation
        all_risk_scores = torch.cat(all_risk_scores)
        all_times = torch.cat(all_times)
        all_events = torch.cat(all_events)

        # Calculate C-index
        c_index = calculate_c_index(all_risk_scores, all_times, all_events)

        print(f"Epoch {epoch + 1}/{epochs}, Fine-Tuning Loss: {total_loss / len(dataloader)}, C-Index: {c_index:.4f}")

        
# Usage without UKB dataset (random)
data = load_and_preprocess_data(ptbxl_data, sampling_rate=500, base_path=path)
events = np.random.choice([0, 1], size=len(data))  # Example events
times = np.random.randint(1, 100, size=len(data))  # Example follow-up times

train_finetuning_model(data, times, events, batch_size=128, epochs=5)
