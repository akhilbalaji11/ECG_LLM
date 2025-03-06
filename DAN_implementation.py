#ECG Structured Report and Embedding Generator with Alignment Training
#Author Akhil Balaji 

import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from lifelines.utils import concordance_index

# Load datasets
path = 'ptbxl/'  # Path of the database folder
ptbxl_data = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
scp_statements = pd.read_csv(path + 'scp_statements.csv', index_col=0)  

# Prepare BioClinicalBERT model and tokenizer, load onto GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

# Enhanced ECG model with Dual Attention Network
class LeadAttentionModule(nn.Module):
    def __init__(self, dim):
        super(LeadAttentionModule, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads=4)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x shape: [batch, time, channels]
        x_norm = self.norm(x)
        attn_output, self.attn_weights = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_output
        x = x + self.ffn(self.norm2(x))
        return x

class TemporalAttentionModule(nn.Module):
    def __init__(self, dim):
        super(TemporalAttentionModule, self).__init__()
        # Changed to match the feature dimension after permute
        self.norm = nn.LayerNorm([1])
        # Changed to match the feature dimension
        self.attention = nn.MultiheadAttention(1, num_heads=1)
        self.ffn = nn.Sequential(
            nn.Linear(1, 4),  # Adjusted dimensions
            nn.GELU(),
            nn.Linear(4, 1)  # Adjusted dimensions
        )
        # Changed to match the feature dimension
        self.norm2 = nn.LayerNorm([1])
        
    def forward(self, x):
        # x shape: [batch, 1, time]
        x = x.permute(0, 2, 1)  # [batch, time, 1]
        x_norm = self.norm(x)
        attn_output, self.attn_weights = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_output
        x = x + self.ffn(self.norm2(x))
        x = x.permute(0, 2, 1)  # [batch, 1, time]
        return x

class ECGDualAttentionNetwork(nn.Module):
    def __init__(self, in_channels=12, feature_dim=128):
        super(ECGDualAttentionNetwork, self).__init__()
        # Initial convolution layers
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2)
        self.norm1 = nn.GroupNorm(1, 32)
        self.activation = nn.GELU()
        
        # Residual blocks with downsampling
        self.res_blocks = nn.ModuleList()
        channels = [32, 64, 128, 128, 128]
        for i in range(len(channels)-1):
            self.res_blocks.append(self._make_res_block(channels[i], channels[i+1], downsample=2))
        
        # Lead Attention module
        self.lead_attention = LeadAttentionModule(channels[-1])
        
        # Temporal Attention module (12 separate modules for each lead)
        self.temporal_attentions = nn.ModuleList([
            TemporalAttentionModule(channels[-1]) for _ in range(in_channels)
        ])
        
        # Projection layer to match dimensions
        self.projection = nn.Conv1d(channels[-1], in_channels, kernel_size=1)
        
        # Pooling and final projection
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_channels * 2, feature_dim)
    
    def _make_res_block(self, in_channels, out_channels, downsample=1):
        # Implementation of residual block with downsampling
        layers = []
        # Downsampling conv
        if downsample > 1:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                                   stride=downsample, padding=1))
        else:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.GroupNorm(1, out_channels))
        layers.append(nn.GELU())
        layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.GroupNorm(1, out_channels))
        
        # Skip connection
        skip = nn.Sequential()
        if in_channels != out_channels or downsample > 1:
            skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=downsample),
                nn.GroupNorm(1, out_channels)
            )
        
        return nn.ModuleList([nn.Sequential(*layers), skip])
    
    def forward(self, x):
        # x shape: [batch, 12, time]
        x = self.activation(self.norm1(self.conv1(x)))
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            identity = x
            x = res_block[0](x)
            identity = res_block[1](identity)
            x = self.activation(x + identity)
        
        # Store intermediate features for visualization
        self.intermediate_features = x
        
        # Reshape for lead attention
        batch_size, channels, time_len = x.shape
        lead_features = x.permute(0, 2, 1)  # [batch, time, channels]
        
        # Apply lead attention
        lead_attn_output = self.lead_attention(lead_features)
        
        # Project lead attention output to match temporal attention dimensions
        lead_attn_output = lead_attn_output.permute(0, 2, 1)  # [batch, channels, time]
        lead_attn_output = self.projection(lead_attn_output)  # [batch, 12, time]
        
        # Apply temporal attention for each lead
        temporal_outputs = []
        for i in range(12):
            lead_i = x[:, i:i+1, :]  # Select one lead
            temporal_outputs.append(self.temporal_attentions[i](lead_i))
        
        temporal_attn_output = torch.cat(temporal_outputs, dim=1)  # [batch, 12, time]
        
        # Combine outputs from both attention mechanisms
        combined = lead_attn_output + temporal_attn_output  # Now both are [batch, 12, time]
        
        # Pooling and final projection
        max_pooled = self.max_pool(combined).squeeze(-1)  # [batch, 12]
        avg_pooled = self.avg_pool(combined).squeeze(-1)  # [batch, 12]
        pooled = torch.cat([max_pooled, avg_pooled], dim=1)  # [batch, 24]
        pooled = self.dropout(pooled)
        
        return self.fc(pooled)  # [batch, feature_dim]
    
    def get_lead_attention_weights(self, x):
        # Forward pass to get attention weights
        _ = self.forward(x)
        return self.lead_attention.attn_weights
    
    def get_temporal_attention_weights(self, x, lead_idx=0):
        # Forward pass to get temporal attention weights
        _ = self.forward(x)
        return self.temporal_attentions[lead_idx].attn_weights

# ECG Decoder for signal reconstruction
class ECGDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=12, output_length=1000):
        super(ECGDecoder, self).__init__()
        self.output_length = output_length
        
        # Initial linear layer to expand latent code
        self.fc = nn.Linear(latent_dim, 128 * 32)
        
        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        channels = [128, 64, 32, 16, 12]  # Channel progression
        for i in range(len(channels)-1):
            self.upsample_blocks.append(self._make_upsample_block(channels[i], channels[i+1]))
        
        # Final convolution
        self.final_conv = nn.Conv1d(12, output_channels, kernel_size=3, padding=1)
        
    def _make_upsample_block(self, in_channels, out_channels):
        # Implementation of upsampling block
        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False))
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2))
        layers.append(nn.GroupNorm(1, out_channels))
        layers.append(nn.GELU())
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x is the latent code
        x = self.fc(x)
        x = x.view(-1, 128, 32)  # Reshape to [batch, channels, time]
        
        # Apply upsampling blocks
        for block in self.upsample_blocks:
            x = block(x)
        
        # Apply final convolution
        x = self.final_conv(x)
        
        # Ensure output has the correct length
        if x.shape[2] != self.output_length:
            x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=False)
        
        return x

# Risk Prediction Model
class RiskPredictionModel(nn.Module):
    def __init__(self, ecg_encoder, feature_dim=128):
        super(RiskPredictionModel, self).__init__()
        self.ecg_encoder = ecg_encoder
        
        # Risk prediction branch
        self.risk_branch = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.25),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        # Extract features from ECG
        features = self.ecg_encoder(x)
        
        # Predict risk score
        risk_score = self.risk_branch(features)
        
        return risk_score

# Initialize models
ecg_model = ECGDualAttentionNetwork(in_channels=12, feature_dim=128).to(device)
ecg_decoder = ECGDecoder(latent_dim=128, output_channels=12).to(device)

# Projection layers for alignment
projection_ecg = nn.Linear(128, 128).to(device)  # Adjust dimensions as needed
projection_text = nn.Linear(768, 128).to(device)  # BioClinicalBERT output is 768-dimensional

# Optimizer to train both the ECG embedding model and projection layers
optimizer = torch.optim.Adam(list(ecg_model.parameters()) + 
                            list(ecg_decoder.parameters()) +
                            list(projection_ecg.parameters()) + 
                            list(projection_text.parameters()), lr=1e-4)

# Function to load raw ECG data
def load_raw_data(df, sampling_rate, base_path):
    data = []
    for filename in df.filename_lr:
        full_path = base_path + filename  # Construct the full path for each ECG file
        record = wfdb.rdsamp(full_path)  # Read the ECG data
        data.append(record[0])  # Append the signal data, ignoring metadata
    return np.array(data)

# Function to retrieve SCP description and category
def get_scp_description_and_category(scp_code, scp_statements):
    if scp_code in scp_statements.index:
        category = scp_statements.loc[scp_code, 'Statement Category']
        description = scp_statements.loc[scp_code, 'SCP-ECG Statement Description']
        return category, description
    return None, None

# Function to create structured report
def create_structured_report(scp_codes, scp_statements):
    report = []
    for code, confidence in scp_codes.items():
        category, description = get_scp_description_and_category(code, scp_statements)
        if category and description:
            report.append(f"{category}: {description}, Confidence score: {confidence}")
    return '\n'.join(report)

# Function to generate text embedding from structured report
def get_batch_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
    return embeddings

# Function to aggregate text embeddings based on confidence scores
def aggregate_embeddings(scp_codes, scp_statements):
    texts = []
    confidences = []
    for code, confidence in scp_codes.items():
        category, description = get_scp_description_and_category(code, scp_statements)
        if category and description:
            texts.append(f"{category}: {description}")
            confidences.append(confidence)
    
    if texts:
        embeddings = get_batch_embeddings(texts).to(device)  # Move embeddings to device
        total_confidence = sum(confidences)
        weighted_embeddings = [(conf / total_confidence) * emb for conf, emb in zip(confidences, embeddings)]
        aggregated_embedding = torch.stack(weighted_embeddings).sum(dim=0).to(device)
        return aggregated_embedding
    return None

# Function to calculate alignment loss
def alignment_loss(e_ecg, e_text):
    return 1 - cosine_similarity(e_ecg, e_text).mean()

# Cox Proportional Hazards loss function
def cox_ph_loss(risk_scores, events, durations):
    """
    Implements the negative log partial likelihood for Cox PH model
    
    Args:
        risk_scores: Predicted risk scores
        events: Binary indicator of whether the event occurred (1) or not (0)
        durations: Time until event or censoring
    """
    # Sort by duration
    idx = torch.argsort(durations, descending=True)
    risk_scores = risk_scores[idx]
    events = events[idx]
    durations = durations[idx]
    
    # Calculate loss
    hazard_ratio = torch.exp(risk_scores)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
    uncensored_likelihood = risk_scores - log_risk
    
    # Apply mask for only uncensored instances
    censored_mask = (events == 1).float()
    neg_likelihood = -torch.sum(uncensored_likelihood * censored_mask) / (torch.sum(censored_mask) + 1e-8)
    
    return neg_likelihood

# Enhanced training function with reconstruction loss
def train_with_reconstruction(ecg_model, text_model, ecg_decoder, projection_ecg, projection_text, optimizer, df, epochs=5, batch_size=32):
    for epoch in range(epochs):
        total_loss = 0
        total_align_loss = 0
        total_recon_loss = 0
        num_batches = 0
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            if len(batch) == 0:
                continue
                
            num_batches += 1
            
            # Load and process raw ECG data
            ecg_data = load_raw_data(batch, sampling_rate=100, base_path=path)
            ecg_data = torch.tensor(ecg_data, dtype=torch.float32).to(device)
            ecg_data = ecg_data.permute(0, 2, 1)  # [batch, leads, time]
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass through ECG model
            z_ecg = ecg_model(ecg_data)
            
            # Reconstruct ECG signal
            reconstructed_ecg = ecg_decoder(z_ecg)
            
            # Calculate reconstruction loss
            recon_loss = F.mse_loss(reconstructed_ecg, ecg_data)
            
            # Process text embeddings for each record
            text_embeddings = []
            for _, row in batch.iterrows():
                # Generate text embedding from structured report
                scp_codes = ast.literal_eval(row['scp_codes'])
                z_text = aggregate_embeddings(scp_codes, scp_statements)
                if z_text is not None:
                    text_embeddings.append(z_text)
            
            if not text_embeddings:
                continue
                
            z_text = torch.stack(text_embeddings).to(device)
            
            # Project embeddings to common space
            e_ecg = projection_ecg(z_ecg[:len(text_embeddings)])
            e_text = projection_text(z_text)
            
            # Calculate alignment loss
            align_loss = alignment_loss(e_ecg, e_text)
            
            # Combined loss (weighted sum)
            loss = 0.7 * align_loss + 0.3 * recon_loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_align_loss += align_loss.item()
            total_recon_loss += recon_loss.item()
        
        # Average losses for the epoch
        avg_loss = total_loss / num_batches
        avg_align_loss = total_align_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Alignment: {avg_align_loss:.4f}, Reconstruction: {avg_recon_loss:.4f}")
    
    return ecg_model, ecg_decoder, projection_ecg, projection_text

# Function to finetune the model for risk prediction
def finetune_for_risk_prediction(model, ecg_decoder, optimizer, train_loader, epochs=100, alpha=0.5):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_risk_loss = 0
        total_recon_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            ecg_data, events, durations = batch
            ecg_data = ecg_data.to(device)
            events = events.to(device)
            durations = durations.to(device)
            
            num_batches += 1
            
            # Forward pass
            risk_scores = model(ecg_data)
            
            # Reconstruct ECG for regularization
            features = model.ecg_encoder(ecg_data)
            reconstructed_ecg = ecg_decoder(features)
            
            # Calculate losses
            recon_loss = F.mse_loss(reconstructed_ecg, ecg_data)
            risk_loss = cox_ph_loss(risk_scores, events, durations)
            
            # Combined loss
            loss = alpha * risk_loss + (1 - alpha) * recon_loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_risk_loss += risk_loss.item()
            total_recon_loss += recon_loss.item()
        
        if epoch % 10 == 0:
            avg_loss = total_loss / num_batches
            avg_risk_loss = total_risk_loss / num_batches
            avg_recon_loss = total_recon_loss / num_batches
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Risk: {avg_risk_loss:.4f}, Recon: {avg_recon_loss:.4f}")
    
    return model

# Function to evaluate model using C-index
def evaluate_cindex(model, test_loader):
    model.eval()
    all_risk_scores = []
    all_events = []
    all_durations = []
    
    with torch.no_grad():
        for batch in test_loader:
            ecg_data, events, durations = batch
            ecg_data = ecg_data.to(device)
            
            risk_scores = model(ecg_data)
            
            all_risk_scores.extend(risk_scores.cpu().numpy().flatten())
            all_events.extend(events.cpu().numpy().flatten())
            all_durations.extend(durations.cpu().numpy().flatten())
    
    # Calculate C-index
    c_index = concordance_index(all_durations, -np.array(all_risk_scores), all_events)
    
    return c_index

# Function for cross-validation of risk prediction model
def cross_validate_risk_model(ecg_model, ecg_decoder, data, events, durations, n_splits=2, n_repeats=5):
    """
    Perform repeated stratified k-fold cross-validation
    
    Args:
        ecg_model: The ECG encoder model
        ecg_decoder: The ECG decoder model
        data: ECG data
        events: Binary indicator of events
        durations: Time to event or censoring
        n_splits: Number of folds
        n_repeats: Number of repetitions
    """
    c_indices = []
    
    for repeat in range(n_repeats):
        # Create stratified folds based on event status
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(data, events)):
            print(f"Repeat {repeat+1}/{n_repeats}, Fold {fold+1}/{n_splits}")
            
            # Split data
            X_train, X_test = data[train_idx], data[test_idx]
            e_train, e_test = events[train_idx], events[test_idx]
            d_train, d_test = durations[train_idx], durations[test_idx]
            
            # Create data loaders
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                         torch.tensor(e_train, dtype=torch.float32), 
                                         torch.tensor(d_train, dtype=torch.float32))
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                        torch.tensor(e_test, dtype=torch.float32), 
                                        torch.tensor(d_test, dtype=torch.float32))
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32)
            
            # Initialize risk model with pretrained encoder
            risk_model = RiskPredictionModel(copy.deepcopy(ecg_model)).to(device)
            optimizer = torch.optim.Adam(risk_model.parameters(), lr=1e-4)
            
            # Finetune
            risk_model = finetune_for_risk_prediction(risk_model, ecg_decoder, optimizer, train_loader, epochs=50)
            
            # Evaluate
            c_index = evaluate_cindex(risk_model, test_loader)
            c_indices.append(c_index)
            print(f"C-index: {c_index:.4f}")
    
    # Calculate average and standard deviation
    mean_c_index = np.mean(c_indices)
    std_c_index = np.std(c_indices)
    
    print(f"Average C-index: {mean_c_index:.4f} ± {std_c_index:.4f}")
    
    return c_indices, mean_c_index, std_c_index

# Visualization functions
def visualize_lead_attention(model, ecg_data):
    """Visualize the lead attention matrix"""
    model.eval()
    with torch.no_grad():
        # Forward pass to get attention weights
        attention_weights = model.get_lead_attention_weights(ecg_data)
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights[0].cpu().numpy(), 
                   xticklabels=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
                   yticklabels=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
                   cmap='viridis')
        plt.title('Lead Attention Matrix')
        plt.tight_layout()
        plt.savefig('lead_attention.png')
        plt.close()

def visualize_temporal_attention(model, ecg_data, lead_idx=0):
    """Visualize the temporal attention for a specific lead"""
    model.eval()
    with torch.no_grad():
        # Forward pass to get temporal attention weights for the specified lead
        temporal_weights = model.get_temporal_attention_weights(ecg_data, lead_idx)
        
        # Plot the original ECG signal
        plt.figure(figsize=(15, 6))
        plt.subplot(2, 1, 1)
        plt.plot(ecg_data[0, lead_idx].cpu().numpy())
        plt.title(f'ECG Signal - Lead {lead_idx}')
        
        # Plot the attention weights
        plt.subplot(2, 1, 2)
        plt.plot(temporal_weights[0].cpu().numpy())
        plt.title(f'Temporal Attention Weights - Lead {lead_idx}')
        plt.tight_layout()
        plt.savefig(f'temporal_attention_lead_{lead_idx}.png')
        plt.close()

# Main execution
def main():
    # Run enhanced training with reconstruction loss
    print("Starting enhanced training with reconstruction loss...")
    trained_ecg_model, trained_decoder, trained_proj_ecg, trained_proj_text = train_with_reconstruction(
        ecg_model, text_model, ecg_decoder, projection_ecg, projection_text, 
        optimizer, ptbxl_data.head(500), epochs=5, batch_size=16
    )
    
    # Save the trained models
    torch.save({
        'ecg_model': trained_ecg_model.state_dict(),
        'ecg_decoder': trained_decoder.state_dict(),
        'projection_ecg': trained_proj_ecg.state_dict(),
        'projection_text': trained_proj_text.state_dict()
    }, 'ecg_llm_pretrained.pt')
    
    print("Pretraining completed and models saved.")
    
    # Example of processing a single record for visualization
    sample_batch = ptbxl_data.head(1)
    ecg_data = load_raw_data(sample_batch, sampling_rate=100, base_path=path)
    ecg_data = torch.tensor(ecg_data, dtype=torch.float32).to(device)
    ecg_data = ecg_data.permute(0, 2, 1)  # [batch, leads, time]
    
    # Visualize attention mechanisms
    print("Generating attention visualizations...")
    visualize_lead_attention(trained_ecg_model, ecg_data)
    visualize_temporal_attention(trained_ecg_model, ecg_data, lead_idx=0)
    
    print("Visualizations saved.")
    
    # Optional: Process all records to generate structured reports and embeddings
    
    print("Processing all records to generate structured reports and embeddings...")
    results = {}
    for ecg_id, row in ptbxl_data.iterrows():
        scp_codes = ast.literal_eval(row['scp_codes'])
        structured_report = create_structured_report(scp_codes, scp_statements)
        latent_text_code = aggregate_embeddings(scp_codes, scp_statements)
        
        if latent_text_code is not None:
            results[ecg_id] = {
                'structured_report': structured_report,
                'latent_text_code': latent_text_code.cpu().numpy()
            }
    
    # Save results
    np.save('ecg_llm_results.npy', results)
    print("Processing completed and results saved.")
    
    # Load the .npy file
    data = np.load('path/to/your/file.npy')

    print(data.shape)  # Print the dimensions of the array
    print(data)        # Print the array contents

# Run training for a few epochs on a small dataset for testing purposes
if __name__ == "__main__":
    main()