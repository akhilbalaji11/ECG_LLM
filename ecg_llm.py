#ECG Structured Report and Embedding Generator with Alignment Training
#Author Akhil Balaji 

import pandas as pd
import numpy as np
import wfdb
import ast
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

# Load datasets
path = 'ptbxl/'  # Path of the database folder
ptbxl_data = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
scp_statements = pd.read_csv(path + 'scp_statements.csv', index_col=0)  

# Prepare BioClinicalBERT model and tokenizer, load onto GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

# Placeholder ECG embedding model (replace with a more sophisticated ECG model if available)
class SimpleECGEmbeddingModel(nn.Module):
    def __init__(self):
        super(SimpleECGEmbeddingModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=5)
        self.fc = nn.Linear(32, 128)  # Change to match the output dimension of text embedding projection

    def forward(self, x):
        x = self.conv1(x)
        x = torch.mean(x, dim=2)  # Mean pooling across time
        x = self.fc(x)
        return x

ecg_model = SimpleECGEmbeddingModel().to(device)

# Projection layers for alignment
projection_ecg = nn.Linear(128, 128).to(device)  # Adjust dimensions as needed
projection_text = nn.Linear(768, 128).to(device)  # BioClinicalBERT output is 768-dimensional

# Optimizer to train both the ECG embedding model and projection layers
optimizer = torch.optim.Adam(list(ecg_model.parameters()) + 
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

# Function to create a structured report
def create_structured_report(scp_codes, scp_statements):
    report = []
    for code, confidence in scp_codes.items():
        category, description = get_scp_description_and_category(code, scp_statements)
        if category and description:
            report.append(f"{category}: {description}, Confidence score: {confidence}")
    return '\n'.join(report)

# Function to generate text embedding from the structured report
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

# Training function to minimize alignment loss
def train_ecg_alignment_loss(df, epochs=5, batch_size=32):
    for epoch in range(epochs):
        total_loss = 0
        num_batches = len(df) // batch_size

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]

            ecg_embeddings = []
            text_embeddings = []

            # Reset gradients
            optimizer.zero_grad()
            
            for _, row in batch.iterrows():
                # Load and process raw ECG data
                ecg_data = load_raw_data(batch, sampling_rate=100, base_path=path)
                ecg_data = torch.tensor(ecg_data, dtype=torch.float32).permute(0, 2, 1).to(device)  # Move to device
                
                # Generate ECG embedding
                z_ecg = ecg_model(ecg_data)
                e_ecg = projection_ecg(z_ecg).to(device)
                ecg_embeddings.append(e_ecg)

                # Generate text embedding from a structured report
                scp_codes = ast.literal_eval(row['scp_codes'])
                structured_report = create_structured_report(scp_codes, scp_statements)
                z_text = aggregate_embeddings(scp_codes, scp_statements).to(device)  # Move to device
                e_text = projection_text(z_text).to(device)
                text_embeddings.append(e_text)

            # Stack embeddings and calculate batch alignment loss
            e_ecg = torch.stack(ecg_embeddings).to(device)
            e_text = torch.stack(text_embeddings).to(device)
            batch_loss = alignment_loss(e_ecg, e_text)

            # Backpropagation and optimization step
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()

        # Average alignment loss for the epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Alignment Loss: {avg_loss}")

# Run training for a few epochs on a small dataset for testing purposes
train_ecg_alignment_loss(ptbxl_data.head(500), epochs=5)


""" # Process each record, we can also limit the number for testing purposes, e.g., ptbxl_data.head(100)
results = {}
for ecg_id, row in ptbxl_data.iterrows():
    scp_codes = ast.literal_eval(row['scp_codes'])
    structured_report = create_structured_report(scp_codes, scp_statements)
    latent_text_code = aggregate_embeddings(scp_codes, scp_statements)
    results[ecg_id] = {
        'structured_report': structured_report,
        'latent_text_code': latent_text_code
    }

# Example output for the first record
example_ecg_id = list(results.keys())[0]
print("Example structured report:", results[example_ecg_id]['structured_report'])
print("Example latent text code embedding:", results[example_ecg_id]['latent_text_code']) """
