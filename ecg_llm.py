import pandas as pd
import ast
from transformers import AutoTokenizer, AutoModel
import torch

# Load datasets
path = 'ptbxl/'  # path of the database folder
ptbxl_data = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
scp_statements = pd.read_csv(path + 'scp_statements.csv', index_col=0)  

# Prepare BioClinicalBERT model and tokenizer, load onto GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

def get_scp_description_and_category(scp_code, scp_statements):
    """Retrieve the statement category and SCP-ECG description for a given SCP code."""
    if scp_code in scp_statements.index:
        category = scp_statements.loc[scp_code, 'Statement Category']
        description = scp_statements.loc[scp_code, 'SCP-ECG Statement Description']
        return category, description
    return None, None

def create_structured_report(scp_codes, scp_statements):
    """Create structured report based on SCP codes and their descriptions with categories."""
    report = []
    for code, confidence in scp_codes.items():
        category, description = get_scp_description_and_category(code, scp_statements)
        if category and description:
            report.append(f"{category}: {description}, Confidence score: {confidence}")
    return '\n'.join(report)

def get_batch_embeddings(texts):
    """Generate embeddings for a batch of texts using BioClinicalBERT with GPU support."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu()  # Move embeddings back to CPU after processing
    return embeddings

def aggregate_embeddings(scp_codes, scp_statements):
    """Aggregate embeddings based on confidence scores with batch processing."""
    texts = []
    confidences = []
    for code, confidence in scp_codes.items():
        category, description = get_scp_description_and_category(code, scp_statements)
        if category and description:
            texts.append(f"{category}: {description}")
            confidences.append(confidence)
    
    if texts:
        embeddings = get_batch_embeddings(texts)
        total_confidence = sum(confidences)
        weighted_embeddings = [(conf / total_confidence) * emb for conf, emb in zip(confidences, embeddings)]
        aggregated_embedding = torch.stack(weighted_embeddings).sum(dim=0)
        return aggregated_embedding
    return None

# Process each record, we can also limit the number for testing purposes, e.g., ptbxl_data.head(100)
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
print("Example latent text code embedding:", results[example_ecg_id]['latent_text_code'])
