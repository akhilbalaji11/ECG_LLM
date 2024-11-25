ECG Structured Report, Embedding Generator, Alignment Training, and Fine-Tuning for Risk Prediction 

Overview
This repository contains code designed to process ECG data from the PTB-XL dataset, extract SCP (Standard Communication Protocol) codes, generate structured reports with confidence scores, create embeddings using a pretrained BioClinicalBERT model, align ECG embeddings with text embeddings, and fine-tune a model for heart failure risk prediction.
The program aims to replicate portions of a study's methodology for ECG analysis using a large language model (LLM), including:
Pretraining on ECG-report alignment tasks and signal reconstruction.
Fine-tuning for heart failure (HF) risk prediction using specific patient cohorts (e.g., hypertension and myocardial infarction).

Purpose
The code accomplishes the following:
Structured Report Generation:
Extracts SCP codes associated with each ECG record.
Constructs human-readable structured reports with categories, descriptions, and confidence scores.
Embedding Generation:
Generates embeddings for structured text reports using BioClinicalBERT.
Generates embeddings for ECG signals using a convolutional neural network (CNN).
Alignment Training:
Aligns ECG embeddings and text embeddings in a shared space by minimizing alignment loss.
Fine-Tuning for Risk Prediction:
Incorporates fine-tuning on two tasks: 
Signal Reconstruction: Reconstructs ECG signals from latent representations.
Risk Prediction: Predicts risk scores for heart failure based on follow-up time and event data.
This approach allows for the integration of ECG signal data with medical language representations, making it suitable for applications such as predictive modeling, similarity analysis, and multimodal alignment tasks.

Functionality
1. Data Loading and Preparation
PTB-XL Dataset:
Loads ECG records and SCP code mappings from ptbxl_database.csv and scp_statements.csv.
Extracts SCP codes and their confidence scores for each ECG record.
Preprocessing:
Applies z-score normalization to ECG signals and zero-pads them to a fixed length of 1024 samples for uniform input.
2. Structured Report Generation
SCP codes are mapped to their corresponding categories and descriptions.
Confidence scores are included in the report for interpretability.
Example report for SCP codes {‘IMI’: 15.0, ‘LNGQT’: 100.0}:
Myocardial Infarction: Inferior myocardial infarction, Confidence score: 15.0
Other ST-T descriptive statements: long QT-interval, Confidence score: 100.0
3. Embedding Generation
Text Embeddings:
Each report's sections are embedded using BioClinicalBERT.
Confidence scores weight the embeddings before aggregation into a single text embedding.
ECG Embeddings:
Raw ECG signals are encoded into latent embeddings using a CNN with residual blocks.
The encoder outputs a 512-dimensional latent representation (z_ecg).
4. Pretraining with Alignment Loss
Objective: Align ECG and text embeddings (e_ecg and e_text​) in a shared latent space.
Loss Function:
Uses cosine embedding distance as the alignment loss.
Projection layers map embeddings into the shared space before computing the loss.
Training:
The model minimizes alignment loss over multiple epochs to improve the similarity between ECG and text representations.
5. Fine-Tuning for Risk Prediction
Objective: Fine-tune the model to predict the risk of heart failure (HF) using ECG signals and follow-up event data.
Tasks:
Signal Reconstruction:
Reconstructs ECG signals from the latent representation (z_ecg) using a placeholder decoder.
Trained using Mean Squared Error (MSE) loss.
Risk Prediction:
Predicts HF risk scores using a simple multi-layer perceptron (MLP) based on z_ecg​.
Trained using a negative log partial likelihood loss function.
Fine-Tuning Loss:
L_finetuning = αL_recon + (1-α)L_risk 
The parameter α balances the contribution of reconstruction and risk prediction losses.
Batch Handling:
Uses stratified sampling to balance censored and uncensored cases during training.
6. Outputs
Structured Reports:
Provides human-readable descriptions and confidence scores for each SCP code.
Latent Representations:
Outputs 512-dimensional embeddings for ECG signals and text reports.
Risk Scores:
Predicts scalar risk scores for each patient based on their ECG signals.
Loss Metrics:
Displays the total fine-tuning loss, as well as L_recon and L_risk, over epochs.

Challenges and Limitations
UK Biobank Dataset:
The fine-tuning process described in the study requires HF follow-up data from the UK Biobank dataset (e.g., UKB-HYP and UKB-MI cohorts).
Without access to this dataset, random events and times are used for testing, which leads to inaccurate risk predictions.
Incomplete Fine-Tuning:
Due to the placeholder event and follow-up time data, the risk prediction results are not representative of real-world performance.
Scaling:
The current implementation is optimized for small-scale testing. Training on a full dataset requires more computational resources and optimization for batch sizes, epochs, and data augmentation.
Translation:
The PTB-XL dataset contains German and English ECG reports. Translation and language refinement are not implemented, which may limit interpretability for multilingual datasets.

Conclusion
This program provides a foundation for multimodal ECG analysis, including structured report generation, embedding alignment, and fine-tuning for risk prediction. However, the fine-tuning process is incomplete due to the lack of real-world data from the UK Biobank dataset. Future enhancements will focus on:
Incorporating real event and follow-up data for fine-tuning.
Scaling the model for larger datasets.
Adding visualization tools for reconstruction quality and risk prediction interpretability.




