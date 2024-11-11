ECG Structured Report and Embedding Generator with Alignment Training

Overview

This repository contains code designed to process ECG data from the PTB-XL dataset, extract SCP (Standard Communication Protocol) codes, generate structured reports with confidence scores, create embeddings using a pretrained BioClinicalBERT model, and align ECG embeddings with text embeddings using a training process to minimize alignment loss.
The goal of this project is to replicate part of a study's methodology on ECG analysis with a large language model (LLM), focusing on predicting heart failure risk by aligning ECG signals with structured text reports.

Purpose

The code provided here:
Extracts and interprets SCP codes associated with each ECG record from the PTB-XL dataset.
Constructs structured ECG reports by mapping SCP codes to categories and descriptions, including confidence scores for each code, which enhances interpretability.
Generates embeddings for each report using BioClinicalBERT, which captures the semantic content of each structured report in a high-dimensional vector space.
Creates ECG signal embeddings using a simple 1D CNN model (or a more advanced model if available).
Aligns ECG and text embeddings in a shared space by training the model to minimize the alignment loss, which quantifies the distance between ECG and text embeddings.
This approach allows for the integration of ECG signal data with medical language representations, making it suitable for applications such as predictive modeling, similarity analysis, and multimodal alignment tasks.

Functionality

1. Data Loading and Preparation
The code loads ptbxl_database.csv and scp_statements.csv files, which contain ECG records and their respective SCP code mappings.
SCP codes for each ECG record are extracted and parsed, along with their confidence scores.
2. Structured Report Generation
For each ECG record, the SCP codes are aligned with their categories and descriptions based on the scp_statements.csv file.
The structured report is created in a human-readable format, where each SCP code’s category, description, and confidence score appear on separate lines. This format provides interpretable information that could assist clinicians in understanding each ECG report.
3. Latent Text Embedding Generation
Using BioClinicalBERT, the code generates embeddings for each SCP code description in the structured report. These embeddings capture the meaning of each report section in a high-dimensional space.
Confidence scores are used to weight the embeddings before they are aggregated into a single, representative embedding for each ECG record.
4. ECG Embedding Generation
A simple 1D CNN model is used to generate embeddings for the raw ECG signals. This model captures essential features from the ECG data and produces a latent ECG embedding.
The ECG embedding (z_ecg) is projected to match the dimensionality of the text embedding using a learnable projection layer.
5. Alignment Training
To align the ECG and text embeddings, two projection layers are defined: one for the ECG embedding (z_ecg) and one for the text embedding (z_text).
The alignment loss is calculated as the cosine embedding distance between the projected ECG and text embeddings (e_ecg and e_text), following the formula provided in the study.
The model is trained by minimizing this alignment loss over multiple epochs, which helps the ECG and text embeddings align more closely in the shared space.
6. Output
The code outputs a structured report for each ECG record, including categories, descriptions, and confidence scores.
It also outputs an aggregated latent text embedding, representing the entire ECG report in a compact vector form.
After training, the alignment loss is displayed, showing how well the ECG and text embeddings align.

Example Output

For an ECG record with SCP codes like {'IMI': 15.0, 'LNGQT': 100.0, 'NST_': 100.0}, the output structured report would look like:
Myocardial Infarction: Inferior myocardial infarction, Confidence score: 15.0
Other ST-T descriptive statements: long QT-interval, Confidence score: 100.0
Basic roots for coding ST-T changes and abnormalities: non-specific ST changes, Confidence score: 100.0
Latent text code embedding: tensor([1.4137e-01, -3.3494e-02,  4.4636e-01,  2.5273e-0… 

Each line in the structured report includes the category, description, and confidence score for each SCP code.

Current Capabilities

This code successfully:
Processes and interprets SCP codes for each ECG record, mapping them to categories and descriptions.
Generates structured reports that are formatted for readability, including confidence scores.
Creates embeddings for each report using a pretrained BioClinicalBERT model, which enables future applications like classification and similarity analysis.
Generates ECG embeddings from raw ECG signals using a simple 1D CNN model.
Aligns ECG and text embeddings by training the model to minimize alignment loss, resulting in better alignment of the multimodal data in the shared space.

Training Process and Expected Results

Training Procedure

Initialize the Optimizer: The optimizer updates the weights of the ECG model and the projection layers.
Run Epochs: The training loop iterates over multiple epochs to improve alignment. For each batch:
Raw ECG data is processed to generate the ECG embedding (z_ecg), which is then projected into the shared space as e_ecg.
The structured report is processed to generate the text embedding (z_text), which is also projected as e_text.
The alignment loss is calculated as the cosine distance between e_ecg and e_text.
Gradients are computed and weights updated to minimize the alignment loss.
Monitor Loss: The average alignment loss is printed after each epoch, showing the progress of the training.

Expected Results

Initial Loss: At the start, the alignment loss is expected to be high (close to 1.0) due to randomly initialized projection layers.
Decreasing Loss: As training progresses, the loss should gradually decrease as the model learns to align ECG and text embeddings.
Final Loss: A well-aligned model should achieve an alignment loss closer to 0.0, indicating high similarity between the ECG and text embeddings in the shared space.

Challenges and Limitations

While this code achieves a portion of the study's goals, there are still limitations and challenges:
Translation and Text Refinement:
The PTB-XL dataset includes ECG reports in German and English. The study used a translation tool and language refinement for these reports. This code does not currently handle translation, which may impact non-English data. Users would need to add translation steps for full compatibility with multilingual data.
Limited Interpretability of Embeddings:
Interpreting high-dimensional embeddings directly is challenging. Additional interpretability tools would be required to provide insights into what each embedding dimension represents.
Model Pretraining and Fine-tuning:
The study also expands on pretraining on additional datasets and fine-tuning the model on specific patient subgroups for heart failure prediction. This code only leverages BioClinicalBERT embeddings without additional fine-tuning on an ECG-specific dataset.
Compute Requirements:
The training process is computationally intensive, especially for large datasets, and is optimized for GPU use. Training on a CPU might be slow.

Conclusion

This code provides a foundational toolset for generating structured reports, embeddings, and performing alignment training for ECG and text data from the PTB-XL dataset. While it effectively replicates part of the study’s methodology, there are several areas for future improvement. With additional enhancements, this code can serve as a robust starting point for further research and applications in ECG data analysis and predictive modeling.


