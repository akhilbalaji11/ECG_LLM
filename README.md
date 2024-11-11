ECG Structured Report and Embedding Generator
Overview
This repository contains code designed to process ECG data from the PTB-XL dataset, extract SCP (Standard Communication Protocol) codes, generate structured reports with confidence scores, and create embeddings using a pretrained BioClinicalBERT model. The goal is to replicate part of the methodology used in a study on ECG analysis with a large language model (LLM) to predict heart failure risk.
Purpose
The code provided here:
Extracts and interprets SCP codes associated with each ECG record from the PTB-XL dataset.
Constructs structured ECG reports by mapping SCP codes to categories and descriptions, with each report line including a confidence score for clarity and interpretability.
Generates embeddings for each structured report using BioClinicalBERT, which represents the semantic content of each report in a high-dimensional vector space. These embeddings can later be used in downstream machine learning models for tasks like classification or prediction.
This approach allows for the integration of ECG signal data with medical language representations, making it suitable for various applications, such as predictive modeling and similarity analysis.
Functionality
1. Data Loading and Preparation
The code loads the ptbxl_database.csv and scp_statements.csv files, which contain ECG records and their respective SCP code mappings.
SCP codes for each ECG record are extracted and parsed, along with their confidence scores.
2. Structured Report Generation
For each ECG record, the SCP codes are aligned with their categories and descriptions based on the scp_statements.csv file.
The structured report is created in a human-readable format, where each SCP code’s category, description, and confidence score appear on separate lines. This format provides interpretable information that could assist clinicians in understanding each ECG report.
3. Embedding Generation with BioClinicalBERT
Using BioClinicalBERT, the code generates embeddings for each SCP code description in the structured report. These embeddings capture the meaning of each report section in a high-dimensional space.
Confidence scores are used to weight the embeddings before they are aggregated into a single, representative embedding for each ECG record.
4. Output
The code outputs a structured report for each ECG record, including categories, descriptions, and confidence scores.
It also outputs an aggregated latent text embedding, representing the entire ECG report in a compact vector form.
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
Creates embeddings for each report using a pre-trained BioClinicalBERT model, which enables future applications like classification and similarity analysis.
Challenges and Limitations
While this code achieves a portion of the study's goals, there are several limitations and challenges:
Translation and Text Refinement:
The original PTB-XL dataset includes ECG reports in German and English. The study used a machine translation tool and ChatGPT to ensure accurate translation and refinement of these reports. However, this code does not currently handle translation, which may impact non-English data. Users would need to add translation steps for full compatibility with multilingual data.
Limited Interpretability of Embeddings:
While the code generates embeddings, interpreting high-dimensional vectors is inherently challenging. The embeddings cannot directly show the relationships between specific SCP codes or the impact of each feature on predictions. Additional interpretability tools would be required to provide insights into what each embedding dimension represents.
Model Pretraining and Fine-tuning:
The study mentions pretraining on additional datasets and fine-tuning the model on specific patient subgroups for heart failure prediction. This code only leverages BioClinicalBERT embeddings without fine-tuning on a specialized ECG dataset. Fine-tuning could improve the accuracy of predictions when using these embeddings in downstream tasks, but it is currently not implemented in this code.
Batch Processing and GPU Dependency:
This code uses BioClinicalBERT, a large transformer model, which can be computationally expensive to run. While the code has been optimized for GPU processing and batch handling, users without GPU access may experience slow processing times. Running this on CPU for a large dataset may not be practical.
Lack of Downstream Task Implementation:
The code is currently limited to generating structured reports and embeddings. The study also focused on prediction tasks (e.g., heart failure risk). Implementing downstream tasks, such as training a classifier on these embeddings, would be an essential next step for users aiming to replicate the study’s predictive modeling capabilities.
Conclusion
This code provides a foundational toolset for generating structured reports and embeddings from the PTB-XL ECG dataset. While it effectively replicates part of the study's methodology, there are still several challenges and areas for improvement. With additional enhancements, this code can serve as a robust starting point for further research and applications in ECG data analysis and predictive modeling.

