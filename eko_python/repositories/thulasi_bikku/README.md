# Lung Disease Prediction using Deep Learning

This repository contains a deep learning model for respiratory disease classification using audio features from lung sounds. The model is trained on two key datasets: ICBHI (International Conference on Biomedical Health Informatics) respiratory sounds database and the Coswara dataset.

## Dataset Overview

The datasets are automatically downloaded and processed when running the notebook:

1. **ICBHI Dataset**: Contains audio recordings of respiratory sounds from patients with various respiratory conditions
2. **Coswara Dataset**: Contains respiratory sound recordings collected via web-based platforms

## Project Structure

- `deep_learning_lung_disease.ipynb`: The main Jupyter notebook containing all code for data processing, model training, and evaluation
- `extract_data.py`: Script for processing the downloaded audio files
- `requirements.txt`: List of required Python packages
- Various PNG files: Visualizations of model performance and results

## Getting Started

1. Clone this repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook deep_learning_lung_disease.ipynb`

The notebook includes code to automatically download the required datasets, so you don't need to manually download any large files.

## Model Performance

The model achieves high accuracy in classifying various respiratory conditions including:
- Asthma
- Bronchiectasis
- COPD (Chronic Obstructive Pulmonary Disease)
- Healthy individuals
- Respiratory symptoms

Detailed performance metrics and visualizations are provided in the notebook. 