# 🏦 Credit Scoring Prediction

A Machine Learning project that predicts the credit risk of a customer and estimates the likelihood of loan default.

The project uses customer financial and loan-related information to classify applicants into different credit risk levels using a Machine Learning model.

## Project Overview

The main goal of this project is to build a Machine Learning based credit scoring system that can help identify customers who may have a higher probability of defaulting on a loan.

The project includes data preprocessing, feature encoding, model training, and a Streamlit web application for making predictions.

## Features

- Credit risk prediction
- Loan default prediction
- Interactive Streamlit web application
- Customer financial information as input
- Categorical feature encoding
- Probability-based prediction
- Low, Medium, and High credit risk classification
- Trained Random Forest Classifier
- Saved Machine Learning model

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Input Features

The application takes the following customer and loan information:

- Age
- Annual Income
- Employment Length
- Loan Grade
- Loan Amount
- Interest Rate
- Loan Percent Income
- Previous Loan Default
- Credit History Length
- Home Ownership
- Loan Purpose

## Machine Learning Workflow

```text
Customer Dataset
       ↓
Data Preprocessing
       ↓
Feature Encoding
       ↓
Feature Selection
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Random Forest Classifier
       ↓
Streamlit Application
       ↓
Credit Risk Prediction