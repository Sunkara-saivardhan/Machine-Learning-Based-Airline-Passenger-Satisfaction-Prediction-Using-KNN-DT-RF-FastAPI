# Airline Passenger Satisfaction Prediction Using KNN, Decision Tree, Random Forest & FastAPI

## Description

This project is an end-to-end Machine Learning application for predicting airline passenger satisfaction. Three classification algorithms — K-Nearest Neighbors (KNN), Decision Tree, and Random Forest — are trained and evaluated to compare their performance.

The selected trained machine learning pipeline is deployed as a REST API using FastAPI, allowing users to provide passenger information and receive a satisfaction prediction through an API endpoint.

## Project Objectives

- Perform data preprocessing and feature preparation.
- Train multiple Machine Learning classification models.
- Compare model performance using evaluation metrics.
- Select a suitable model for prediction.
- Save the trained Machine Learning pipeline.
- Deploy the model using FastAPI.
- Provide real-time predictions through a REST API.

## Machine Learning Models

The project uses the following classification algorithms:

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

## Dataset

The dataset contains airline passenger information and satisfaction levels.

### Target Variable

The target variable is:

- `Neutral or Dissatisfied`
- `Satisfied`

### Features

The model uses passenger-related features including:

- Gender
- Age
- Customer Type
- Type of Travel
- Class
- Flight Distance
- Departure Delay
- Arrival Delay
- Departure and Arrival Time Convenience
- Ease of Online Booking
- Check-in Service
- Online Boarding
- On-board Service
- Seat Comfort
- Leg Room Service
- Cleanliness
- Food and Drink
- In-flight Service
- In-flight Wifi Service
- In-flight Entertainment
- Baggage Handling

`ID` and `Gate Location` were excluded from the modeling features.

## Data Preprocessing

The preprocessing pipeline includes:

- Numerical feature imputation using the median
- Numerical feature scaling using `StandardScaler`
- Categorical feature imputation using the most frequent value
- Categorical feature encoding using `OneHotEncoder`
- `handle_unknown="ignore"` for categorical features

The preprocessing steps and machine learning model are combined into a single Scikit-learn Pipeline.

## Model Evaluation

The models are evaluated using classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Random Forest Performance

The Random Forest model achieved approximately **94.68% test accuracy** in the current experiment.

The model was also evaluated using 5-fold cross-validation, achieving an average accuracy of approximately **95.99%**.

## FastAPI Deployment

The trained Machine Learning pipeline is saved as:

```text
model.pkl
