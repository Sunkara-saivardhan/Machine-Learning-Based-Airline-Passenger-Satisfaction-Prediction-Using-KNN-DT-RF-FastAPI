# Airline Passenger Satisfaction Prediction Using KNN, Decision Tree, Random Forest & FastAPI

## Description

This project is an end-to-end Machine Learning application for predicting airline passenger satisfaction. Three classification algorithms — K-Nearest Neighbors (KNN), Decision Tree, and Random Forest — are trained and evaluated to compare their performance.

The selected trained Machine Learning pipeline is deployed as a REST API using FastAPI and hosted on Render, allowing users to provide passenger information and receive satisfaction predictions through an API endpoint.

## 🚀 Live API

**Live Application:**  
[https://machine-learning-based-airline-passenger.onrender.com](https://machine-learning-based-airline-passenger.onrender.com)

### Swagger API Documentation

You can interact with and test the API using Swagger UI:

[**Open Swagger Documentation**](https://machine-learning-based-airline-passenger.onrender.com/docs)

---

## Project Objectives

- Perform data preprocessing and feature preparation.
- Train multiple Machine Learning classification models.
- Compare model performance using evaluation metrics.
- Select a suitable model for prediction.
- Save the trained Machine Learning pipeline.
- Deploy the model using FastAPI.
- Host the API on Render.
- Provide real-time passenger satisfaction predictions through a REST API.

## Machine Learning Models

The project uses the following classification algorithms:

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

## Dataset

The dataset contains airline passenger information and satisfaction levels.

### Target Variable

The target variable contains two classes:

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
- Handling unknown categorical values using `handle_unknown="ignore"`

The preprocessing steps and Machine Learning model are combined into a single Scikit-learn Pipeline.

## Model Evaluation

The models are evaluated using:

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
