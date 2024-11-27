# DemoMLFlow

## Objective

This project is a demo of how to use MLFlow to track experiments and models, in combination with 
Hydra and Optuna.

## Setup

Add a .env file with the following content:

```
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
```

Run the following command to start the MLFlow server:

```
mlflow server --host 127.0.0.1 --port 5000
```
