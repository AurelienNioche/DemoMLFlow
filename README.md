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

## Experiments

Run experiments with the following command:

```
python src/main.py
```

Look at the results in the MLFlow dashboard at the address `http://127.0.0.1:5000/`.

## Optuna

Tune hyperparameters with Optuna with the following command:

```
python src/optuna_tuning.py
```

Look at the results in the Optuna dashboard:

```
optuna-dashboard sqlite:///study/example.db
```