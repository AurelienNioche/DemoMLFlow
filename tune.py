import optuna
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

from main import train


# Define the objective function for Optuna
def objective(trial):
    with initialize(config_path="."):
        config = compose(config_name="config")

        # Sample hyperparameters using Optuna
        config.params.learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        config.params.batch_size = trial.suggest_int('batch_size', 16, 128)
        # config.params.num_epochs = trial.suggest_int('num_epochs', 5, 20)

        # Print out the configuration for debug purposes
        print(OmegaConf.to_yaml(config))

        # Mock training function, replace with your actual training code
        accuracy = mock_train(config.params)
        return accuracy


# Mock training function
def mock_train(params: DictConfig):
    # Simulate model accuracy based on hyperparameters for demonstration
    import random
    return random.uniform(0.8, 1.0)


def main():

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # Display the best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
