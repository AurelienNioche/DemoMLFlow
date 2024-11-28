import os

import hydra
import optuna
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig

from main import train


# Define the objective function for Optuna
def objective(trial):
    GlobalHydra.instance().clear()
    with initialize(config_path="../conf", version_base=None):
        config = compose(config_name="config")

        # Sample hyperparameters using Optuna
        for key in config.model.optuna.hyperparameters.keys():
            method = config.model.optuna.hyperparameters[key].method
            args = config.model.optuna.hyperparameters[key].args
            config.model.training[key] = getattr(trial, method)(key, **args)

        # config.params.num_epochs = trial.suggest_int('num_epochs', 5, 20)

        # Print out the configuration for debug purposes
        print(OmegaConf.to_yaml(config))

        # Mock training function, replace with your actual training code
        accuracy = train(config)
        return accuracy


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    optuna_storage = os.getenv("OPTUNA_STORAGE")

    if optuna_storage:
        # Extract the directory from the SQLite URI
        directory = os.path.dirname(optuna_storage.replace('sqlite:///', ''))
        os.makedirs(directory, exist_ok=True)
    study = optuna.create_study(
        direction=cfg.model.optuna.direction,
        study_name=cfg.model.optuna.study_name,
        storage=optuna_storage,
        load_if_exists=True)
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
