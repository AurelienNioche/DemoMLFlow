import os

import hydra
import optuna
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig

from main import train


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

    # Define the objective function for Optuna
    def objective(_trial):
        # Sample hyperparameters using Optuna
        for _key in cfg.model.optuna.hyperparameters.keys():
            method = cfg.model.optuna.hyperparameters[_key].method
            args = cfg.model.optuna.hyperparameters[_key].args
            cfg.model.training[_key] = getattr(_trial, method)(_key, **args)

        # Print out the configuration for debug purposes
        print(OmegaConf.to_yaml(cfg))

        # Train
        return train(cfg)

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
