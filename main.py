import os
import warnings
import logging

import hydra
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import torch
import mlflow
from mlflow.data.pandas_dataset import from_pandas
from omegaconf import DictConfig, OmegaConf


def filter_mlflow_logs():
    """
    Removing all the warnings from the mlflow logs
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
    loggers = (
        [name for name in logging.root.manager.loggerDict if "mlflow" in name]
        + ["mlflow.system_metrics.system_metrics_monitor"]
    )
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        if logger_name == "mlflow.tracking._tracking_service.client":
            logger.addFilter(lambda record: record.levelno != logging.WARNING)
        else:
            logger.addFilter(lambda record: record.levelno == logging.ERROR)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    filter_mlflow_logs()

    print("=" * 100)
    print("Config")
    print("=" * 100)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 100)

    model_artifact_path = "my_fancy_model"
    model_reg_name = "my_fancy_model_for_registration"

    metadata_artifact_path = "metadata"
    metadata_local_folder_path = "data"
    metadata_file_name = "metadata.json"

    metadata_name = "metadata"

    model_uri = os.getenv("MLFLOW_TRACKING_URI")

    load_dotenv(override=True)
    model = Model()

    mlflow.set_tracking_uri(model_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    metadata_local_file = f"{metadata_local_folder_path}/{metadata_file_name}"

    with mlflow.start_run(
            run_name=cfg.mlflow.run_name,
            description=cfg.mlflow.run_description) as run:

        # Log params from config
        mlflow.log_params(dict(**cfg.flavor))

        # Run and dynamic log of metrics ---------------------------------------------------------------------
        for epoch in range(0, 3):
            mlflow.log_metric(key="train", value=np.random.random() - epoch*np.random.random()/100, step=epoch)
            mlflow.log_metric(key="valid", value=np.random.random() - epoch*np.random.random()/100, step=epoch)

        mlflow.log_metric(key="test", value=np.random.random(), step=None)

        # Log metrics as artifacts --------------------------------------------------------------------------------
        metrics = {
            "precision": np.random.random(),
            "recall": np.random.random(),
        }
        mlflow.log_table(data=metrics, artifact_file="tables/metrics.json")
        mlflow.set_tags(cfg.mlflow.run_tags)

        # Log the model ------------------------------------------------------------------------------
        model_info = mlflow.pytorch.log_model(
            model, artifact_path=model_artifact_path)

        # Log metadata --------------------------------------------------------------------------------
        mlflow.log_artifact(metadata_local_file, artifact_path=metadata_artifact_path)
        experiment_id = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name).experiment_id
        run_id = run.info.run_id
        metadata = pd.read_csv(metadata_local_file)
        source = f"{os.getenv('MLFLOW_TRACKING_URI')}/#/experiments/{experiment_id}/runs/{run_id}/artifacts/{metadata_artifact_path}/{metadata_local_file}"
        dataset = from_pandas(df=metadata, source=source, name=metadata_name)
        mlflow.log_input(dataset, context="training")

        # Log figures --------------------------------------------------------------------------------
        fig, ax = plt.subplots()
        ax.plot(range(10), np.random.random(10))
        mlflow.log_figure(fig, "figures/whatever_time_series.png")

        # Client for the registered model -------------------------------------------------
        client = mlflow.tracking.MlflowClient()

        # Register the model -------------------------------------------------------------------------
        models = client.search_registered_models(f"name='{cfg.mlflow.model_name}'")

        if not models:
            # Register the model if it doesn't exist
            new_version = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=cfg.mlflow.model_name,
                tags=cfg.mlflow.model_tags,)
        else:
            # Explicitly create a new version
            new_version = client.create_model_version(
                name=cfg.mlflow.model_name,
                source=model_info.model_uri,
                run_id=run.info.run_id
            )
        # Version specific stuff ---------------------
        client.set_registered_model_alias(
            name=cfg.mlflow.model_name,
            alias=cfg.mlflow.model_version_alias,
            version=new_version.version)
        client.update_model_version(
            name=cfg.mlflow.model_name,
            version=new_version.version,
            description=cfg.mlflow.model_version_description
        )
        # High level description/tags for the registered model shared across all versions ----------------
        client.update_registered_model(
            name=cfg.mlflow.model_name,
            description=cfg.mlflow.model_description,
        )
        for k, val in cfg.mlflow.model_tags.items():
            client.set_registered_model_tag(model_reg_name, k, val)


if __name__ == '__main__':
    main()

