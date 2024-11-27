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
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


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


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 10)  # 10 output units for 10 classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input tensor
        x = self.fc1(x)
        return x


class DataHandler:

    def __init__(self):
        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # Download and load the training data
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # Split the dataset into training, validation, and test sets
        train_size = int(0.8 * len(dataset))  # 80% for training
        valid_size = int(0.1 * len(dataset))  # 10% for validation
        test_size = len(dataset) - train_size - valid_size  # 10% for testing
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

        # Create DataLoaders for each dataset
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class Trainer:
    def __init__(self, model: Model, data_handler: DataHandler, cfg: DictConfig):
        self.model = model
        self.data_handler = data_handler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.model.training.learning_rate)

    def epoch(self):
        train_losses = []
        for data, labels in self.data_handler.train_loader:
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track the loss
            train_losses.append(loss.item())

        with torch.no_grad():
            valid_losses = []
            for data, labels in self.data_handler.valid_loader:
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                valid_losses.append(loss.item())

        return np.mean(train_losses), np.mean(valid_losses)

    def test(self):
        test_losses = []
        for data, labels in self.data_handler.test_loader:
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            test_losses.append(loss.item())
        return np.mean(test_losses)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    filter_mlflow_logs()
    print("=" * 100)
    print("Config")
    print("=" * 100)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 100)

    load_dotenv(override=True)
    model_uri = os.getenv("MLFLOW_TRACKING_URI")

    model = Model()
    data_handler = DataHandler()
    trainer = Trainer(model, data_handler, cfg)

    mlflow.set_tracking_uri(model_uri)

    exp_name = cfg.model.mlflow.experiment_name
    print("Experiment name:", exp_name)
    try:
        mlflow.set_experiment(exp_name)
    except mlflow.exceptions.MlflowException:
        print(f"Check the connection to the tracking server: {model_uri}")
        exit(0)
        # raise ValueError(f"Experiment {cfg.mlflow.experiment_name} not found. Check the connection to the tracking server.")
    run_name = cfg.model.mlflow.run_name
    print(f"Run name: {run_name}")
    print("=" * 100)

    with mlflow.start_run(
            run_name=run_name,
            description=cfg.model.mlflow.run_description) as run:

        # Log params from config
        mlflow.log_params(dict(**cfg.model.training))

        # Training loop
        num_epochs = 5
        for epoch in range(num_epochs):
            train_loss, valid_loss = trainer.epoch()
            print(f'Epoch [{epoch+1}/{num_epochs}], Valid loss: {valid_loss:.4f}')

            # Run and dynamic log of metrics ---------------------------------------------------------------------
            mlflow.log_metric(key="train", value=train_loss, step=epoch)
            mlflow.log_metric(key="valid", value=valid_loss, step=epoch)

        test_loss = trainer.test()
        mlflow.log_metric(key="test", value=test_loss, step=None)
        print("=" * 100)
        print(f"Test loss: {test_loss:.4f}")
        print("=" * 100)

        # Log metrics as artifacts --------------------------------------------------------------------------------
        metrics = {
            "precision": np.random.random(),
            "recall": np.random.random(),
        }
        mlflow.log_table(data=metrics, artifact_file="tables/metrics.json")
        mlflow.set_tags(cfg.model.mlflow.run_tags)

        # Log the model ------------------------------------------------------------------------------
        model_info = mlflow.pytorch.log_model(
            model, artifact_path=cfg.model.mlflow.model_artifact_path) # , registered_model_name=cfg.mlflow.model_name)

        # Log metadata --------------------------------------------------------------------------------
        local_path = f"{cfg.data.metadata_local_folder_path}/{cfg.data.metadata_file_name}"
        mlflow.log_artifact(
            local_path=local_path,
            artifact_path=cfg.model.mlflow.metadata_artifact_path,)
        experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id
        run_id = run.info.run_id
        metadata = pd.read_csv(local_path)
        source = f"{os.getenv('MLFLOW_TRACKING_URI')}/#/experiments/{experiment_id}/runs/{run_id}/artifacts/{cfg.model.mlflow.metadata_artifact_path}/{cfg.data.metadata_file_name}"
        dataset = from_pandas(df=metadata, source=source, name=cfg.model.mlflow.dataset_name)
        mlflow.log_input(dataset, context="training")

        # Log figures --------------------------------------------------------------------------------
        fig, ax = plt.subplots()
        ax.plot(range(10), np.random.random(10))
        mlflow.log_figure(fig, "figures/whatever_time_series.png")

        # Client for the registered model -------------------------------------------------
        client = mlflow.tracking.MlflowClient()

        # Register the model -------------------------------------------------------------------------
        model_name = cfg.model.mlflow.model_name
        models = client.search_registered_models(f"name='{model_name}'")

        if not models:
            # Register the model if it doesn't exist
            new_version = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=model_name,
                tags=cfg.model.mlflow.model_tags,)
        else:
            # Explicitly create a new version
            new_version = client.create_model_version(
                name=model_name,
                source=model_info.model_uri,
                run_id=run.info.run_id
            )
        # Version specific stuff ---------------------
        client.set_registered_model_alias(
            name=model_name,
            alias=cfg.model.mlflow.model_version_alias,
            version=new_version.version)
        client.update_model_version(
            name=model_name,
            version=new_version.version,
            description=cfg.model.mlflow.model_version_description
        )
        # High level description/tags for the registered model shared across all versions ----------------
        client.update_registered_model(
            name=model_name,
            description=cfg.model.mlflow.model_description,
        )
        for k, val in cfg.model.mlflow.model_tags.items():
            client.set_registered_model_tag(model_name, k, val)


if __name__ == '__main__':

    main()

