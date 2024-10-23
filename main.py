import os

import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import torch
import mlflow

from mlflow.data.pandas_dataset import from_pandas


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)


def main():

    experiment_name = "my favorite experiment"

    run_name = "my favorite model"
    run_tags = {"tag1": "value1", "tag2": "value2"}
    run_description = "This is a wonderful model trained on XYZ data."
    run_params = {"param1": 1, "param2": 2}

    model_artifact_path = "my_fancy_model"
    model_reg_name = "my_fancy_model_for_registration"

    metadata_artifact_path = "metadata"
    metadata_local_folder_path = "metadata"
    metadata_file_name = "metadata.json"

    metadata_name = "metadata"

    load_dotenv(override=True)
    model = Model()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)

    metadata_local_file = f"{metadata_local_folder_path}/{metadata_file_name}"

    # Model registration ----------------------------------------------------
    # Model level tags and description --------------------------------------
    model_reg_tags = {
        "model_characteristic": "value1"
    }
    model_reg_description = "This is a wonderful model trained on XYZ data."
    # Version specific tags and description ---------------------------------
    version_specific_reg_tags = {
        "tag1": "value1",
        "tag2": "value2"
    }
    version_specific_reg_alias = "alias1"
    version_specific_reg_description = "This is the description for version 1"

    with mlflow.start_run(run_name=run_name, description=run_description) as run:

        mlflow.log_params(run_params)

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
        mlflow.set_tags(run_tags)

        # Log the model ------------------------------------------------------------------------------
        model_info = mlflow.pytorch.log_model(
            model, artifact_path=model_artifact_path)

        # Log metadata --------------------------------------------------------------------------------
        mlflow.log_artifact(metadata_local_file, artifact_path=metadata_artifact_path)
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        run_id = run.info.run_id
        metadata = pd.read_csv(metadata_local_file)
        source = f"{os.getenv('MLFLOW_TRACKING_URI')}/#/experiments/{experiment_id}/runs/{run_id}/artifacts/{metadata_artifact_path}/{metadata_local_file}"
        dataset = from_pandas(df=metadata, source=source, name=metadata_name)
        mlflow.log_input(dataset, context="training")

        # Log figures --------------------------------------------------------------------------------
        fig, ax = plt.subplots()
        ax.plot(range(10), np.random.random(10))
        mlflow.log_figure(fig, "figures/whatever_time_series.png")

        # Register the model -------------------------------------------------------------------------

        latest_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=model_reg_name,
            tags=version_specific_reg_tags)

        # Client for updating the registered model -------------------------------------------------
        client = mlflow.tracking.MlflowClient()

        # Version specific stuff ---------------------

        client.set_registered_model_alias(
            name=model_reg_name,
            alias=version_specific_reg_alias,
            version=latest_version.version)
        client.update_model_version(
            name=model_reg_name,
            version=latest_version.version,
            description=version_specific_reg_description
        )
        # High level description/tags for the registered model shared across all versions ----------------
        client.update_registered_model(
            name=model_reg_name,
            description=model_reg_description
        )
        for k, val in model_reg_tags.items():
            client.set_registered_model_tag(model_reg_name, k, val)

    # print(mlflow.MlflowClient().get_run(run.info.run_id).data)


if __name__ == '__main__':
    main()

