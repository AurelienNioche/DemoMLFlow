import os

import \
    pandas as pd
from dotenv import load_dotenv
import mlflow
import matplotlib.pyplot as plt
import numpy as np


def main():
    load_dotenv(override=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("my favorite experiment")
    print(os.getenv("MLFLOW_TRACKING_USERNAME"))
    with mlflow.start_run(run_name="my favorite model") as run:

        mlflow.log_params({"param1": 1, "param2": 2})

        for epoch in range(0, 3):
            mlflow.log_metric(key="train", value=2 * epoch, step=epoch)
            mlflow.log_metric(key="valid", value=2 * epoch, step=epoch)

        mlflow.log_metric(key="test", value=0.5, step=None)

        my_df = pd.DataFrame({
            "inputs": ["What is MLflow?", "What is Databricks?"],
            "outputs": ["MLflow is ...", "Databricks is ..."],
            "toxicity": [0.0, 0.0],
        })
        print(my_df)

        mlflow.log_table(data=my_df.to_dict(), artifact_file="tables/my_table.json")

        fig, ax = plt.subplots()
        ax.plot(range(0, 10), [x ** 2 for x in range(0, 10)])
        mlflow.log_figure(fig, "figures/whatevertimeseries.png")

        fig, ax = plt.subplots()
        ax.imshow(np.array([[1, 2], [3, 4]]))
        mlflow.log_figure(fig, "figures/whateverimage.png")

    print(mlflow.MlflowClient().get_run(run.info.run_id).data)


if __name__ == '__main__':
    main()

