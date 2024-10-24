import io
import os
import numpy as np
import pandas as pd
import pickle
import json
import logging

import mlflow
from mlflow.models import infer_signature
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Literal
from datetime import timedelta
import time

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)

MLFLOW_EXPERIMENT_NAME = 'vladislav_dubov'

BUCKET = Variable.get("S3_BUCKET")
DEFAULT_ARGS = {
    "owner": "Dubov Vladislav",
    "email": "vlvldubov@edu.hse.ru",
    "email_on_failure": True,
    "email_on_retry": False,
    "retry": 3,
    "retry_delay": timedelta(minutes=1)
}
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"
model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

def check_experiment_exists(name):
    return mlflow.search_experiments(
        filter_string=f"name = '{MLFLOW_EXPERIMENT_NAME}'"
    )

def create_dag(dag_id: str):
    def init(owner: str) -> Dict[str, Any]:
        _LOG.info("Init")

        if not check_experiment_exists(MLFLOW_EXPERIMENT_NAME):
            mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        run_id = ''

        with mlflow.start_run(run_name="dubov_vv") as parent_run:
            run_id = parent_run.info.run_id

        return {
            "init_timestamp_start": time.time(),
            "run_id": run_id,
            "experiment_id": mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME).experiment_id
        }

    def get_data(**kwargs) -> Dict[str, Any]:
        _LOG.info("get_data started")
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="init")
        metrics["get_data_timestamp_start"] = time.time()
        owner = kwargs["owner"]
        owner_path = ''.join(owner.split(' '))
        housing = fetch_california_housing(as_frame=True)
        data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)

        s3_hook = S3Hook("s3_connection")
        filebuffer = io.BytesIO()
        data.to_pickle(filebuffer)
        filebuffer.seek(0)

        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"{owner_path}/datasets/california_housing.pkl",
            bucket_name=BUCKET,
            replace=True,
        )
        metrics["dataset_length"] = len(data)
        metrics["get_data_timestamp_end"] = time.time()
        return metrics

    def prepare_data(**kwargs) -> Dict[str, Any]:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="get_data")
        metrics["prepare_data_timestamp_start"] = time.time()
        s3_hook = S3Hook("s3_connection")
        owner = kwargs["owner"]
        owner_path = ''.join(owner.split(' '))
        file = s3_hook.download_file(key=f"{owner_path}/datasets/california_housing.pkl", bucket_name=BUCKET)
        data = pd.read_pickle(file)

        X, y = data[FEATURES], data[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_fitted = scaler.fit_transform(X_train)
        X_test_fitted = scaler.transform(X_test)

        for name, data in zip(
            ["X_train", "X_test", "y_train", "y_test"],
            [X_train_fitted, X_test_fitted, y_train, y_test],
        ):
            filebuffer = io.BytesIO()
            pickle.dump(data, filebuffer)
            filebuffer.seek(0)
            s3_hook.load_file_obj(
                file_obj=filebuffer,
                key=f"{owner_path}/datasets/{name}.pkl",
                bucket_name=BUCKET,
                replace=True,
            )
        metrics["feature"] = FEATURES
        metrics["prepare_data_timestamp_end"] = time.time()
        return metrics

    def train_model(**kwargs) -> Dict[str, Any]:
        ti = kwargs["ti"]
        m_name = kwargs["m_name"]
        xcom_data = ti.xcom_pull(task_ids="init")
        print('xcom_data', xcom_data)
        with mlflow.start_run(run_name=m_name, parent_run_id=xcom_data['run_id'], experiment_id=xcom_data['experiment_id'], nested=True):
            metrics = ti.xcom_pull(task_ids="prepare_data")
            metrics[f"train_model_{m_name}_timestamp_start"] = time.time()
            owner = kwargs["owner"]
            owner_path = ''.join(owner.split(' '))
            s3_hook = S3Hook("s3_connection")
            data = {}
            for name in ["X_train", "X_test", "y_train", "y_test"]:
                file = s3_hook.download_file(
                    key=f"{owner_path}/datasets/{name}.pkl",
                    bucket_name=BUCKET,
                )
                data[name] = pd.read_pickle(file)

            model = models[m_name]
            model.fit(data["X_train"], data["y_train"])
            prediction = model.predict(data["X_test"])

            signature = infer_signature(data["X_test"], prediction)
            model_info = mlflow.sklearn.log_model(model, m_name, signature=signature)
            mlflow.evaluate(
                model=model_info.model_uri,
                data=data["X_test"].copy(),
                targets=np.array(data["y_test"]),
                model_type="regressor",
                evaluators=["default"],
            )

            metrics[f"train_model_{m_name}_timestamp_end"] = time.time()
            return metrics

    def save_results(**kwargs) -> None:
        s3_hook = S3Hook("s3_connection")
        ti = kwargs["ti"]
        owner = kwargs["owner"]
        owner_path = ''.join(owner.split(' '))
        metrics = {}

        for model_name in models.keys():
            metrics.update(ti.xcom_pull(task_ids=f"train_{model_name}"))

        filebuffer = io.BytesIO()
        filebuffer.write(json.dumps(metrics).encode())
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"{owner_path}/results/data.json",
            bucket_name=BUCKET,
            replace=True,
        )
        _LOG.info(metrics)

    dag = DAG(
        dag_id = dag_id,
        schedule_interval = "0 1 * * * ",
        start_date = days_ago(2),
        catchup = False,
        tags = ["mlops"],
        default_args = DEFAULT_ARGS
    )

    with dag:
        task_init = PythonOperator(
            task_id="init", 
            python_callable=init, 
            dag=dag, 
            op_kwargs={'owner': DEFAULT_ARGS["owner"]}
        )

        task_get_data = PythonOperator(
            task_id="get_data", 
            python_callable=get_data, 
            dag=dag, 
            op_kwargs={'owner': DEFAULT_ARGS["owner"]}
        )

        task_prepare_data = PythonOperator(
            task_id="prepare_data", 
            python_callable=prepare_data, 
            dag=dag, 
            op_kwargs={'owner': DEFAULT_ARGS["owner"]}
        )

        trains_tasks = []

        for model_name in models.keys():
            task_train_model = PythonOperator(
                task_id=f"train_{model_name}", 
                python_callable=train_model, 
                dag=dag, 
                op_kwargs={'m_name': model_name, 'owner': DEFAULT_ARGS["owner"]}
            )
            trains_tasks.append(task_train_model)

        task_save_results = PythonOperator(
            task_id="save_results", 
            python_callable=save_results, 
            dag=dag,
            op_kwargs={'owner': DEFAULT_ARGS["owner"]}
        )

        task_init >> task_get_data >> task_prepare_data >> trains_tasks >> task_save_results

configure_mlflow()
create_dag("dubov_vladislav")