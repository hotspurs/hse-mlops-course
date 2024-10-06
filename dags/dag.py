import io
import numpy as np
import pandas as pd
import pickle
import json

from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Literal
from datetime import timedelta
import time
import logging

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

def create_dag(dag_id: str, m_name: Literal["random_forest", "linear_regression", "desicion_tree"]):
    def init(m_name: Literal["random_forest", "linear_regression", "desicion_tree"], owner: str) -> Dict[str, Any]:
        _LOG.info("Init")
        return {
            "init_timestamp_start": time.time(),
            "model_name": m_name
        }

    def get_data(**kwargs) -> Dict[str, Any]:
        _LOG.info("get_data started")
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="init")
        metrics["get_data_timestamp_start"] = time.time()
        owner = kwargs["owner"]
        owner_path = ''.join(owner.split(' '))
        m_name = kwargs["m_name"]
        housing = fetch_california_housing(as_frame=True)
        data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)

        s3_hook = S3Hook("s3_connection")
        filebuffer = io.BytesIO()
        data.to_pickle(filebuffer)
        filebuffer.seek(0)

        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"{owner_path}/{m_name}/datasets/california_housing.pkl",
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
        m_name = kwargs["m_name"]
        file = s3_hook.download_file(key=f"{owner_path}/{m_name}/datasets/california_housing.pkl", bucket_name=BUCKET)
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
                key=f"{owner_path}/{m_name}/datasets/{name}.pkl",
                bucket_name=BUCKET,
                replace=True,
            )
        metrics["feature"] = FEATURES
        metrics["prepare_data_timestamp_end"] = time.time()
        return metrics

    def train_model(**kwargs) -> Dict[str, Any]:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="prepare_data")
        metrics["train_model_timestamp_start"] = time.time()
        owner = kwargs["owner"]
        m_name = kwargs["m_name"]
        owner_path = ''.join(owner.split(' '))
        s3_hook = S3Hook("s3_connection")
        data = {}
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(
                key=f"{owner_path}/{m_name}/datasets/{name}.pkl",
                bucket_name=BUCKET,
            )
            data[name] = pd.read_pickle(file)

        model = models[m_name]
        model.fit(data["X_train"], data["y_train"])
        prediction = model.predict(data["X_test"])

        metrics["r2_score"] = r2_score(data["y_test"], prediction)
        metrics["rmse"] = mean_squared_error(data["y_test"], prediction) ** 0.5
        metrics["mae"] = median_absolute_error(data["y_test"], prediction)
        metrics["train_model_timestamp_end"] = time.time()
        return metrics

    def save_results(**kwargs) -> None:
        s3_hook = S3Hook("s3_connection")
        ti = kwargs["ti"]
        owner = kwargs["owner"]
        owner_path = ''.join(owner.split(' '))
        m_name = kwargs["m_name"]
        metrics = ti.xcom_pull(task_ids="train_model")
        filebuffer = io.BytesIO()
        filebuffer.write(json.dumps(metrics).encode())
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"{owner_path}/{m_name}/results/data.json",
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
            op_kwargs={'m_name': m_name, 'owner': DEFAULT_ARGS["owner"]}
        )

        task_get_data = PythonOperator(
            task_id="get_data", 
            python_callable=get_data, 
            dag=dag, 
            op_kwargs={'m_name': m_name, 'owner': DEFAULT_ARGS["owner"]}
        )

        task_prepare_data = PythonOperator(
            task_id="prepare_data", 
            python_callable=prepare_data, 
            dag=dag, 
            op_kwargs={'m_name': m_name, 'owner': DEFAULT_ARGS["owner"]}
        )

        task_train_model = PythonOperator(
            task_id="train_model", 
            python_callable=train_model, 
            dag=dag, 
            op_kwargs={'m_name': m_name, 'owner': DEFAULT_ARGS["owner"]}
        )

        task_save_results = PythonOperator(
            task_id="save_results", 
            python_callable=save_results, 
            dag=dag,
            op_kwargs={'m_name': m_name, 'owner': DEFAULT_ARGS["owner"]}
        )

        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results

for model_name in models.keys():
    create_dag(f"dubov_vladislav_{model_name}", model_name)