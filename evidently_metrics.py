import os
import time
import pickle
import random
import logging
import datetime

import joblib
import pandas as pd
import psycopg2 as psycopg
from prefect import flow, task
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

SEND_TIMEOUT = 10
rand = random.Random()


def read_dataframe(filename: str):
    df = pd.read_csv(filename)
    df.drop('loan_id', axis=1, inplace=True)
    education = {' Graduate': 1, ' Not Graduate': 0}
    self_employed = {' Yes': 1, ' No': 0}
    loan_status = {' Approved': 1, ' Rejected': 0}
    df[' education'] = df[' education'].apply(lambda x: education[x])
    df[' self_employed'] = df[' self_employed'].apply(lambda x: self_employed[x])
    df[' loan_status'] = df[' loan_status'].apply(lambda x: loan_status[x])
    return df


create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float
)
"""


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


reference_data = load_pickle(os.path.join("./output", "val_orch.pkl"))
with open('models/rf.b', 'rb') as f_in:
    model = joblib.load(f_in)

raw_data = read_dataframe('./data/loan_approval_dataset.csv')

begin = datetime.datetime(2022, 2, 1, 0, 0)
num_features = [
    'no_of_dependents',
    'loan_amount',
    'residential_assets_value',
    'bank_asset_value',
]
cat_features = [' education', ' self_employed', ' loan_status']
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None,
)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
)


@task(retries=2, retry_delay_seconds=5, name="calculate metrics")
def prep_db():
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=new_password"
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE dbname='loanprediction'")
        if len(res.fetchall()) == 0:
            conn.execute("create database loanprediction;")
        with psycopg.connect(
            "host=localhost port=5432 dbname=loanprediction user=postgres password=new_password"
        ) as conn:
            conn.execute(create_table_statement)


@task(retries=2, retry_delay_seconds=5, name="calculate metrics postgres")
def calculate_metrics_postgresql(curr, i):
    current_data = raw_data[
        (raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i)))
        & (raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))
    ]
    current_data['prediction'] = model.predict(
        current_data[num_features + cat_features].fillna(0)
    )

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current'][
        'share_of_missing_values'
    ]

    curr.execute(
        "insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
        (
            begin + datetime.timedelta(i),
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
        ),
    )


@flow
def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect(
        "host=localhost port=5432 dbname=loanprediction user=postgres password=new_password"
    ) as conn:
        for i in range(0, 27):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, i)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent")


if __name__ == '__main__':
    batch_monitoring_backfill()
