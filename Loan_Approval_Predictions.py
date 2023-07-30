#!/usr/bin/env python
# coding: utf-8

# # Loan Approval Predictions
# 
# **This is #1 attempt of Mlzoomcamp project**

# # Problem Statement:
# The loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals or organizations for obtaining loans from a lending institution. It includes various factors such as cibil score, income, employment status, loan term, loan amount, assets value, and loan status. 
# 
#         
#* Data Columns
#   * Loan_id: the Number of Loan 
#   * no_of_dependents
#   * education
#   * self_employed
#   * income_annum
#   * loan_amount                 
#   * loan_term                   
#   * cibil_score                 
#   * residential_assets_value    
#   * commercial_assets_value     
#   * luxury_assets_value         
#   * bank_asset_value            
#   * loan_status: Our target Column for prediction            
# 

# # 1. Importing Libraries Used
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle 
import os
from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, classification_report
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
import json
from datetime import date
from optuna.samplers import TPESampler
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.context import get_run_context
from prefect_email import EmailServerCredentials, email_send_message
import sys




# Constants 

RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_jobs']
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
HPO_EXPERIMENT_NAME = "LoanApprovalExpirement_HPO"
EXPERIMENT_NAME = "LoanApprovalExpirement"
dest_path = "./output" #sys.argv[1] 
raw_data_path =  "./data" #sys.argv[2]
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Notify incase of failure
def notify_exc_by_email(exc):
    context = get_run_context()
    flow_run_name =context.flow_run.name
    email_server_credentials = EmailServerCredentials.load("emailnotification")
    email_send_message(
        email_server_credentials=email_server_credentials,
        subject=f"Flow run {flow_run_name!r} failed",
        msg=f"Flow run {flow_run_name!r} failed due to {exc}.",
        email_to=email_server_credentials.username,
    )
# Notify incase of Success
def notify_Success_by_email(Success):
    context = get_run_context()
    flow_run_name =context.flow_run.name
    email_server_credentials = EmailServerCredentials.load("emailnotification")
    email_send_message(
        email_server_credentials=email_server_credentials,
        subject=f"Flow run {flow_run_name!r} Success",
        msg=f"Flow run {flow_run_name!r} {Success}.",
        email_to=email_server_credentials.username,
    )
# ## 3. Preprocess the data 

# ## 3.1 Preprocess data Function

def read_dataframe(filename: str):
    df = pd.read_csv(filename)
    df.drop('loan_id',axis=1,inplace=True)
    education={' Graduate':1,' Not Graduate':0}
    self_employed ={' Yes':1,' No':0}
    loan_status={' Approved':1,' Rejected':0}
    df[' education']=df[' education'].apply(lambda x: education[x])
    df[' self_employed']=df[' self_employed'].apply(lambda x: self_employed[x])
    df[' loan_status']=df[' loan_status'].apply(lambda x: loan_status[x])
    return df


# ## 3.2 Storing pickle files

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


# ## 3.3 Data Spliting
@task(retries=3, retry_delay_seconds=2)
def run_data_prep(raw_data_path: str, dest_path: str, filename: str = "loan_approval_dataset"):
    # Load parquet files
    df = read_dataframe(
        os.path.join(raw_data_path, f"{filename}.csv")
    )
    

    # Extract the target
    X=df.drop([' loan_status'],axis=1)
    y=df[' loan_status']

    # split train and validate and test
    x_train, x_val, y_train, y_val=train_test_split(X,y,test_size=0.2,random_state=42)
    # x_train,x_val, y_train, y_val=train_test_split(X_train,Y_train,test_size=0.2,random_state=42)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save datasets
    dump_pickle((x_train, y_train), os.path.join(dest_path, "train_orch.pkl"))
    dump_pickle((x_val, y_val), os.path.join(dest_path, "val_orch.pkl"))
    # dump_pickle((x_test, y_test), os.path.join(dest_path, "test_orch.pkl"))


# # 4. Loading Data Pickle

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# # Experiment Tracking
# # Adding Optimization

@task(log_prints=True)
def run_optimization(data_path: str, num_trials: int):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(HPO_EXPERIMENT_NAME)
    X_train, y_train = load_pickle(os.path.join(data_path, "train_orch.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val_orch.pkl"))

    def objective(trial):
        with mlflow.start_run():
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 50, 1),
                'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
                'random_state': 42,
                'n_jobs': -1
            }
            mlflow.log_param("param",params)
            rf = RandomForestClassifier(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            mlflow.log_metric('acc',acc)
        
        return acc
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)

def converttodict(str1):
    return json.loads(str1)



# # Interacting with Model Registry

def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train_orch.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val_orch.pkl"))
    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])

        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_acc = accuracy_score(y_val, rf.predict(X_val))*100
        ConfusionMatrixDisplay(confusion_matrix(y_val, rf.predict(X_val))).plot()
        mlflow.log_metric("val_acc", val_acc)
        mlflow.sklearn.log_model(rf, artifact_path="models")
        with open("models/rf.b", "wb") as f_out:
            pickle.dump(rf,f_out)
        mlflow.log_artifact("models/rf.b", artifact_path="models")
        markdown__acc_report = f"""# ACC Report

        ## Summary

        Loan Acceptance Prediction 

        ## ACC RF Model


        | Region    | accuracy |
        |:----------|-------:|
        | {date.today()} | {val_acc:.2f} |
        """

        create_markdown_artifact(
            key="loanacceptance-model-report", markdown=markdown__acc_report
        )


@task(log_prints=True)
def run_register_model(data_path: str, top_n: int,HPO_EXPERIMENT_NAME,EXPERIMENT_NAME):
    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.acc ASC"]
    )
    for run in range(len(runs)-2):
        print(runs[run].data.params)
        param = converttodict(runs[run].data.params['param'].replace("\'", "\""))
        train_and_log_model(data_path=data_path, params=param)
    # Select the model with the highest test acc
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.acc DESC"])[0]

    # Register the best model
    
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=EXPERIMENT_NAME)

@flow
def main_flow():
    try:
        print("Start ..")
        run_data_prep(raw_data_path = raw_data_path, dest_path= dest_path)
        run_optimization(dest_path, 5)
        run_register_model(dest_path, 5,HPO_EXPERIMENT_NAME,EXPERIMENT_NAME)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        latest_versions = client.get_latest_versions(name=EXPERIMENT_NAME)
        for version in latest_versions:
            # Make them Staging phase
            print(f"version: {version.version}, stage: {version.current_stage}") 
            model_version = version.version
            new_stage = "Staging"
            client.transition_model_version_stage(
                name=EXPERIMENT_NAME,
                version=model_version,
                stage=new_stage,
                archive_existing_versions=False
            )
        # move version 3 to production
        client.transition_model_version_stage(
        name=EXPERIMENT_NAME,
        version=1,
        stage="Production",
        archive_existing_versions=True)
        print("Model moved to production")
        notify_Success_by_email("Flow has been Done Succesfully")
    except Exception as exc:
        print("exc")
        notify_exc_by_email(exc)
        raise

if __name__ == '__main__':
    main_flow()