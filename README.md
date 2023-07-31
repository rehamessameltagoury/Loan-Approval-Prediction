
# Loan Approval Prediction 

# Problem Statement:
The loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals or organizations for obtaining loans from a lending institution. It includes various factors such as cibil score, income, employment status, loan term, loan amount, assets value, and loan status. 

        
* Data Columns
  * Loan_id: the Number of Loan 
  * no_of_dependents
  * education
  * self_employed
  * income_annum
  * loan_amount                 
  * loan_term                   
  * cibil_score                 
  * residential_assets_value    
  * commercial_assets_value     
  * luxury_assets_value         
  * bank_asset_value            
  * loan_status: Our target Column for prediction    

  ## Loan_Approval_prediction.ipynb
    In this notebook you will find the Data exploration and interacting with MLFLOW Model Registry
  ## Loan_Approval_prediction.py
    ### read_dataframe :
        Function used to read the csv data from data folder and then encode `loan_status` , `education` , `self_employed` and returns dataframe
    ### run_data_prep:
         this function is considered to be the first task,it takes the Data path and output the pickle files (train , Val) to be passed to Random Forest Model
    ### run_optimization:
          this function creates an optimize study to run mutliple parameters Random forest model and pick the best one
    ### train_and_log_model:
          Function used to train and log model parameters to mlflow
    ### run_register_model:
        takes data path and two experments name the optimization experiment and normal one to compare and pick the best model for registring it.
    ### main_flow:
        at the end it picks the best model which is version 1  with acc = 98% and move it to production  state
  ### the script is deployed on prefect locally and send emails when there is failure or success to notify the developer
      
  ## How to Run:
    1- Download data folder
  
    2- Install requirements.txt  `pip install -r requirements.txt`
  
    2- Run Loan_Approval_prediction.py  `python Loan_Approval_prediction.py`
  
    3- for deploying
  Create work pool named loan_approval:  `prefect deploy Loan_Approval_prediction.py:main_flow -n loan1 -p loan_approval`
  
  Get worker `prefect worker start --pool 'loan_approval'`
  
  Start Run  `prefect deployment run 'main-flow/loan1'`
  
    #### please note if you want the emails to be activited add your app password in Cred.py
    4- For Monitoring
  
  Run docker compose
        `docker-compose up --build`
  
  Run evidently file 
         `python .\evidently_metrics.py`
  
  Open Grafana on port 3000
