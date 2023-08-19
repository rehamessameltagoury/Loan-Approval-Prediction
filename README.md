
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
  ## evidently_metrics.py:
  I used to connect with postgressql database and send my data to be saved in it and monitor them on grafana Dashboard I picked these three metrics to be monitored
  `drift_score` ,    `number_of_drifted_columns`  , `share_of_missing_values`
  ## tests/model_test.py:
    In the test file I tested two functionalities first is the accuarcy metric if it's above thershold(90%) ,
    second if the first row is predicted correctly
      
  ## How to Run:
    1. Download data folder

    2. Download Anaconda from `https://anaconda.org/`

    3. Create Anaconda Environment `conda create -n mlops python=3.9.16`

    4. Activate your environment  `conda activate mlops`
  
    5. Install requirements.txt  `pip install -r requirements.txt` or `conda install --file requirements.txt`

    6. Add you Email and password in Cred.py

    7. Run mflow `mlflow server --backend-store-uri=sqlite:///mlflow.db`

    8. open anaconda terminal and Launch Prefect `prefect server start`

    9. open another anaconda terminal and configure prefect settings by running `prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api`
  
    10. Run Loan_Approval_prediction.py  `python Loan_Approval_prediction.py`
  
    11. for deploying
  Create work pool named loan_approval:  `prefect deploy Loan_Approval_prediction.py:main_flow -n loan1 -p loan_approval`
  
  Get worker `prefect worker start --pool 'loan_approval'`
  
  Start Run  `prefect deployment run 'main-flow/loan1'`
  
    #### please note if you want the emails to be activited add your app password in Cred.py
    12. For Monitoring
  
  Run docker compose
        `docker-compose up --build`
  
  Run evidently file 
         `python .\evidently_metrics.py`

    13. for Best Practices:

    1. Activate Conda Enviroment  `conda activate mlops`

    2. install pytest  `conda install -n mlops pytest`

    3. Make sure mlflow is working by `mlflow server --backend-store-uri=sqlite:///mlflow.db`

    4. Make sure prefect is running  `prefect server start`

    5. Testing the code: unit tests with pytest

    Run model_test.py by:
    Modifing the path of both data and models to suit your local computer
    Opening anaconda terminal and run `pytest`

    6. Makefiles and make:

        a. download make using `conda install -c conda-forge m2w64-make`

        b. run in the terminal `mingw32-make`

    7. Git pre-commit hooks

        a. Creating a hook for the repo Run  `pre-commit install`

        b. Run `git add .pre-commit-config.yaml`

        c. run a git commit `git commit -m "testing" `

    8. Code quality: linting and formatting
    
    Run `black --diff .` , `isort --diff .`

    

