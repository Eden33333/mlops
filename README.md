# MLOps

## 2. mlflow
### 2.1 Requirement
#### 2.1.1 Vitural Environment
Vitural environment is for minimumn dependency to install package
1. Create vene by conda
  `conda create -n my_enev`
2. Activate the environment
   `conda activate my_enev`
3. Install package
   `conda install --file requirement.txt `
   *requirement.txt*:
   ```
      python==3.9
      mlflow
      jupyter
      sckit-learn
      panadas
      seaborn
      hyperopt
      xgboost
    ```

#### 2.1.2 Configure the mlflow backend(storage)
##### 2.1.2.1 Why we need backend and what's that:
1. Centralised tracking: default mlflow making it hard to collaborate and fill you memory with too many files
2. Scalability: Scale up quickly and efficient to query
3. Persistent
4. Remote Access
5. Model registry
-   Backend store:  A persistent store for various run metadata. i.e properties such as parameters, run_id, run_time, start, and end times. This would typically be a database such as a SQLite one. But could also just be local files too.
-   Artifast store: This persists large data associated for each model, such as model weights (e.g. a pickled scikit-learn model), images (e.g. PNGs), model and data files (e.g. Parquet file). MLflow stores artifacts ina a local file (mlruns) by default, but also supports different storage options such as Amazon S3 and Azure Blob Storage. See set up #3 in the image above.
**Note:** Backend store just store meta data which can be stored in database while Artifest store like package and model which need to be stored in file/cloud storage
##### 2.1.2.2 Connect the backend 
1. You can run MLflow easily with
`mlflow ui`
2. Create SQLite backend in local computer
`mlflow ui --backend-store-uri sqlite:///path/to/mlflow.db`
3. create database in your local computer but it can also be sent to remote service
`mlflow service --backend-store-uri sqlite:///mlflow.db`
4. storing artifast
`mlflow service --backend-store-uri sqlite:///mlflow.db --defaut-artifast-root /artifast`
5.  Connect artifact remotely

### 2.2 basic syntax running in jupyter_notebook

- Initialize
-   'mlflow.set_track_uri("sqlite:///mlflow.db")'
-   'mlflow.set_experienment("new_experience")'
- Track the model
-  Basic way
    ```
    0 mlflow.xgboost.autolog(disable=True)
    1 With mlflow.start_run():
    2    mlflow.set_tag("developer","hanchen") #load developer
            #Model init
            booster = xgb.train(
                params=best_params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )
     3       mlflow.log_params(params) # record parameters       
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
    4       mlflow.log_metrics("rmse",rmse)
    5       mlflow.xgboost.log_model("booster",artifact_path="models_mlflow") # customized model mlflow.pyfunc.log_model()
    ```
- Easy way but for specific models
   `mlflow.xgboost.autolog()` # this is a independent syntax, you don't need to put within `with mlflow.start_run()`
  Autolog will allow you to "log metrics, parameters, and models without the need for explicit log statements". However this is currently supported for a few libraries (but most of the ones you would normally use). Just for the following pacakge:
  ```
  Scikit-learn
  TensorFlow and Keras
  Gluon
  XGBoost
  LightGBM
  Statsmodels
  Spark
  Fastai
  Pytorch
  ```
- Save the artifact
-   simple case, as before shows use `log_model`
-   Other case, like preprocessing step:
```    
    lr = Lasso(alpha=0.001)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_val)
    mean_squared_error(y_val, y_pred, squared=False)
    # You need to create model file first
    with open("model/lr.bin","wb") as f: # the extension *bin* is binary format
      pickle.dump((dv,lr),f)
```

- Use the model
  ```
  xgb = mlflow.xgboost.loae_moedel(model.URI)
  xgb.predict()
  ```

### 2.3 Model Registry
1. You can regist the model by [this link](https://mlflow.org/docs/latest/model-registry.html) after `mlflow.{}.log_model("",model_param)`
2. You can also regist model programatically by 
```
from mlflow.tracking import MlflowClient
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
client = MlflowClient(tracking_uri = MLFLOW_TRACKING_URI )

experience = client.search_experiments("experience_name")
runs = client.search_runs(
    experiment_ids = experience.id,
    filter_string = "",
    run_view_type = ViewType.ACTIVE_ONLY,
    max_results = top_n, # 5-10
    order_by = ["metrics.val_rmse ASC"]
    )


```
This is not to deploy the model but is to help you to choose which model is ready to deploy. There are three model versions
```
for run in runs:
  run_id = run.info.run_id
  run_path = f"runs:/{run_id}/model
  mlflow.register_model(model_uri = run_path,name="modle_name")
```
### 2.4 MLclient Class
This is programatically to help search the model
```
  from mlflow.tracking import MLflowClient
  MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
  client = MlfowClient(tracking_uri=MLFLOW_TRACKING_URI)
1 experience = client.search_experience()
  for exp in experience:
    print()
2 run = client.search_run()
client.get_lastes_version(name = model_name)
```
