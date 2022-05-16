import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA

random_state = 42

data_path = "data"
train_path = os.path.join(data_path, 'train.csv')
train_data = pd.read_csv(train_path, index_col='Id')
test_path = os.path.join(data_path, 'test.csv')
test_data = pd.read_csv(test_path, index_col='Id')



def create_pipeline( model_params, random_state):
    pipeline_steps = []
    pipeline_steps.append(("scaler", StandardScaler()))

    #pipeline_steps.append(("PCA", PCA(n_components=0.95, svd_solver='full', random_state=random_state)))
    #pipeline_steps.append(("ICA", FastICA(n_components=15, random_state=random_state)))

#    pipeline_steps.append(("classifier", KNeighborsClassifier(**model_params)) )
    pipeline_steps.append(("classifier", RandomForestClassifier(random_state=random_state, **model_params)))

    return Pipeline(steps=pipeline_steps)




def test_run(pipeline, x, y):
    cv = StratifiedKFold(shuffle=True)
    with mlflow.start_run():
        results = cross_validate(pipeline, x, y, cv=cv, scoring='accuracy')
        mlflow.log_param("model", 'KNeighbors')
        for key, value in model_params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("FE", 'None')
        #    mlflow.log_param("FE", 'Aspect and Slope sc no drop')
        mlflow.log_metric('accuracy', np.mean(results['test_score']))


def final_run(pipeline, x, y, test_x):
    pipeline.fit(x, y)
    test_y = pipeline.predict(test_x)
    res = pd.DataFrame(test_y, index=test_x.index, columns=['Cover_Type'])
    res.to_csv('submission.csv')



x = train_data.drop("Cover_Type", axis=1)
#x['aspect_sin'] = np.sin(np.deg2rad(x['Aspect']))
#x['aspect_cos'] = np.cos(np.deg2rad(x['Aspect']))
#x['slope_sin'] = np.sin(np.deg2rad(x['Slope']))
#x['slope_cos'] = np.cos(np.deg2rad(x['Slope']))
#x = x.drop(['Aspect', 'Slope'], axis=1)


y = train_data["Cover_Type"]

#model_params = {'n_neighbors': 2}
model_params = {'n_estimators': 200}
pipeline = create_pipeline(
    random_state=random_state,
    model_params=model_params
)

#test_run(pipeline, x, y)
final_run(pipeline, x, y, test_data)
