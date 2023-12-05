import sys
import os
import numpy as np
import pandas as pd
import yaml
from collections import Counter
from yaml.loader import SafeLoader
from joblib import dump, load
from pprint import pprint
import json

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss, RepeatedEditedNearestNeighbours, TomekLinks
from imblearn.pipeline import Pipeline as imbPipeline

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.metrics import classification_report_imbalanced

import json

def evaluate_model(y_test, y_pred, category, clf, save_results=True):
    # evaluate model
    results = classification_report_imbalanced(y_test, y_pred, output_dict=save_results)
    results['oversampling'] = config['oversampling']
    results['undersampling'] = config['undersampling']
    results = {str(k):str(v) for k,v in results.items()}

def generate_binary_model(X_train, X_test, y_train, y_test, category):
    # Preprocessing
    for clf in config['classifiers']:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', eval(clf)())
        ])

        if clf == 'SVC':
            pipeline.set_params(clf__probability=True)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        #save_model_category(pipeline, category)
        evaluate_model(y_test, y_pred, category, pipeline, save_results=True)


with open('config.yaml', encoding='utf8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

