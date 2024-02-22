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

with open('config.yaml', encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

def load_data(path):
    header = pd.read_csv(path + 'header.csv')
    header = list(pd.Series(header.iloc[:,0]))

    train = pd.DataFrame(np.load(path + 'raw_train.npy'), columns=header)
    test = pd.DataFrame(np.load(path + 'raw_test.npy'), columns=header)

    y_train = train.pop('attack_cat') # change label, attack_cat ...
    y_test = test.pop('attack_cat') # change label, ...

    train, _, y_train, _ = train_test_split(train, y_train, test_size=0.9, random_state=42)

    return train, test, y_train, y_test

def save_results(results):
    # save results
    if config['resampling']:
        with open('results/' + config['dataset'].split("\\")[-2] + '/' + config['oversampling'] + '_' + config['undersampling'] + '/' + config['oversampling'] + '_' + config['undersampling'] +'.json', "w") as outfile:
            json.dump(results, outfile)
    else:
        with open('results/' + config['dataset'].split("\\")[-2] + '/' + 'results.json', "w") as outfile:
            json.dump(results, outfile)

def save_results_category(results, category, model_name):
    # create directory if it doesn't exist
    if not os.path.exists('results/' + config['dataset'].split('\\')[-2] + '/' + config['oversampling'] + '_' + config['undersampling']) and config['resampling']:
        os.makedirs('results/' + config['dataset'].split('\\')[-2] + '/' + config['oversampling'] + '_' + config['undersampling'])

    # save results
    if config['resampling']:
        with open('results/' + config['dataset'].split("\\")[-2] + '/' + config['oversampling'] + '_' + config['undersampling'] + '/' + str(category) + '_' + model_name + '.json', "w") as outfile:
            json.dump(results, outfile)
    else:
        with open('results/' + config['dataset'].split("\\")[-2] + '/' + str(category) + '_' + model_name + '.json', "w") as outfile:
            json.dump(results, outfile)

def save_model_category(model, category):
    # create directory if it doesn't exist
    if not os.path.exists('models/' + config['dataset'].split('\\')[-2] + '/' + config['oversampling'] + '_' + config['undersampling']) and config['resampling']:
        os.makedirs('models/' + config['dataset'].split('\\')[-2] + '/' + config['oversampling'] + '_' + config['undersampling'])
    
    # save model
    if config['resampling']:
        dump(model, 'models/' + config['dataset'].split('\\')[-2] + '/' + config['oversampling'] + '_' + config['undersampling'] + '/' + str(category) + '_' + type(model[-1]).__name__ + '.joblib')
    else:
        dump(model, 'models/' + config['dataset'].split('\\')[-2] + '/' + str(category) + '_' + type(model[-1]).__name__ + '.joblib')

def generate_binary_model(X_train, X_test, y_train, y_test, category):
    # Preprocessing
    for clf in config['classifiers']:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', eval(clf)())
        ])

        # TODO load clf parameters from config file
        if clf == 'SVC':
            pipeline.set_params(clf__probability=True)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        save_model_category(pipeline, category)
        evaluate_model(y_test, y_pred, category, pipeline, save_results=True)

def evaluate_model(y_test, y_pred, category, clf, save_results=True):
    # evaluate model
    results = classification_report_imbalanced(y_test, y_pred, output_dict=save_results)
    results['oversampling'] = config['oversampling']
    results['undersampling'] = config['undersampling']
    results = {str(k):str(v) for k,v in results.items()}
    save_results_category(results, category, type(clf[-1]).__name__)

def select_binary_models(dataset, oversampling, undersampling, n_categories):
    # for each category load json results and retrieve f1_score
    f1_scores = []
    max_f1_scores = []
    for category in range(n_categories):
        if config['resampling']:
            for clf in config['classifiers']:
                with open('results/' + dataset + '/' + oversampling + '_' + undersampling + '/' + str(category) + '_' + clf + '.json') as json_file:
                    data = json.load(json_file)
                    f1_scores.append((clf, float(data['avg_f1'])))
            max_f1_scores.append(max(f1_scores, key=lambda x: x[1]))
            f1_scores = []

    print(max_f1_scores)
    # return a list of paths to the best models 
    models = ['models/' + config['dataset'].split("\\")[-2] + '/' + config['oversampling'] + '_' + config['undersampling'] + '/' + str(id) + '_' + max_f1_scores[id][0] + '.joblib' for id in range(len(max_f1_scores))]
    
    return models


def two_step_ensemble(X_test, y_test, n_categories):   
    # Load models in a list full path
    """
    if config['resampling']:
        models = ['models/' + config['dataset'].split("\\")[-2] + '/' + config['oversampling'] + '_' + config['undersampling'] + '/' + file for file in os.listdir('models/' + config['dataset'].split('\\')[-2] + '/' + config['oversampling'] + '_' + config['undersampling'] + '/') if not os.path.isdir('models/' + config['dataset'].split('\\')[-2] + '/' + file)]
    else:
        models = ['models/' + config['dataset'].split("\\")[-2] + '/' + file for file in os.listdir('models/' + config['dataset'].split('\\')[-2] + '/') if not os.path.isdir('models/' + config['dataset'].split('\\')[-2] + '/' + file)]
    """

    # Select best algorithm for an specific resampling method
    models = select_binary_models(config['dataset'].split('\\')[-2], config['oversampling'], config['undersampling'], n_categories)

    # Load best model from Mejores folder
    # models = ['models/' + config['dataset'].split("\\")[-2] + '/Mejores/' + file for file in os.listdir('models/' + config['dataset'].split('\\')[-2] + '/Mejores/')]

    #retrieve model from models with id_legit from config.yaml
    legit_model = models[int(config['id_legit'])]
    
    print(models, legit_model)

    #remove legit model from models list
    models.remove(legit_model)

    # load models from models list
    models = [load(model) for model in models]

    print(models)

    # predict using legit model.
    legit_model = load(legit_model)
    y_pred_legit = legit_model.predict(X_test)

    # samples labeled with label 0 are legit, so we remove them from the dataset
    X_test_attack = X_test[y_pred_legit != 0]
    y_test_attack = y_test[y_pred_legit != 0]

    # predict with each model using x_test and save prediction in np array
    y_pred_prob_attack = np.zeros((len(X_test_attack), len(models)))
    for i, model in enumerate(models):
        #pipeline = load('models/' + config['dataset'].split("\\")[-2] + '/' + models[i])
        y_pred_prob_attack[:, i] = model.predict_proba(X_test_attack)[:,0]
    
    # for each row in predictions, get highest value for each row and save in np array
    y_pred_attack = np.argmax(y_pred_prob_attack, axis=1)

    # predictions with label equal or greeater than id_legit add 1 to the label
    y_pred_attack = np.vectorize(lambda x: x + 1 if x >= config['id_legit'] else x)(y_pred_attack)
    
    # samples labelled with label 0 are legit, change label to id_legit
    y_pred = np.vectorize(lambda x: config['id_legit'] if x == 0 else x)(y_pred_legit)

    # insert y_pred_attack in positions where y_pred_legit is not 0
    y_pred[y_pred_legit != 0] = y_pred_attack

    #evaluate model using y_test
    results = classification_report_imbalanced(y_test, y_pred, digits=4, output_dict=True)
    print(results)


    results['oversampling'] = config['oversampling']
    results['undersampling'] = config['undersampling']
    results = {str(k):str(v) for k,v in results.items()}

    save_results(results)
    


def main():
    X_train, X_test, y_train, y_test = load_data(config['dataset'])
    N_CATEGORIES = len(np.unique(y_train))

    print(np.unique(y_train, return_counts=True))

    # Binarize labels
    for category_id in range(len(np.unique(y_train))):
        X_train, X_test, y_train, y_test = load_data(config['dataset'])
        print('Generating binary model for category', category_id)
        if config['resampling']:
            # calculate threshold to decide between oversampling and undersampling
            samples_per_class = np.unique(y_train, return_counts=True)[1]
            threshold = (samples_per_class[category_id]/(N_CATEGORIES - 1))

            # generate a list from 0 to N_CATEGORIES excluding category_id
            categories = [i for i in range(N_CATEGORIES) if i != category_id]

            # get a dict of ids from classes with less samples than the threshold. Values will be the threshold
            os_sampling_strategy = {i: int(threshold) for i in categories if threshold > samples_per_class[i] }
            us_sampling_strategy = {i: int(threshold) for i in categories if threshold < samples_per_class[i]}

            oversampling = eval(config['oversampling'])(sampling_strategy=os_sampling_strategy)
            undersampling = eval(config['undersampling'])(sampling_strategy=us_sampling_strategy)

            print(os_sampling_strategy)
            print('s/c', samples_per_class[category_id], 'th', threshold)

            # create pipeline with oversampling and undersampling
            pipeline = imbPipeline([
                ('oversampling', oversampling),
                ('undersampling', undersampling),
            ])

            # TODO set params for RepeatedEditedNearestNeighbours from config file
            if config['undersampling'] == 'RepeatedEditedNearestNeighbours':
                pipeline.set_params(undersampling__n_neighbors=4)
                
            # fit pipeline
            # try catch to avoid InvalidParameterError
            try:
                X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)
            except:
                print('Dict not allowed for undersampling. Using default')
                undersampling = eval(config['undersampling'])()
                pipeline = imbPipeline([
                    ('oversampling', oversampling),
                    ('undersampling', undersampling),
                ])
                # TODO set params for RepeatedEditedNearestNeighbours from config file
                if config['undersampling'] == 'RepeatedEditedNearestNeighbours':
                    pipeline.set_params(undersampling__n_neighbors=4)
                X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)

            X_train = X_train_res
            y_train = y_train_res
            del X_train_res, y_train_res

        print(np.unique(y_train, return_counts=True))

        binarizer = np.vectorize(lambda x: 0 if x == category_id else 1)
        y_train_bin = binarizer(y_train)
        y_test_bin = binarizer(y_test)       

        generate_binary_model(X_train, X_test, y_train_bin, y_test_bin, category_id)
    
    two_step_ensemble(X_test, y_test, N_CATEGORIES)

if __name__== "__main__":
    main()