import numpy as np
import pandas as pd
from qboost import QBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time 
from sklearn.preprocessing import MinMaxScaler

# Uploading data 

# data = pd.read_csv('Surgical-deepnet.csv', sep=",")
# X = np.array(data.drop("complication", axis=1))
# y = np.array(data["complication"])

data = pd.read_csv('fraud_detection_bank_dataset.csv', sep=",")
X = np.array(data.drop(['id',"targets"], axis=1))
y = np.array(data["targets"])

print(np.unique(y, return_counts=True))

# Normalisation 
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

n_features = X.shape[1]

def my_qboost_predictions():
    start_model = time.perf_counter_ns()
    clf4 = QBoostClassifier(X_train, y_train, lam = 0.01)
    end_model = time.perf_counter_ns()

    start_predict = time.perf_counter_ns()
    prediction = clf4.predict_class(X_test)
    end_predict = time.perf_counter_ns()

    runtime_model = (end_model - start_model) / 1e6
    runtime_predict = (end_predict - start_predict) / 1e6
    #clf4.report_baseline(X_test, y_test)

    return prediction, runtime_model, runtime_predict

def my_AdaBoost_predictions():
    start_model = time.perf_counter_ns()
    clf4 = AdaBoostClassifier(n_estimators=n_features)
    clf4.fit(X_train, y_train)
    end_model = time.perf_counter_ns()

    start_predict = time.perf_counter_ns()
    prediction = clf4.predict(X_test)
    end_predict = time.perf_counter_ns()

    runtime_model = (end_model - start_model) / 1e6
    runtime_predict = (end_predict - start_predict) / 1e6

    return prediction, runtime_model, runtime_predict

classifiers = [my_qboost_predictions, my_AdaBoost_predictions]

# Loading the data sets
np.set_printoptions(suppress=True) # Scientific notation 

# Shuffling the data for a random selection for training and test data
idx = np.arange(len(y))

import csv

with open('results_2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["classifier", "accuracy", "precision", "recall", "f1", "runtime model", "runtime predict"]
    writer.writerow(field)

    for clf in classifiers:

        accuracy_arr = []
        precision_arr = []
        recall_arr = []
        f1_arr = []
        runtime_model_arr = []
        runtime_predict_arr = []

        for _ in range(50): 

            np.random.shuffle(idx)

            # Using 2/3 of our data set for training, 1/3 for testing
            idx_train = idx[:2*len(idx)//3]
            idx_test = idx[2*len(idx)//3:]

            n_features = X.shape[1]
            # Setting up the data points for training and testing
            X_train = X[idx_train]
            X_test = X[idx_test]

            # Setting up the labels for training and testing.  Labels should be -1, +1 for QBoost.
            y_train = 2 * y[idx_train] - 1  
            y_test = 2 * y[idx_test] - 1

            prediction, runtime_model, runtime_predict = clf()

            accuracy_arr.append(accuracy_score(y_test, prediction))
            precision_arr.append(precision_score(y_test, prediction))
            recall_arr.append(recall_score(y_test, prediction))
            f1_arr.append(f1_score(y_test, prediction))
            runtime_model_arr.append(runtime_model)
            runtime_predict_arr.append(runtime_predict) # returns miliseconds
        


        print(clf.__name__)
        print(f'accuracy array: {accuracy_arr}')
        print(f'with mean: {np.mean(accuracy_arr):.4f}; and with STD: {np.std(accuracy_arr):.4f}\n')

        print(f'precision array: {precision_arr}')
        print(f'with mean: {np.mean(precision_arr):.4f}; and with STD: {np.std(precision_arr):.4f}\n')

        print(f'recall array: {recall_arr}')
        print(f'with mean: {np.mean(recall_arr):.4f}; and with STD: {np.std(recall_arr):.4f}\n')

        print(f'F1 array: {f1_arr}')
        print(f'with mean: {np.mean(f1_arr):.4f}; and with STD: {np.std(f1_arr):.4f}\n')

        print(f'Time model: {runtime_model_arr}')
        print(f'with mean: {np.mean(runtime_model_arr):.4f}; and with STD: {np.std(runtime_model_arr):.4f}\n\n')

        print(f'Time prediction: {runtime_predict_arr}')
        print(f'with mean: {np.mean(runtime_predict_arr):.4f}; and with STD: {np.std(runtime_predict_arr):.4f}\n\n')
        
        for i in range(len(accuracy_arr)):
                writer.writerow([clf.__name__, accuracy_arr[i], precision_arr[i], recall_arr[i], 
                                f1_arr[i], runtime_model_arr[i], runtime_predict_arr[i]])
