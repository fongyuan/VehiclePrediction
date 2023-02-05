# Note: This is an implementation that does not use auto-regression.
# Working on 220 features, I'll build 30 models to predict each timestep.

# basics
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils
from Regression import LassoRegression

# sklearn models for experimentation
from sklearn import linear_model as LM
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

if __name__ == "__main__":
    print("Training...")

    num_train_files = 2308
    num_val_files = 524
    num_test_files = 20
    h = 0.1 # seconds between each timestep

    # Initializing array to contain the final data
    X_train = np.empty((0,70)) # number of features
    y_train_x = np.empty((0,30))
    y_train_y = np.empty((0,30))

    # Load training data
    for i in range(num_train_files):
        # Load Dataset
        X = pd.read_csv(os.path.join('..', 'data', 'train', 'X', 'X_' + str(i) + '.csv'))
        y = pd.read_csv(os.path.join('..', 'data', 'train', 'y', 'y_' + str(i) + '.csv'))

        # Just a sanity check to make sure that the training data has 30 datapoints
        if y.to_numpy().shape[0] != 30:
            continue

        features = np.empty((0,))
        for idx, name in np.ndenumerate(X.columns):
            if "role" in name:
                pos = X.iloc[:,idx[0]+2:idx[0]+4].to_numpy()

                pos = np.concatenate(
                    (pos[-1,:], 
                    utils.velocity(pos[-2,:], pos[-1,:], h), 
                    utils.acceleration(pos[-3,:], pos[-2,:], pos[-1,:], h))
                )

                if X.iloc[0,idx[0]+1] != 0.0 and "car" in X.iloc[0,idx[0]+1]:
                    pos = np.append(pos, 1)
                else:
                    pos = np.append(pos, 0)
                # Make sure that the agent positions come first in all feature vectors
                if X[name][0] != 0.0 and "agent" in X[name][0]:
                    features = np.append(pos, features)
                else:
                    features = np.append(features, pos)

        X_train = np.append(X_train, features[np.newaxis,:], axis=0)
        y_train_x = np.append(y_train_x, y[" x"].to_numpy()[np.newaxis,:], axis=0)
        y_train_y = np.append(y_train_y, y[" y"].to_numpy()[np.newaxis,:], axis=0)

    # Train the model
    num_models = 30
    models_x = []
    models_y = []

    for i in range(num_models):
        # Linear Regression (L1)
        # models_x.append(LM.Lasso(alpha=0.01, max_iter=4000).fit(X_train, y_train_x[:,i]))
        # models_y.append(LM.Lasso(alpha=0.01, max_iter=4000).fit(X_train, y_train_y[:,i]))
        # models_x.append(Ridge(alpha=1).fit(X_train, y_train_x[:,i]))
        # models_y.append(Ridge(alpha=1).fit(X_train, y_train_y[:,i]))
        # models_x.append(MLPRegressor().fit(X_train, y_train_x[:,i]))
        # models_y.append(MLPRegressor().fit(X_train, y_train_y[:,i]))
        models_x.append(LassoRegression(lammy=0.01, maxEvals=3000))
        models_x[i].fit(X_train, y_train_x[:,i][:,np.newaxis])
        models_y.append(LassoRegression(lammy=0.01, maxEvals=3000))
        models_y[i].fit(X_train, y_train_y[:,i][:,np.newaxis])


 ##################################################################################################################
 # VALIDATE #
 ############
    print("Validating...")

    validation_error = 0
    X_val = np.empty((0,70)) # number of features
    y_val_x = np.zeros((0,30))
    y_val_y = np.zeros((0,30))

    # Load validation data
    for i in range(num_val_files):
        # Load Dataset
        X = pd.read_csv(os.path.join('..', 'data', 'val', 'X', 'X_' + str(i) + '.csv'))
        y = pd.read_csv(os.path.join('..', 'data', 'val', 'y', 'y_' + str(i) + '.csv'))

        # Just a sanity check to make sure that the validation data has 30 datapoints
        if y.to_numpy().shape[0] != 30:
            continue

        features = np.empty((0,))
        for idx, name in np.ndenumerate(X.columns):
            if "role" in name:
                pos = X.iloc[:,idx[0]+2:idx[0]+4].to_numpy()
                pos = np.concatenate(
                    (pos[-1,:],
                    utils.velocity(pos[-2,:], pos[-1,:], h), 
                    utils.acceleration(pos[-3,:], pos[-2,:], pos[-1,:], h))
                )

                if X.iloc[0,idx[0]+1] != 0.0 and "car" in X.iloc[0,idx[0]+1]:
                    pos = np.append(pos, 1)
                else:
                    pos = np.append(pos, 0)

                # Make sure that the agent positions come first in all feature vectors
                if X[name][0] != 0.0 and "agent" in X[name][0]:
                    features = np.append(pos, features)
                else:
                    features = np.append(features, pos)

        X_val = np.append(X_val, features[np.newaxis,:], axis=0)
        y_val_x = np.append(y_val_x, y[" x"].to_numpy()[np.newaxis,:],axis=0)
        y_val_y = np.append(y_val_y, y[" y"].to_numpy()[np.newaxis,:],axis=0)
    
    ypred_x = np.zeros((X_val.shape[0],30))
    ypred_y = np.zeros((X_val.shape[0],30))
    for i in range(num_models):
        ypred_x[:,i] = models_x[i].predict(X_val).flatten()
        ypred_y[:,i] = models_y[i].predict(X_val).flatten()

    validation_error = np.sum(np.power(y_val_x - ypred_x, 2)) + np.sum(np.power(y_val_y - ypred_y, 2)) 
    
    validation_error = np.sqrt(validation_error/(ypred_x.size*2))
    print("Validation Error: {}".format(validation_error))

 ##################################################################################################################
 # TEST #
 ########

    X_test = np.empty((0,70)) # number of features

    # Load test data
    for i in range(num_test_files):
        # Load Dataset
        X = pd.read_csv(os.path.join('..', 'data', 'test', 'X', 'X_' + str(i) + '.csv'))

        features = np.empty((0,))
        for idx, name in np.ndenumerate(X.columns):
            if "role" in name:
                pos = X.iloc[:,idx[0]+2:idx[0]+4].to_numpy()
                pos = np.concatenate(
                    (pos[-1,:],
                    utils.velocity(pos[-2,:], pos[-1,:], h), 
                    utils.acceleration(pos[-3,:], pos[-2,:], pos[-1,:], h))
                )

                if X.iloc[0,idx[0]+1] != 0.0 and "car" in X.iloc[0,idx[0]+1]:
                    pos = np.append(pos, 1)
                else:
                    pos = np.append(pos, 0)

                # Make sure that the agent positions come first in all feature vectors
                if X[name][0] != 0.0 and "agent" in X[name][0]:
                    features = np.append(pos, features)
                else:
                    features = np.append(features, pos)

        X_test = np.append(X_test, features[np.newaxis,:], axis=0)

    # Predictions
    ypred_x = np.zeros((X_test.shape[0],30))
    ypred_y = np.zeros((X_test.shape[0],30))
    for i in range(num_models):
        ypred_x[:,i] = models_x[i].predict(X_test).flatten()
        ypred_y[:,i] = models_y[i].predict(X_test).flatten()

    # Mapping predictions in a submittable format
    y_test = np.empty((0,2))
    for i in range(ypred_x.shape[0]):
        for j in range(1,31):
            y_test = np.append(y_test, np.array([[str(i) + "_x_" + str(j), str(ypred_x[i,j-1])]]), axis=0)
            y_test = np.append(y_test, np.array([[str(i) + "_y_" + str(j), str(ypred_y[i,j-1])]]), axis=0)

    with open('submission.csv', mode='w') as submission_file:
        submission_write = csv.writer(submission_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        submission_write.writerow(['id', 'location'])
        for row in y_test:
            submission_write.writerow([row[0], row[1]])




    
        
        
        
                    

