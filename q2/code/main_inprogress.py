# basics
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils

# sklearn models for experimentation
from sklearn import linear_model as LM

if __name__ == "__main__":
    print("Training...")

    num_train_files = 2308
    num_val_files = 524
    num_test_files = 20
    h = 0.1 # seconds between each timestep

    # Initializing array to contain the final data
    X_train = np.empty((0,8*10)) # number of features
    y_train_x = np.empty((0,))
    y_train_y = np.empty((0,))

    # Load training data
    for i in range(num_train_files):
        # Load Dataset
        X = pd.read_csv(os.path.join('..', 'data', 'train', 'X', 'X_' + str(i) + '.csv'))
        y = pd.read_csv(os.path.join('..', 'data', 'train', 'y', 'y_' + str(i) + '.csv'))

        features = np.empty((0,))
        for idx, name in np.ndenumerate(X.columns):
            if "role" in name:
                pos = X.iloc[:,idx[0]+2:idx[0]+4].to_numpy()
                agent_feature = np.concatenate(
                    (pos[-1,:], pos[-2,:],
                    utils.velocity(pos[-2,:], pos[-1,:], h), 
                    utils.acceleration(pos[-3,:], pos[-2,:], pos[-1,:], h))
                )
                if X[name][0] != 0.0 and "agent" in X[name][0]:
                    features = np.append(agent_feature, features)
                else:
                    features = np.append(features, agent_feature)

        X_train = np.append(X_train, features[np.newaxis,:], axis=0)
        y_train_x = np.append(y_train_x, y[" x"][0])
        y_train_y = np.append(y_train_y, y[" y"][0])

    # Train the model
    # # Linear Regression (No Regularization)
    # model_x = LM.LinearRegression().fit(X_train, y_train_x)
    # model_y = LM.LinearRegression().fit(X_train, y_train_y)

    # Linear Regression (L1)
    model_x = LM.Lasso().fit(X_train, y_train_x)
    model_y = LM.Lasso().fit(X_train, y_train_y)


 ##################################################################################################################
 # VALIDATE #
 ############
    print("Validating...")

    validation_error = 0
    # Load validation data
    for i in range(num_val_files):
        # Load Dataset
        X = pd.read_csv(os.path.join('..', 'data', 'val', 'X', 'X_' + str(i) + '.csv'))
        y = pd.read_csv(os.path.join('..', 'data', 'val', 'y', 'y_' + str(i) + '.csv'))

        # Just a sanity check to make sure that the validation data has 30 datapoints
        if y.to_numpy().shape[0] != 30:
            continue

        features = np.empty((0,8))
        for idx, name in np.ndenumerate(X.columns):
            if "role" in name:
                pos = X.iloc[:,idx[0]+2:idx[0]+4].to_numpy()
                agent_feature = np.concatenate(
                    (pos[-1,:], pos[-2,:],
                    utils.velocity(pos[-2,:], pos[-1,:], h), 
                    utils.acceleration(pos[-3,:], pos[-2,:], pos[-1,:], h))
                )
                if X[name][0] != 0.0 and "agent" in X[name][0]:
                    features = np.append(agent_feature[np.newaxis,:], features, axis=0)
                else:
                    features = np.append(features, agent_feature[np.newaxis,:], axis=0)

        X_val = features.flatten()[np.newaxis,:]
        y_train_x = np.append(y_train_x, y[" x"][0])
        y_train_y = np.append(y_train_y, y[" y"][0])

        y_val = np.empty((0,2))

        # autoregressive prediciton 
        for j in range(1,31):
            new_features = np.empty((0,2))
            for k in range(10): 
                features[[0,k]] = features[[k,0]]
                X_val = features.flatten()[np.newaxis,:]
                pred_x = model_x.predict(X_val)[0]
                pred_y = model_y.predict(X_val)[0]
                if k == 0:
                    y_val = np.append(y_val, np.array([[pred_x, pred_y]]), axis=0)
                new_features = np.append(new_features, np.array([[pred_x, pred_y]]), axis=0)

            # Update test data with the prediction
            features = np.append(features[:, 2:],new_features, axis=1)
            
        validation_error += np.sum(np.power(y.loc[:,[" x", " y"]].to_numpy() - y_val, 2))/(y_val.size) # size is 2*30
    
    validation_error = np.sqrt(validation_error/num_val_files)
    print("Validation Error: {}".format(validation_error))

 ##################################################################################################################
 # TEST #
 ########

    # y_test = np.empty((0,2))
    # # Load test data
    # for i in range(num_test_files):
    #     # Load Dataset
    #     X = pd.read_csv(os.path.join('..', 'data', 'test', 'X', 'X_' + str(i) + '.csv'))

    #     for idx, name in np.ndenumerate(X.columns):
    #         if "role" in name and "agent" in str(X[name][0]):
    #             agent_pos = X.iloc[:,idx[0]+2:idx[0]+4]
    #             agent_pos = agent_pos.to_numpy().flatten()
    #             break

    #     X_test = agent_pos[np.newaxis,:]

    #     # autoregressive prediciton 
    #     for j in range(1,31):
    #         pred_x = model_x.predict(X_test)[0]
    #         pred_y = model_y.predict(X_test)[0]

    #         y_test = np.append(y_test, np.array([[str(i) + "_x_" + str(j), str(pred_x)]]), axis=0)
    #         y_test = np.append(y_test, np.array([[str(i) + "_y_" + str(j), str(pred_y)]]), axis=0)

    #         # Update test data with the prediction
    #         X_test = np.append(X_test[:,2:],np.array([[pred_x, pred_y]]), axis=1)

    # with open('submission.csv', mode='w') as submission_file:
    #     submission_write = csv.writer(submission_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    #     submission_write.writerow(['id', 'location'])
    #     for row in y_test:
    #         submission_write.writerow([row[0], row[1]])




    
        
        
        
                    

