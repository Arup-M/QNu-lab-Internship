## -*- coding: utf-8 -*-
# lm_training.py

import numpy as np
# Linear Means Classifier training on plaintext data
def linear_mean_train_on_plaintext(X_train, y_train):
    """
    Parameters:
        X_train: np.ndarray, shape (n_samples, n_features)
        y_train: np.ndarray, shape (n_samples,)
    Returns:
        dict of class_id: mean_vector
    """
    # Ensure X_train is a numpy array
    X_train = np.array(X_train) 
    y_train = np.array(y_train)  # Ensure y_train is a numpy array
     
    # Check if the number of samples in X_train matches the number of labels in y_train
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Number of samples in X_train must match number of labels in y_train.") 

    class_means = {}
    for cls in np.unique(y_train):
        class_means[cls] = X_train[y_train == cls].mean(axis=0)

    return class_means



def linear_mean_train_on_encrypted_data_byckks(enc_X_train, y_train):
    class_sums = {}
    class_counts = {}

    for enc_vector, label in zip(enc_X_train, y_train):
        if label not in class_sums:
            class_sums[label] = enc_vector
            class_counts[label] = 1
        else:
            class_sums[label] += enc_vector
            class_counts[label] += 1


    class_means = {}
    for label in class_sums:
        count = class_counts[label]
        enc_mean = class_sums[label] * (1 / count)
        class_means[label] = enc_mean

    return class_means


def linear_mean_train_on_encrypted_data_bybfv(enc_X_train, y_train):
    class_sums = {}
    class_counts = {}

    for enc_vector, label in zip(enc_X_train, y_train):
        if label not in class_sums:
            class_sums[label] = enc_vector
            class_counts[label] = 1
        else:
            class_sums[label] += enc_vector
            class_counts[label] += 1

    return class_sums, class_counts


def train_linear_mean_classifier(X_train, y_train, training_mode='plaintext', scheme='ckks'):
    
    if training_mode.lower() == 'plaintext':
        return linear_mean_train_on_plaintext(X_train, y_train)
    elif training_mode.lower() == 'encrypted':
        if scheme.lower() == 'ckks':
            return linear_mean_train_on_encrypted_data_byckks(X_train, y_train)
        elif scheme.lower() == 'bfv':
            return linear_mean_train_on_encrypted_data_bybfv(X_train, y_train)
        else:
            raise ValueError("Unsupported scheme type. Use 'ckks' or 'bfv'.")
    else:
        raise ValueError("Unsupported training mode. Use 'plaintext' or 'encrypted'.")