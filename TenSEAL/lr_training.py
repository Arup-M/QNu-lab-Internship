# -*- coding: utf-8 -*-
# lr_training.py

import numpy as np

# =========================== Traininng on Plaintext data ============================================
def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

LEARNING_RATE = 0.01
EPOCHS = 100
def train_logistic_regression_plaintext_data(X_train, y_train, lr=LEARNING_RATE, epochs=EPOCHS, verbose=False):
    """
    Train logistic regression model on plaintext data using batch gradient descent.

    Parameters:
        X_train (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y_train (np.ndarray): Label vector of shape (n_samples,)
        lr (float): Learning rate
        epochs (int): Number of training iterations
        verbose (bool): Print loss during training if True

    Returns:
        dict: Model dictionary with keys 'weights' and 'bias'
    """
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for epoch in range(epochs):
        # Linear model
        linear_output = np.dot(X_train, weights) + bias

        # Apply sigmoid
        predictions = sigmoid(linear_output)

        # Compute gradients
        errors = predictions - y_train
        dw = np.dot(X_train.T, errors) / n_samples
        db = np.mean(errors)

        # Update weights and bias
        weights -= lr * dw
        bias -= lr * db

        # Optionally print loss
        if verbose and epoch % 10 == 0:
            loss = -np.mean(
                y_train * np.log(predictions + 1e-9) + (1 - y_train) * np.log(1 - predictions + 1e-9)
            )
            print(f"[Epoch {epoch}] Loss: {loss:.4f}")

    return weights, bias

# ======================================================================================================


# =========================== Training on Encrypted Data ===============================================


import numpy as np

def encrypted_sigmoid_approx(enc_val):
    """
    Approximate sigmoid using a low-degree polynomial: 0.5 + 0.197x - 0.004x^3
    Suitable for CKKS encryption.
    """
    return enc_val.polyval([0.5, 0.197, 0, -0.004])




LEARNING_RATE = 0.01
EPOCHS = 1


def train_logistic_regression_on_encrypted_data(enc_X_train, y_train, lr=LEARNING_RATE, epochs=EPOCHS, verbose=False):
    """
    Train logistic regression on encrypted data using CKKS.
    Note: very limited update rule due to HE constraints.
    
    Parameters:
        enc_X_train (List[CKKSVector]): Encrypted input samples
        y_train (List[int]): Plaintext labels (0/1)
        tenseal_manager (TenSealManager): HE context manager
        lr (float): Learning rate
        epochs (int): Number of epochs
        verbose (bool): Logging

    Returns:
        Tuple[np.ndarray, float]: plaintext weights and bias
    """
    from config import DIMENSION as dim
    from client import CLIENT_CONTEXT_PATH
    from tenseal_utils import TenSealManager
    # Load context WITH secret key (It is illigal but we have no option currently)
    tenseal_manager = TenSealManager(scheme="ckks")
    #stenseal_manager.load(CLIENT_CONTEXT_PATH, scheme="ckks")
    weights = np.zeros(dim)
    bias = 0.0

    for epoch in range(epochs):
        if verbose:
            print(f"[Encrypted Training] Epoch {epoch + 1}/{epochs}")
        for i, (enc_x, y) in enumerate(zip(enc_X_train, y_train)):
            
            enc_dot = enc_x.dot(weights) + bias
            enc_prob = encrypted_sigmoid_approx(enc_dot)
            enc_error = enc_prob - y

            # Decrypt error to compute update (limitation due to lack of HE gradient)
            error = tenseal_manager.decrypt(enc_error)[0]
            x_plain = np.array(tenseal_manager.decrypt(enc_x))

            # Gradient step in plaintext
            weights -= lr * error * x_plain
            if np.any(np.abs(weights) > 1e3):
                weights = weights / 10**41  # normalize to control magnitude
            bias -= lr * error

    return weights, bias
