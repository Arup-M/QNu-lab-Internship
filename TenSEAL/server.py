# -*- coding: utf-8 -*-
# server.py
# This script implements the server-side training of a Linear Means Classifier (LMC) on encrypted data using the TenSEAL library.
# The training data is encrypted and sent from the client side, and the server decrypts it to train the model.
# The trained model is then saved to a file for future use or evaluation.
import os
import pickle
import logging
from tenseal_utils import TenSealManager 
from lm_training import train_linear_mean_classifier

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

ENCRYPTED_DATA_PATH = "encrypted_data.pkl"  # Path to the encrypted data file
PUBLIC_CONTEXT_PATH = "public_context.ctx"  # Path to save the public context file
OUTPUT_MODEL_PATH = "encrypted_lmc_model.pkl"  # Path to save the trained model


def server_train_plaintext_data(verbose=False):
    """
    Server side function to train a Linear Means Classifier on plaintext data.
    This function receives the plaintext training data, trains the model, and saves it.
    
    Parameters:
        verbose (bool): If True, print additional information during training.
    
    Returns:
        dict: A dictionary containing the class means for each class.
    """
    if verbose:
        logging.info("[Server] Loading plaintext training data...")

    # Check input files
    if not os.path.exists(ENCRYPTED_DATA_PATH):
        raise FileNotFoundError(f"[Error] Required file not found: {ENCRYPTED_DATA_PATH}")

    # Load the plaintext data from file
    if verbose:
        logging.info(f"[Server] Loading plaintext training data from {ENCRYPTED_DATA_PATH}...")
    with open(ENCRYPTED_DATA_PATH, 'rb') as f:
        X_train, y_train = pickle.load(f)

    if verbose:
        logging.info(f"[Server] Loaded {len(X_train)} plaintext training vectors and calling 'train_linear_mean_classifier'...")

    # Train the Linear Means Classifier on plaintext data
    lmc_model = train_linear_mean_classifier(X_train, y_train, training_mode='plaintext')

    if verbose:
        logging.info("[Server] Training completed. Saving the trained model...")

    # Save the trained model to a file
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump(lmc_model, f)

    if verbose:
        logging.info(f"[Server] Trained model saved to {OUTPUT_MODEL_PATH}.")
    
    return lmc_model  # Return the trained model for further use or evaluation


def server_train_encrypted_data_byckks(verbose=False):
    tenseal_manager = TenSealManager(verbose=verbose)
    tenseal_manager.load(PUBLIC_CONTEXT_PATH)  # Load the public context

    if verbose:
        logging.info("[Server] Loading encrypted training data...")

    # Check input files
    for file_path in [ENCRYPTED_DATA_PATH, PUBLIC_CONTEXT_PATH]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Error] Required file not found: {file_path}")

    # Load the encrypted data from file
    with open(ENCRYPTED_DATA_PATH, 'rb') as f:
        serialized_enc_X_train, y_train = pickle.load(f)

    # Deserialize each encrypted vector
    enc_X_train = [tenseal_manager.deserialize(vec_bytes) for vec_bytes in serialized_enc_X_train]
    if verbose:
        logging.info(f"[Server] Loaded {len(enc_X_train)} encrypted training vectors.")

    lmc_model = train_linear_mean_classifier(enc_X_train, y_train, training_mode="encrypted", scheme="ckks")
    if verbose:
        logging.info("[Server] Training completed. Saving the trained model...")

    serialized_lmc_model = {cls: enc_mean.serialize() for cls, enc_mean in lmc_model.items()}
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump(serialized_lmc_model, f)
    if verbose:
        logging.info(f"[Server] Trained model saved to {OUTPUT_MODEL_PATH}.")

    return serialized_lmc_model  # Return the trained model for further use or evaluation 


def server_train_encrypted_data_bybfv(verbose=False):
    tenseal_manager = TenSealManager(verbose=verbose)
    tenseal_manager.load(PUBLIC_CONTEXT_PATH)  # Load the public context

    if verbose:
        logging.info("[Server] Loading encrypted training data...")

    # Check input files
    for file_path in [ENCRYPTED_DATA_PATH, PUBLIC_CONTEXT_PATH]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Error] Required file not found: {file_path}")

    # Load the encrypted data from file
    with open(ENCRYPTED_DATA_PATH, 'rb') as f:
        serialized_enc_X_train, y_train = pickle.load(f)

    # Deserialize each encrypted vector
    enc_X_train = [tenseal_manager.deserialize(vec_bytes) for vec_bytes in serialized_enc_X_train]
    if verbose:
        logging.info(f"[Server] Loaded {len(enc_X_train)} encrypted training vectors.")

    lmc_model = train_linear_mean_classifier(enc_X_train, y_train, training_mode="encrypted", scheme="bfv")
    if verbose:
        logging.info("[Server] Training completed. Saving the trained model...")
    
    # lmc_model is a tuple: (class_sums: dict[int, BFVVector], class_counts: dict[int, int])
    class_sums, class_counts = lmc_model
    # Serialize the encrypted vectors before pickling
    serialized_class_sums = {cls: vec.serialize() for cls, vec in class_sums.items()}
    serialized_lmc_model = (serialized_class_sums, class_counts)

    # Save the trained model to a file
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump(serialized_lmc_model, f)

    if verbose:
        logging.info(f"[Server] Trained model saved to {OUTPUT_MODEL_PATH}.")

    return serialized_lmc_model  # Return the serialized model


def server_train_linear_mean_classifier(training_mode='plaintext', scheme='ckks', verbose=False):
    """
    Server side function to train a Linear Means Classifier (LMC) on encrypted data.
    
    Parameters:
        training_mode (str): 'plaintext' for training on plaintext data, 'encrypted' for encrypted data.
        scheme (str): 'ckks' or 'bfv' for the encryption scheme.
        verbose (bool): If True, print additional information during training.
    
    Returns:
        dict: A dictionary containing the class means for each class.
    """
    if training_mode.lower() == 'plaintext':
        return server_train_plaintext_data(verbose)
    elif training_mode.lower() == 'encrypted':
        if scheme.lower() == 'ckks':
            return server_train_encrypted_data_byckks(verbose)
        elif scheme.lower() == 'bfv':
            return server_train_encrypted_data_bybfv(verbose)
        else:
            raise ValueError("Unsupported scheme type. Use 'ckks' or 'bfv'.")
    else:
        raise ValueError("Unsupported training mode. Use 'plaintext' or 'encrypted'.")


# =================== For Logistic Regression Training on Plaintext data by ckks  ===================
import numpy as np
from lr_training import train_logistic_regression_plaintext_data

def server_train_logistic_regression_on_plaintext_byckks(verbose=False):
    """""
    Server side function to train a logistic regression model on plaintext data.
    This function receives the plaintext training data, trains the model, and saves it.
    Parameters:
        verbose (bool): If True, print additional information during training.
    Returns:
        tuple: A tuple containing the trained weights and bias.
    """

    if verbose:
        logging.info("[Server] Loading plaintext training data for logistic regression...")

    # Check input files
    if not os.path.exists(ENCRYPTED_DATA_PATH):
        raise FileNotFoundError(f"[Error] Required file not found: {ENCRYPTED_DATA_PATH}")

    # Load the plaintext data from file
    with open(ENCRYPTED_DATA_PATH, 'rb') as f:
        X_train, y_train = pickle.load(f)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    if verbose:
        logging.info(f"[Server] Loaded {len(X_train)} plaintext training vectors for logistic regression.")

    # Train the logistic regression model on plaintext data
    weights, bias = train_logistic_regression_plaintext_data(X_train, y_train, verbose=verbose)

    if verbose:
        logging.info("[Server] Logistic regression training completed. Saving the trained model...")

    # Save the trained model to a file
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump({'weights': weights, 'bias': bias}, f)

    if verbose:
        logging.info(f"[Server] Trained logistic regression model saved to {OUTPUT_MODEL_PATH}.")
        logging.info(f"[Server] Final weights shape: {weights.shape}")
        logging.info(f"[Server] Final bias: {bias:.4f}")

    return weights, bias  # Return the trained weights and bias

# ==============================================================================================

# ================= Logistic Regrassion Training on Encrypted Data by ckks ======================

import pickle
from client import CLIENT_CONTEXT_PATH
from lr_training import train_logistic_regression_on_encrypted_data
from tenseal_utils import TenSealManager

ENC_DATA_PATH = "encrypted_data.pkl"
PUBLIC_CONTEXT_PATH = "public_context.ctx"

def server_train__logistic_regression_on_ciphertext_byckks(verbose=False):
    """
    Server-side training function for encrypted logistic regression.
    Loads encrypted training data and public context to train model.

    Returns:
        Tuple[np.ndarray, float]: plaintext weights and bias
    """

    # Load context WITH secret key (It is illigal but we have no option currently)
    tenseal_manager = TenSealManager(scheme="ckks")
    #tenseal_manager.load(CLIENT_CONTEXT_PATH, scheme="ckks")

    # Load encrypted data and labels
    with open(ENC_DATA_PATH, 'rb') as f:
        serialized_enc_X_train, y_train = pickle.load(f)

    if verbose:
        logging.info(f"[Server] Loaded {len(serialized_enc_X_train)} encrypted training samples.")

    # Deserialize encrypted vectors
    enc_X_train = [tenseal_manager.deserialize(s) for s in serialized_enc_X_train]

    # Train model
    weights, bias = train_logistic_regression_on_encrypted_data(enc_X_train, y_train,  verbose=verbose)
    if verbose:
        logging.info(f"[Server] Training completed.")
    return weights, bias

# ========================================================================================================


# ================================ Binding All Type of Training Mode ======================================

def server_train_logistic_regression(training_mode, scheme='ckks', verbose=False):
    if training_mode.lower() == 'plaintext':
        return server_train_logistic_regression_on_plaintext_byckks(verbose=verbose)
    
    elif training_mode.lower() == 'encrypted':
        if scheme == 'ckks':
            return server_train__logistic_regression_on_ciphertext_byckks(verbose=verbose)
        else:
            raise NotImplementedError("Only CKKS encrypted training is implemented.")
    
    else:
        raise ValueError("Unsupported training mode.")

# =========================================================================================================
