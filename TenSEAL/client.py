# -*- coding: utf-8 -*-
# client.py
# This script implements the client-side functionality to prepare and send encrypted training data to a server.
# # It uses the TenSEAL library to encrypt the training data features while leaving the labels unencrypted.
# # The encrypted data is serialized and saved to a file, which can then be sent to the server for training a Linear Means Classifier (LMC).

import pickle
import numpy as np
import logging
from tenseal_utils import TenSealManager

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

CLIENT_CONTEXT_PATH = "/shared/client_secret_context.ctx"
CONTEXT_PATH = "public_context.ctx"  # Path to save the serialized context
ENC_DATA_PATH = "encrypted_data.pkl"  # Path to save the encrypted data


def prepare_and_send_data_toserver( 
        X_train, y_train, 
        training_mode='plaintext',  
        scheme_type="ckks",
        client_context_path= CLIENT_CONTEXT_PATH,
        context_path=CONTEXT_PATH, 
        enc_data_path=ENC_DATA_PATH, 
        verbose=False):
    """
    client side function to prepare and send encrypted data to server.
    This function encrypts the training data (Note: encrypt features but no need to encrypt labels) using the TenSEAL manager,
    serializes the context, and saves it to a file.
    parameters:
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.
        tenseal_manager (TenSealManager): Instance of TenSealManager for encryption/decryption.
        context_path (str): Path to save the serialized context.
        verbose (bool): If True, print additional information during encryption.
    Returns:
        tuple: A tuple containing the list of encrypted vectors and the labels.
    """
    # Create an instance of TenSealManager
    tenseal_manager = TenSealManager(scheme=scheme_type, verbose=verbose)
    if verbose:
        logging.info(f"[Client] TenSealManager initialized with {scheme_type} scheme.")
    # Save the context with the secret key to a file
    tenseal_manager.save(CLIENT_CONTEXT_PATH)
    if verbose:
        logging.info(f"[Client] Context with secret key saved to {CLIENT_CONTEXT_PATH}.")

    # make the context public for server-side training removing the secret key
    # This allows the server to use the context without needing the secret key.
    tenseal_manager.make_context_public()
    if verbose:
        logging.info("[Client] Context is now public. Server can use it for training without secret key.")
    # save the context to a file
    tenseal_manager.save(CONTEXT_PATH)
    if verbose:
        logging.info(f"[Client] Public Context saved to {CONTEXT_PATH}.")

    if verbose:
        logging.info("[Client] Preparing and encrypting training data...")

    if training_mode.lower() == 'encrypted':

        # Initialize the TenSealManager with the specified scheme
        scheme_type = scheme_type.lower()
        if scheme_type not in ["ckks", "bfv"]:
            raise ValueError("Unsupported scheme type. Use 'ckks' or 'bfv'.")
        
        dim = X_train.shape[1]
        with open("config.py", "w") as f:
            f.write(f"DIMENSION = {dim}\n")
            
        # Encrypt the training features "X_train" using the TenSealManager
        enc_X_train = tenseal_manager.encrypt(X_train)
        y_train = y_train.tolist()  # labels are not encrypted, just converted to list
        if verbose:
            logging.info(f"[Client] Encrypted {len(enc_X_train)} training vectors using {scheme_type} scheme.")

        # save the encrypted data and unencrypted labels to a file
        enc_data_path = "encrypted_data.pkl"
        # Serialize each encrypted vector
        serialized_enc_X_train = [vec.serialize() for vec in enc_X_train]
        with open(enc_data_path, 'wb') as f:
            # Save the encrypted training data and labels as a pickle file
            # Note: enc_X_train is a list of encrypted vectors, y_train is a list of labels
            # The encrypted vectors are serialized using pickle, which allows for easy storage and retrieval
            # This file can be sent to the server for training the Linear Means Classifier (LMC)
            # The server will use the public context to decrypt the encrypted vectors during training
            pickle.dump((serialized_enc_X_train, y_train), f)  # save encrypted data as a pickle file
        if verbose:
            logging.info(f"[Client] Encrypted data saved to {enc_data_path}.")

    elif training_mode.lower() == 'plaintext':
        # If training mode is plaintext, just convert the labels to list
        y_train = y_train.tolist()
        enc_X_train = X_train.tolist()  # No encryption, just convert to list
        if verbose:
            logging.info("[Client] Training mode is plaintext. No encryption performed on training data.")

        # save the plaintext data to a file
        with open(enc_data_path, 'wb') as f:
            pickle.dump((enc_X_train, y_train), f)
        if verbose:
            logging.info(f"[Client] Plaintext data saved to {enc_data_path}.") 

    else:
        raise ValueError("Unsupported training mode. Use 'plaintext' or 'encrypted'.")

    logging.info("[Client] Data preparation and encryption completed. Ready to send to server.")
