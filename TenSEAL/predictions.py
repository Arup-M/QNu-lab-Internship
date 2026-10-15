# -*- coding: utf-8 -*-
# predictions.py
# This module contains functions for making predictions using a Linear Means Classifier (LMC) on encrypted data.
# It includes a function to predict the class of an encrypted test vector by finding the nearest class mean in Euclidean space.
# The class means are pre-computed during the training phase and stored in a dictionary.
# The prediction function uses the TenSEAL library for encrypted computations.



import tenseal as ts
import numpy as np
# Linear Means Classifier prediction function (input: encrypted data)
def predict_linear_mean_plaintext_model_byckks(enc_X_test, class_means, tenseal_manager, verbose=False):
    """
    Predicts class by finding the nearest class mean in Euclidean space.
    
    Parameters:
        enc_X_test (ts.CKKSVector): Encrypted test vector
        class_means (dict): Dictionary of class label -> mean vector
        tenseal_manager (TenSealManager): TenSEAL encryption manager
        
    Returns:
        pred_class (int): Predicted class label
    """
    # Check if enc_X_test is a CKKS encrypted vector
    assert isinstance(enc_X_test, ts.CKKSVector), "Expected CKKS encrypted vector"

    # Initialize minimum distance and predicted class
    min_dist = float('inf')
    pred_class = None

    for cls, mean_vec in class_means.items():
        # Encrypt the class mean vector
        mean_vec_array = np.array(mean_vec,dtype=np.float64)
        enc_mean_vec = tenseal_manager.encrypt(mean_vec_array)
        # Compute the distance between the encrypted test vector and the encrypted class mean
        enc_dist = enc_X_test - enc_mean_vec
        diff_sq = enc_dist* enc_dist
        dist = diff_sq.sum()

        # Decrypt the squared distance
        decrypted_dist = tenseal_manager.decrypt(dist) # it returns a list
        decrypted_dist = decrypted_dist[0]
        # Find the minimum distance and corresponding class
        if decrypted_dist < min_dist:
            min_dist = decrypted_dist
            pred_class = cls

    #if verbose:
    #    print(f"Predicted class: {pred_class} with distance: {min_dist}")
    return pred_class






# Linear Means Classifier prediction function for encrypted data using encrypted class means
# This function finds the nearest class mean in Euclidean space for encrypted data using encrypted class means
# It computes the distance between the encrypted test vector and each encrypted class mean,
# decrypts the squared distance, and returns the class with the minimum distance.
# This is useful when the class means are also encrypted, allowing for secure predictions without revealing the class means.
def predict_linear_mean_encrypted_model_byckks(enc_X_test, encrypted_class_means, tenseal_manager, verbose=False):
    """
    Predicts class by finding the nearest class mean in Euclidean space for encrypted data using encrypted class means.
    Parameters:
        enc_X_test (ts.CKKSVector): Encrypted test vector
        encrypted_class_means (dict): Dictionary of class label -> encrypted mean vector
        tenseal_manager (TenSealManager): TenSEAL encryption manager
        verbose (bool): If True, print additional information during prediction.
    Returns:
        pred_class (int): Predicted class label
    """

    #ensure enc_X_test is a CKKS encrypted vector
    assert isinstance(enc_X_test, ts.CKKSVector), "Expected CKKS encrypted vector"

    # Initialize minimum distance and predicted class
    min_dist = float('inf')
    pred_class = None


    for cls, enc_mean_vec_bytes in encrypted_class_means.items():
        enc_mean_vec = tenseal_manager.deserialize(enc_mean_vec_bytes)
        # Compute the distance between the encrypted test vector and the encrypted class mean
        enc_diff = enc_X_test - enc_mean_vec
        diff_sq = enc_diff * enc_diff
        enc_dist = diff_sq.sum()

        # Decrypt the squared distance
        if tenseal_manager.can_decrypt(enc_dist):
            decrypted_dist = tenseal_manager.decrypt(enc_dist) # it returns a list
            decrypted_dist = decrypted_dist[0]  # Get the first element since it's a single value
        else:
            raise ValueError("Cannot decrypt the encrypted distance. Ensure the context is public and the vector is encrypted correctly.")
        
        #if verbose:
        #    print(f"Class {cls}: Decrypted distance = {decrypted_dist}")

        # Find the minimum distance and corresponding class
        if decrypted_dist < min_dist:
            min_dist = decrypted_dist
            pred_class = cls

    #if verbose:
    #   print(f"Predicted class: {pred_class} with min distance: {min_dist}")

    # Return the predicted class
    return pred_class





# Linear Means Classifier prediction function for BFV encrypted data
# This function finds the nearest class mean in Euclidean space for BFV encrypted data.
# It computes the distance between the encrypted test vector and each class mean,
# decrypts the squared distance, and returns the class with the minimum distance.
# This is useful when the class means are already computed and stored as plaintext vectors.
# The input enc_X_test is expected to be a BFV encrypted vector.
# The class means should be integers since BFV only supports integer vectors.
def predict_linear_mean_plaintext_model_bybfv(enc_X_test, class_means, tenseal_manager, verbose=False):
    """
    Predicts class by finding the nearest class mean in Euclidean space for BFV encrypted data.
    Parameters:
        enc_X_test (ts.BFVVector): Encrypted test vector
        class_means (dict): Dictionary of class label -> mean vector
        tenseal_manager (TenSealManager): TenSEAL encryption manager
    Returns:
        pred_class (int): Predicted class label
    Note:
        This function assumes that the class means are already computed and stored as plaintext vectors.
        The input enc_X_test is expected to be a BFV encrypted vector.
        The class means should be integers since BFV only supports integer vectors.    
    """

    # Check if enc_X_test is a BFV encrypted vector
    assert isinstance(enc_X_test, ts.BFVVector), "Expected BFV encrypted vector"

    # Initialize minimum distance and predicted class
    min_dist = float('inf')
    pred_class = None

    for cls, mean_vec in class_means.items():
        # BFV only supports integer vectors, so we convert the mean vector to integers
        mean_vec_int = np.round(mean_vec).astype(int)
        # Encrypt the class mean vector
        mean_vec_array = np.array(mean_vec_int) 
        enc_mean_vec = tenseal_manager.encrypt(mean_vec_array)
        # Compute the distance between the encrypted test vector and the encrypted class mean
        enc_dist = enc_X_test - enc_mean_vec
        diff_sq = enc_dist * enc_dist
        dist = diff_sq.sum()

        # Decrypt the squared distance
        decrypted_dist = tenseal_manager.decrypt(dist) # it returns a list
        decrypted_dist = decrypted_dist[0]  # Get the first element since it's a single value
        # Find the minimum distance and corresponding class
        if decrypted_dist < min_dist:
            min_dist = decrypted_dist
            pred_class = cls
    
    #if verbose:
    #    print(f"Predicted class: {pred_class} with distance: {min_dist}")
    # Return the predicted class
    return pred_class








def predict_linear_mean_encrypted_model_bybfv(enc_X_test, encrypted_class_sums_counts, tenseal_manager, verbose=False):
    """
    Predict class using encrypted BFV data with encrypted class sums and counts.
    """
    assert isinstance(enc_X_test, ts.BFVVector), "Expected BFV encrypted vector"

    encrypted_class_sums, class_counts = encrypted_class_sums_counts

    min_dist = float('inf')
    pred_class = None

    for cls, enc_sum in encrypted_class_sums.items():
        count = class_counts[cls]
         
        enc_sum = tenseal_manager.deserialize(enc_sum)  # <-- Deserialize the bytes
        
        # Decrypt the encrypted sum vector
        decrypted_sum = np.array(tenseal_manager.decrypt(enc_sum))

        # Compute mean (integer division for BFV)
        mean_vec = (decrypted_sum // count).astype(int)

        # Re-encrypt the mean
        enc_mean_vec = tenseal_manager.encrypt(mean_vec)

        # Encrypted Euclidean distance
        enc_diff = enc_X_test - enc_mean_vec
        diff_sq = enc_diff * enc_diff
        enc_dist = diff_sq.sum()

        if tenseal_manager.can_decrypt(enc_dist):
            decrypted_dist = tenseal_manager.decrypt(enc_dist)[0]
        else:
            raise ValueError("Cannot decrypt the encrypted distance.")

        #if verbose:
        #    print(f"Class {cls}: Decrypted distance = {decrypted_dist}")

        if decrypted_dist < min_dist:
            min_dist = decrypted_dist
            pred_class = cls

    return pred_class



## Main prediction function that selects the appropriate prediction method based on training mode and scheme type
# This function checks the training mode ('plaintext' or 'encrypted') and the scheme type ('ckks' or 'bfv').
# It then calls the appropriate prediction function for the Linear Means Classifier (LMC).
# The function assumes that the class means are already computed and stored as plaintext vectors
# or encrypted vectors depending on the training mode.
# The input enc_X_test is expected to be a CKKS or BFV encrypted vector based on the scheme type specified.
# The class means should be integers since BFV only supports integer vectors
# or can be plaintext vectors for CKKS scheme.
def predict_linear_mean_model(enc_X_test, class_means, tenseal_manager, training_mod = 'plaintext', scheme_type="ckks", verbose=False):
    """
    Predicts class for an encrypted test vector using a Linear Means Classifier.
    Parameters:
        enc_X_test (ts.CKKSVector or ts.BFVVector): Encrypted test vector
        class_means (dict): Dictionary of class label -> mean vector (can be encrypted or plaintext)
        tenseal_manager (TenSealManager): TenSEAL encryption manager
        training_mod (str): 'plaintext' or 'encrypted' indicating the training mode
        scheme_type (str): 'ckks' or 'bfv' indicating the encryption scheme used
        verbose (bool): If True, print additional information during prediction.
    Returns:
        pred_class (int): Predicted class label 
    """

    if training_mod.lower() == 'plaintext':
        if scheme_type.lower() == 'ckks':
            return predict_linear_mean_plaintext_model_byckks(enc_X_test, class_means, tenseal_manager, verbose)
        elif scheme_type.lower() == 'bfv':
            return predict_linear_mean_plaintext_model_bybfv(enc_X_test, class_means, tenseal_manager, verbose)
        else:
            raise ValueError("Unsupported scheme type for plaintext model. Use 'ckks' or 'bfv'.")
    
    elif training_mod.lower() == 'encrypted':
        if scheme_type.lower() == 'ckks':
            return predict_linear_mean_encrypted_model_byckks(enc_X_test, class_means, tenseal_manager, verbose)
        elif scheme_type.lower() == 'bfv':
            # Expect class_means to be a tuple of (encrypted_class_sums, class_counts)
            return predict_linear_mean_encrypted_model_bybfv(enc_X_test, class_means, tenseal_manager, verbose)
        else:
            raise ValueError("Unsupported scheme type for encrypted model. Use 'ckks' or 'bfv'.")
    
    else:
        raise ValueError("Unsupported training mode. Use 'plaintext' or 'encrypted'.")
    # Note: The function assumes that the class means are already computed and stored as plaintext vectors
    # or encrypted vectors depending on the training mode.
    # The input enc_X_test is expected to be a CKKS or BFV encrypted vector
    # based on the scheme type specified.
    # The class means should be integers since BFV only supports integer vectors.




# ================= Prediction for Plaintext Logistic Regression =================
from tenseal_utils import TenSealManager
def predict_logistic_regression_plaintext_model_byckks(enc_X_test, plaintext_model,  tenseal_manager , verbose=False):
    """"
    Predict class for an encrypted test vector using a plaintext logistic regression model.
    Parameters:
        enc_X_test (ts.CKKSVector): Encrypted test vector
        plaintext_model (dict): Dictionary containing 'weights' and 'bias' of the logistic regression model (weight, bias)
        tenseal_manager (TenSealManager): TenSEAL encryption manager
        verbose (bool): If True, print additional information during prediction.
    Returns:
        pred_class (int): Predicted class label (0 or 1)
    """

    weights = plaintext_model[0]
    bias = plaintext_model[1]
   
    # Encrypted dot product: <w, x>
    enc_dot = enc_X_test.dot(weights)
    # Add bias (plaintext)
    enc_logit = enc_dot + bias
    
    # Decrypt the logit to apply sigmoid in plaintext
    logit = tenseal_manager.decrypt(enc_logit)[0]

    # Apply sigmoid
    prob = 1 / (1 + np.exp(-logit))

    # Return binary class prediction
    pred_class = int(prob >= 0.5)
    return pred_class

# ==========================================================================================

# ========================= Predictions of encrypted Logistic Regression ===================


# predictions.py
import numpy as np
from tenseal_utils import TenSealManager

def approx_sigmoid(x):
    """
    Low-degree polynomial approximation of the sigmoid function.
    Suitable for use in homomorphic encryption context.
    
    Parameters:
        x (float or np.ndarray): Input value or array

    Returns:
        float or np.ndarray: Approximated sigmoid
    """
    return 0.5 + 0.197 * x - 0.004 * x ** 3



def predict_logistic_regression_encrypted_model_byckks(enc_x, enc_model, tenseal_manager, verbose=False):
    """
    Predict using plaintext model and encrypted input.

    Parameters:
        enc_x (CKKSVector): Encrypted input vector
        model (Tuple): (weights, bias)
        tenseal_manager (TenSealManager): Manager with secret context
        verbose (bool): If True, log internals

    Returns:
        int: predicted class (0 or 1)
    """
    weights, bias = enc_model

    # Compute <w, x> + b
    enc_dot = enc_x.dot(weights) + bias

    # Decrypt
    logit = tenseal_manager.decrypt(enc_dot)[0]
    #prob = 1 / (1 + np.exp(-logit))
    prob = approx_sigmoid(logit)
    prob = np.clip(prob, -999, 999)
    pred = int(prob >= 0.5)

    if verbose:
        print(f"[Predict] Decrypted logit: {logit:.4f}, prob: {prob:.4f}, class: {pred}")
    return pred

def pred_logistic_regression_model(enc_X_test, model, tenseal_manager, training_mod = 'plaintext', scheme_type="ckks", verbose=False):
    if training_mod.lower() == 'plaintext':
        return predict_logistic_regression_plaintext_model_byckks(enc_X_test, model,  tenseal_manager , verbose=False)
    elif training_mod.lower() == 'encrypted':
        predict_logistic_regression_encrypted_model_byckks(enc_X_test, model, tenseal_manager, verbose=False)
    else:
         raise ValueError("Unsupported training mode. Use 'plaintext' or 'encrypted'.")

