## -*- coding: utf-8 -*-
import os
import time
import numpy as np
import psutil
import tracemalloc
import pandas as pd
import logging
from tenseal_utils import TenSealManager
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from client import prepare_and_send_data_toserver
from server import server_train_linear_mean_classifier
from predictions import predict_linear_mean_model


CSV_FILE_PATH = "boston.csv"  # Default dataset file
OUTPUT_DIR = "results"  # Default output directory for metrics

def monitor_resource_usage():
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)  # warm up / reset the counters
    time.sleep(0.1)  # small delay
    cpu = process.cpu_percent(interval=0.5)  # measure over 0.5s
    ram = process.memory_info().rss / (1024 * 1024)
    return cpu, ram

# Run the benchmark for Linear Means Classifier by CKKS scheme in TenSEAL
# This function benchmarks the Linear Means Classifier using the CKKS scheme in TenSEAL.
# It prepares the data, trains the model, performs inference, and evaluates metrics.
# It supports both plaintext and encrypted training modes, and saves the results to a CSV file. 
def benchmark_tenseal_ckks_for_lmc(csv_path=CSV_FILE_PATH, output_dir=OUTPUT_DIR, scheme="ckks", training_mode="plaintext", verbose=False):
    """
    Run the benchmark for Linear Means Classifier using CKKS scheme in TenSEAL.
    Parameters:
        csv_path (str): Path to the CSV dataset file.
        output_dir (str): Directory to save the output metrics.
        scheme (str): Encryption scheme to use, default is 'ckks'.
        training_mode (str): 'plaintext' or 'encrypted' to select training approach.
        verbose (bool): If True, print detailed logs.
    Returns:
        None: Saves metrics to a CSV file in the specified output directory.
    """

    os.makedirs(output_dir, exist_ok=True)

    if training_mode == "plaintext":
        output_path = os.path.join(output_dir, "metrics_tenseal_ckks_lmc_plaintext.csv")
    elif training_mode == "encrypted":
        output_path = os.path.join(output_dir, "metrics_tenseal_ckks_lmc_encrypted.csv")
    else:
        raise ValueError("Invalid training_mode. Use 'plaintext' or 'encrypted'.")

    if verbose:
        logging.info(f"Loading dataset from {csv_path}...")

    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Convert regression target to binary classification if needed
    if df.iloc[:, -1].nunique() > 10:
        df.iloc[:, -1] = pd.qcut(df.iloc[:, -1], q=2, labels=[0, 1])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----------------------------------------
    # Data preparation and server-side training
    # ----------------------------------------
    process = psutil.Process(os.getpid())
    preparation_start = time.time()
    tracemalloc.start()  # Start tracing memory allocations
    CPU_before = process.cpu_percent(interval=None)  # warm up / reset the counters
    prepare_and_send_data_toserver(X_train, y_train, training_mode=training_mode, scheme_type=scheme, verbose=verbose)
    preparation_end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop tracing memory allocations
    prep_ram = peak / (1024 * 1024)  # Convert to MB

    CPU_after = process.cpu_percent(interval=None)
    prep_cpu = CPU_after - CPU_before


    process = psutil.Process(os.getpid())
    tracemalloc.start()  # Start tracing memory allocations
    CPU_before = process.cpu_percent(interval=None)  # warm up / reset the counters
    train_start = time.time()
    class_means = server_train_linear_mean_classifier(training_mode=training_mode, scheme=scheme, verbose=verbose)
    train_end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop tracing memory allocations
    train_ram = peak / (1024 * 1024)  # Convert to MB
    CPU_after = process.cpu_percent(interval=None)
    train_cpu = CPU_after - CPU_before
    

    # ----------------------------------------
    # Inference
    # ----------------------------------------
    tenseal_manager = TenSealManager(scheme=scheme)

    # Encrypt the test data
    enc_X_test = tenseal_manager.encrypt(X_test)

    if verbose:
        logging.info(f"Encrypted {len(enc_X_test)} test vectors using {scheme} scheme.")

   
    process = psutil.Process(os.getpid())
    tracemalloc.start()  # Start tracing memory allocations
    CPU_before = process.cpu_percent(interval=None)  # warm up / reset the counters
    infer_start = time.time()
    predictions = []
    for enc_vector in enc_X_test:
        pred = predict_linear_mean_model(enc_vector, class_means, tenseal_manager, training_mod=training_mode, scheme_type=scheme, verbose=verbose)
        predictions.append(pred)
    infer_end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop tracing memory allocations
    infer_ram = process.memory_info().rss / (1024 * 1024)
    CPU_after = process.cpu_percent(interval=None)
    infer_cpu = CPU_after - CPU_before
    

    # ----------------------------------------
    # Metrics
    # ----------------------------------------
    accuracy = accuracy_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    ciphertext_sizes = []
    plaintext_sizes = []
    noise_errors = []
    
    for i in range(len(enc_X_test)):
        enc_vec = enc_X_test[i]
        original_vec = X_test[i]
        serialized = enc_vec.serialize()
        ciphertext_sizes.append(len(serialized))
        plaintext_sizes.append(original_vec.nbytes)
        decrypted_vec = tenseal_manager.decrypt(enc_vec)
        noise_errors.append(np.max(np.abs(original_vec - decrypted_vec)))

    avg_ciphertext_size = np.mean(ciphertext_sizes)
    avg_plaintext_size = np.mean(plaintext_sizes)
    avg_expansion = np.mean([c/p for c, p in zip(ciphertext_sizes, plaintext_sizes)])
    avg_noise = np.mean(noise_errors)

    
    metrics = {
        "training_time": train_end - train_start,
        "preparation_time": preparation_end - preparation_start,
        "inference_time": infer_end - infer_start,
        "accuracy": accuracy,
        "mse": mse,
        "scheme": scheme,
        "training_mode": training_mode,
        "avg_ciphertext_size_bytes": avg_ciphertext_size,
        "avg_plaintext_size_bytes": avg_plaintext_size,
        "avg_expansion_factor": avg_expansion,
        "avg_noise": avg_noise,
        "prep_cpu": prep_cpu,
        "prep_ram_mb": prep_ram,
        "train_cpu": train_cpu,
        "train_ram_mb": train_ram,
        "infer_cpu": infer_cpu,
        "infer_ram_mb": infer_ram
    }

    # Save metrics to CSV (overwrite mode)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path, index=False)

    if verbose:
        logging.info(f"Benchmark completed. Metrics saved to {output_path}.")
        logging.info(f"Preparation time: {metrics['preparation_time']:.4f} seconds")
        logging.info(f"Training time: {metrics['training_time']:.4f} seconds")
        logging.info(f"Inference time: {metrics['inference_time']:.4f} seconds")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"MSE: {metrics['mse']:.4f}")





## Run the benchmark for Linear Means Classifier by BFV scheme in TenSEAL
# This function benchmarks the Linear Means Classifier using the BFV scheme in TenSEAL.
# It prepares the data, trains the model, performs inference, and evaluates metrics.
# It supports both plaintext and encrypted training modes, and saves the results to a CSV file.
def benchmark_tenseal_bfv_for_lmc(csv_path="boston.csv", output_dir="results", scheme="bfv", training_mode="plaintext", verbose=False):
    """
    Run the benchmark for Linear Means Classifier using BFV scheme in TenSEAL.
    
    Parameters:
        csv_path (str): Path to the CSV dataset file.
        output_dir (str): Directory to save the output metrics.
        scheme (str): Encryption scheme to use, default is 'bfv'.
        training_mode (str): 'plaintext' or 'encrypted' to select training approach.
        verbose (bool): If True, print detailed logs.
    returns:
        None: Saves metrics to a CSV file in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set output filename based on training mode
    if training_mode == "plaintext":
        output_path = os.path.join(output_dir, "metrics_tenseal_bfv_lmc_plaintext.csv")
    elif training_mode == "encrypted":
        output_path = os.path.join(output_dir, "metrics_tenseal_bfv_lmc_encrypted.csv")
    else:
        raise ValueError("Invalid training_mode. Use 'plaintext' or 'encrypted'.")

    if verbose:
        logging.info(f"Loading dataset from {csv_path}...")

    # Load dataset
    df = pd.read_csv(csv_path)

    # Convert regression to binary classification if needed
    if df.iloc[:, -1].nunique() > 10:
        df.iloc[:, -1] = pd.qcut(df.iloc[:, -1], q=2, labels=[0, 1])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----------------------------
    # Data preparation and server training
    # ----------------------------
    process = psutil.Process(os.getpid())
    preparation_start = time.time()
    tracemalloc.start()  # Start tracing memory allocations
    CPU_before = process.cpu_percent(interval=None)  # warm up / reset the counters
    prepare_and_send_data_toserver(X_train, y_train, training_mode=training_mode, scheme_type=scheme, verbose=verbose)
    preparation_end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop tracing memory allocations
    prep_ram = peak / (1024 * 1024)  # Convert to MB
    CPU_after = process.cpu_percent(interval=None)
    prep_cpu = CPU_after - CPU_before
    
    
    # ----------------------------
    process = psutil.Process(os.getpid())
    tracemalloc.start()  # Start tracing memory allocations
    CPU_before = process.cpu_percent(interval=None)  # warm up / reset the counters
    train_start = time.time()
    class_means = server_train_linear_mean_classifier(training_mode=training_mode, scheme=scheme,  verbose=verbose)
    train_end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop tracing memory allocations
    train_ram = peak / (1024 * 1024)  # Convert to MB
    CPU_after = process.cpu_percent(interval=None)
    train_cpu = CPU_after - CPU_before
   

    # ----------------------------
    # Inference
    # ----------------------------
    tenseal_manager = TenSealManager(scheme=scheme)
    enc_X_test = []

    for x in X_test:
        x_int = (x * 1000).astype(int)  # Scale and convert to int for BFV
        enc_vec = tenseal_manager.encrypt(x_int)
        enc_X_test.append(enc_vec)

    if verbose:
        logging.info(f"Encrypted {len(enc_X_test)} test vectors using {scheme} scheme.")

    process = psutil.Process(os.getpid())
    tracemalloc.start()  # Start tracing memory allocations
    CPU_before = process.cpu_percent(interval=None)  # warm up / reset the counterss
    infer_start = time.time()
    predictions = []
    for enc_vector in enc_X_test:
        pred = predict_linear_mean_model(enc_vector, class_means, tenseal_manager, training_mod=training_mode, scheme_type=scheme, verbose=verbose)
        predictions.append(pred)
    infer_end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop tracing memory allocations
    infer_ram = peak / (1024 * 1024)  # Convert to MB
    CPU_after = process.cpu_percent(interval=None)
    infer_cpu = CPU_after - CPU_before
    

    if verbose:
        logging.info(f"Inference completed for {len(predictions)} test vectors.")

    # ----------------------------
    # Evaluation Metrics
    # ----------------------------
    accuracy = accuracy_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    ciphertext_sizes = []
    plaintext_sizes = []
    noise_errors = []
    
    for i in range(len(enc_X_test)):
        enc_vec = enc_X_test[i]
        original_vec = X_test[i]
        serialized = enc_vec.serialize()
        ciphertext_sizes.append(len(serialized))
        plaintext_sizes.append(original_vec.nbytes)
        decrypted_vec = tenseal_manager.decrypt(enc_vec)
        noise_errors.append(np.max(np.abs(original_vec - decrypted_vec)))

    avg_ciphertext_size = np.mean(ciphertext_sizes)
    avg_plaintext_size = np.mean(plaintext_sizes)
    avg_expansion = np.mean([c/p for c, p in zip(ciphertext_sizes, plaintext_sizes)])
    avg_noise = np.mean(noise_errors)

    
    metrics = {
        "training_time": train_end - train_start,
        "preparation_time": preparation_end - preparation_start,
        "inference_time": infer_end - infer_start,
        "accuracy": accuracy,
        "mse": mse,
        "scheme": scheme,
        "training_mode": training_mode,
        "avg_ciphertext_size_bytes": avg_ciphertext_size,
        "avg_plaintext_size_bytes": avg_plaintext_size,
        "avg_expansion_factor": avg_expansion,
        "avg_noise": avg_noise,
        "prep_cpu": prep_cpu,
        "prep_ram_mb": prep_ram,
        "train_cpu": train_cpu,
        "train_ram_mb": train_ram,
        "infer_cpu": infer_cpu,
        "infer_ram_mb": infer_ram
    }


    # Save metrics (overwrite)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path, index=False)

    if verbose:
        logging.info(f"Benchmark completed. Metrics saved to {output_path}.")
        logging.info(f"Preparation time: {metrics['preparation_time']:.4f} seconds")
        logging.info(f"Training time: {metrics['training_time']:.4f} seconds")
        logging.info(f"Inference time: {metrics['inference_time']:.4f} seconds")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"MSE: {metrics['mse']:.4f}")



# ========================= Benchmark for Logistic Regression by ckks in TeanSEAL =========================
from client import CLIENT_CONTEXT_PATH, CONTEXT_PATH, ENC_DATA_PATH, prepare_and_send_data_toserver
from lr_training import sigmoid
from server import server_train_logistic_regression
from predictions import pred_logistic_regression_model

def benchmark_tenseal_ckks_for_logistic_regression(csv_path=CSV_FILE_PATH, output_dir=OUTPUT_DIR, scheme='ckks', training_mode='plaintext', verbose=False):
    """
    Run the benchmark for Logistic Regression on plaintext data.
    
    Parameters:
        csv_path (str): Path to the CSV dataset file.
        output_dir (str): Directory to save the output metrics.
        training_mode (str): 'plaintext' or 'encrypted' to select training approach.
        verbose (bool): If True, print detailed logs.
    
    Returns:
        None: Saves metrics to a CSV file in the specified output directory.
    """
   
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"metrics_tenseal_ckks_lrc_{training_mode}.csv")

    if verbose:
        logging.info(f"Loading dataset from {csv_path}...")

    # Load dataset
    df = pd.read_csv(csv_path)

    # Convert regression target to binary classification if needed
    if df.iloc[:, -1].nunique() > 10:
        df.iloc[:, -1] = pd.qcut(df.iloc[:, -1], q=2, labels=[0, 1])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # client: Prepare & send data to server

    process = psutil.Process(os.getpid())
    # Monitor resource usage before sending data
    tracemalloc.start()  # Start tracing memory allocations
    CPU_before = process.cpu_percent(interval=None)  # warm up /
    t0 = time.time()
    prepare_and_send_data_toserver(X_train, y_train, training_mode=training_mode, scheme_type=scheme, verbose=verbose)
    t1 = time.time()
    preparation_time = t1 - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop tracing memory allocations
    prep_ram = peak / (1024 * 1024)  # Convert to MB
    CPU_after = process.cpu_percent(interval=None)
    prep_cpu = CPU_after - CPU_before
   



    #  Server Train logistic regression model on plaintext data
    # Monitor resource usage before training
    process = psutil.Process(os.getpid())
    tracemalloc.start()  # Start tracing memory allocations
    CPU_before = process.cpu_percent(interval=None)  # warm up /
    train_start = time.time()
    model = server_train_logistic_regression(training_mode=training_mode, scheme='ckks', verbose=False) # model = (weights, bias)
    train_end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop tracing memory allocations
    train_ram = peak / (1024 * 1024)  # Convert to MB
    CPU_after = process.cpu_percent(interval=None)
    train_cpu = CPU_after - CPU_before
    
    

    # Encrypted Inference
    tenseal_manager = TenSealManager(scheme=scheme)
    enc_X_test = tenseal_manager.encrypt(X_test)
    if verbose:
        logging.info(f"Encrypted {len(enc_X_test)} test vectors using {scheme} scheme.")

    
    # Monitor resource usage before inference
    process = psutil.Process(os.getpid())
    tracemalloc.start()  # Start tracing memory allocations
    CPU_before = process.cpu_percent(interval=None)  # warm up / reset the counters   
    t4 = time.time()
    predictions = []
    for enc_vec in enc_X_test:
        pred = pred_logistic_regression_model(enc_vec, model, tenseal_manager, training_mod=training_mode, verbose=verbose)
        predictions.append(pred)
    t5 = time.time()
    inference_time = t5 - t4
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop tracing memory allocations
    infer_ram = peak / (1024 * 1024)  # Convert to MB
    CPU_after = process.cpu_percent(interval=None)
    infer_cpu = CPU_after - CPU_before
    
    none_count = sum(1 for p in predictions if p is None)
    logging.info(f"Number of None values: {none_count}")
    # Evaluation Metrics
    if none_count >=1:
        accuracy = 'NA'
        mse = 'NA'
    else:
        accuracy = accuracy_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)

    ciphertext_sizes = []
    plaintext_sizes = []
    noise_errors = []
    
    for i in range(len(enc_X_test)):
        enc_vec = enc_X_test[i]
        original_vec = X_test[i]
        serialized = enc_vec.serialize()
        ciphertext_sizes.append(len(serialized))
        plaintext_sizes.append(original_vec.nbytes)
        decrypted_vec = tenseal_manager.decrypt(enc_vec)
        noise_errors.append(np.max(np.abs(original_vec - decrypted_vec)))

    avg_ciphertext_size = np.mean(ciphertext_sizes)
    avg_plaintext_size = np.mean(plaintext_sizes)
    avg_expansion = np.mean([c/p for c, p in zip(ciphertext_sizes, plaintext_sizes)])
    avg_noise = np.mean(noise_errors)

    
    metrics = {
        "training_time": train_end - train_start,
        "preparation_time": preparation_time,
        "inference_time": inference_time,
        "accuracy": accuracy,
        "mse": mse,
        "scheme": scheme,
        "training_mode": training_mode,
        "avg_ciphertext_size_bytes": avg_ciphertext_size,
        "avg_plaintext_size_bytes": avg_plaintext_size,
        "avg_expansion_factor": avg_expansion,
        "avg_noise": avg_noise,
        "prep_cpu": prep_cpu,
        "prep_ram_mb": prep_ram,
        "train_cpu": train_cpu,
        "train_ram_mb": train_ram,
        "infer_cpu": infer_cpu,
        "infer_ram_mb": infer_ram
    }

    # Save metrics to CSV (overwrite mode)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path, index=False)

    if verbose:
        logging.info(f"Benchmark completed. Metrics saved to {output_path}.")
        logging.info(f"Preparation time: {preparation_time:.4f} seconds")
        logging.info(f"Training time: {metrics['training_time']:.4f} seconds")
        logging.info(f"Inference time: {metrics['inference_time']:.4f} seconds")
        logging.info(f"Accuracy: {metrics['accuracy']}")
        logging.info(f"MSE: {metrics['mse']}")

# =======================================================================================================


# ============= Benchmark of Logistic Regression Trained on Ciphertext by ckks in  TenSEAL ================ 
