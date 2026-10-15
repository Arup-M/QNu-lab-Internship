## -*- coding: utf-8 -*-
import logging
from benchmark import benchmark_tenseal_ckks_for_lmc
from benchmark import benchmark_tenseal_bfv_for_lmc

# Set up logging to print to stdout (safe inside Docker)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

if __name__ == "__main__":
    logging.info("<==========Starting benchmark for Linear Means Classifier with TenSEAL==========>\n")
    
    # Benchmark using CKKS scheme
    logging.info("Generating benchmark metrics for Linear Means Classifier using CKKS scheme in TenSEAL...\n")
    logging.info("<========== First case: Training on plaintext data and inference on encrypted data ==========>\n")
    benchmark_tenseal_ckks_for_lmc(csv_path="boston.csv", output_dir="results_ckks", scheme="ckks", training_mode="plaintext", verbose=True)
    logging.info("<========== Second case: Training on encrypted data and inference on encrypted data ==========>\n")
    benchmark_tenseal_ckks_for_lmc(csv_path="boston.csv", output_dir="results_ckks", scheme="ckks", training_mode="encrypted", verbose=True)
    logging.info("Benchmarking completed for Linear Means Classifier with TenSEAL using CKKS scheme.\n")

    # Benchmark using BFV scheme
    logging.info("Generating benchmark metrics for Linear Means Classifier using BFV scheme in TenSEAL...\n")
    logging.info("<========== First case: Training on plaintext data and inference on encrypted data ==========>\n")   
    benchmark_tenseal_bfv_for_lmc(csv_path="boston.csv", output_dir="results_bfv", scheme="bfv", training_mode="plaintext", verbose=True)
    logging.info("<========== Second case: Training on encrypted data and inference on encrypted data ==========>\n")
    benchmark_tenseal_bfv_for_lmc(csv_path="boston.csv", output_dir="results_bfv", scheme="bfv", training_mode="encrypted", verbose=True)
    logging.info("Benchmarking completed for Linear Means Classifier with TenSEAL.")

    # ================== Benchmark for Logistic Regression ==================
    from benchmark import benchmark_tenseal_ckks_for_logistic_regression
    logging.info("\n<========== Starting benchmark for Logistic Regression with TenSEAL ==========>\n")
    logging.info("Generating benchmark metrics for Logistic Regression using CKKS scheme in TenSEAL...\n")
    logging.info("<========== First case: Training on plaintext data and inference on encrypted data ==========>\n")
    benchmark_tenseal_ckks_for_logistic_regression(csv_path="boston.csv", output_dir="results_ckks", scheme="ckks", training_mode="plaintext", verbose=True)
    logging.info("<========== Second case: Training on encrypted data and inference on encrypted data ==========>\n")
    benchmark_tenseal_ckks_for_logistic_regression(csv_path="boston.csv", output_dir="results_ckks", scheme="ckks", training_mode="encrypted", verbose=True)
    logging.info("Benchmarking completed for Logistic Regression with TenSEAL using CKKS scheme.\n")
