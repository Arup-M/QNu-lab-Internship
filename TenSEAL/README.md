# TenSEAL machine-learning benchmarks

Experimental internship code for benchmarking a Linear Means Classifier with CKKS and BFV, and logistic regression with CKKS, using TenSEAL. `main.py` runs the plaintext-training and encrypted-training benchmark cases.

## Files

- `benchmark.py`, `tenseal_utils.py`, `lm_training.py`, `lr_training.py`, and `predictions.py`: benchmark, encryption, training, and inference helpers.
- `client.py` and `server.py`: client/server helper code used by the experiments.
- `boston.csv` and `clean_boston.csv`: input data supplied with the original project.
- `results/`, `results_bfv/`, and `results_ckks/`: historical benchmark CSV files.
- `Dockerfile`, `docker-compose.yml`, and `run_benchmark.sh`: original container configuration and runner.

## Run the benchmark

From this directory, with Docker and Compose installed:

```sh
docker compose build
docker compose run --rm fhe_benchmark python main.py
```

The container setup uses Python 3.10 and provides the `/shared` directory expected by the code. It mounts `results_ckks/` and `results_bfv/` so new output CSVs are retained on the host. Running a benchmark can overwrite saved CSV files. The original shell runner uses the older `docker-compose` command.

For an existing Python environment, install `requirements.txt` and run `python main.py`; local execution also requires a writable `/shared` directory. Dependencies are unpinned.

## Experimental limitations

The code and saved results are preserved as an experimental snapshot. A full benchmark run has not been verified during this upload. Some encrypted logistic-regression paths decrypt intermediate values and contain context-handling and prediction-return limitations; this snapshot should not be treated as a validated end-to-end private training system.

## Generated files

Cryptographic contexts (`*.ctx`), serialized artifacts (`*.pkl`), Python caches, and the nested duplicate project folder are omitted. Generate any required contexts and encrypted artifacts locally. Keep secret contexts and keys private. The ignore files prevent these generated files from entering Git commits and Docker build contexts.
