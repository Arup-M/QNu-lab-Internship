#!/usr/bin/env bash
set -euo pipefail

echo "🔄 1) Build (or rebuild) images"
docker-compose build

echo "🚀 2) Run client + server (foreground, stop on exit)…"
docker-compose up --abort-on-container-exit

echo "📊 3) Run main.py to print benchmark tables"
docker-compose run --entrypoint "" --rm fhe_benchmark python main.py

echo "✅  Benchmark complete — results CSVs are in:" 
echo "   results_ckks/metrics_tenseal_ckks_lmc_plaintext.csv"
echo "   results_ckks/metrics_tenseal_ckks_lmc_encrypted.csv"
echo "   results_bfv/metrics_tenseal_bfv_lmc_plaintext.csv"     
echo "   results_bfv/metrics_tenseal_bfv_lmc_encrypted.csv"
echo "   results_ckks/metrics_logistic_regression.csv"
