#!/usr/bin/env python3
"""
generate_journal.py - Citation & Academic Journal Generation Phase Entrypoint
"""

import os
import sys
import logging

# Add the project root to sys.path to enable src.* imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.reporting import (
    load_telemetry_data,
    load_dataset_size,
    generate_markdown_journal,
    generate_metrics_plot
)

# Setup detailed logging
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "generate_journal.log"), mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("JournalGen")

# Paths Configuration
METRICS_PATH = os.path.join(BASE_DIR, "logs", "training_metrics.json")
DATASET_PATH = os.path.join(BASE_DIR, "data", "esco_data.json")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
JOURNAL_PATH = os.path.join(ARTIFACTS_DIR, "experimental_journal.md")
PLOT_PATH = os.path.join(ARTIFACTS_DIR, "training_metrics_plot.png")


def main():
    try:
        metrics = load_telemetry_data(METRICS_PATH)
        dataset_size = load_dataset_size(DATASET_PATH)
        generate_metrics_plot(metrics, PLOT_PATH)
        generate_markdown_journal(metrics, dataset_size, JOURNAL_PATH)
    except Exception as e:
        logger.error(f"[-] Journal generation failed: {str(e)}", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
