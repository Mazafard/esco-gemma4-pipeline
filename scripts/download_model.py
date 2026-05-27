#!/usr/bin/env python3
"""
download_model.py - Cache Model Weights (Phase 1b) Entrypoint
"""

import os
import sys
import argparse
import logging

# Add the project root to sys.path to enable src.* imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.cache import preload_model_weights

# Setup detailed logging
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "download.log"), mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("DownloadModel")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="unsloth/gemma-4-E4B-it")
    args = parser.parse_args()

    try:
        preload_model_weights(args.base_model)
    except Exception as e:
        logger.error(f"[-] Failed to cache model weights: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
