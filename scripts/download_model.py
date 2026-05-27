#!/usr/bin/env python3
"""
download_model.py - Cache Model Weights (Phase 1b)
This script leverages Unsloth to accurately download and cache the precise quantized weights 
so that the fine-tuning phase can start immediately without network operations.
"""

import os
import sys
import argparse
import logging
import torch

# Base Directory Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# Setup detailed logging
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

    CUDA_AVAILABLE = torch.cuda.is_available()

    if not CUDA_AVAILABLE:
        logger.info("[-] No CUDA detected. Skipping Unsloth model caching, relying on CPU fallback.")
        return

    try:
        from unsloth import FastLanguageModel
        
        logger.info(f"--> Initiating pre-caching for model: {args.base_model} in 4-bit...")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.base_model,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
                local_files_only=True,
            )
            logger.info("[+] Loaded model directly from local cache!")
        except Exception:
            logger.info("--> Cache miss. Downloading model from Hugging Face...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.base_model,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
                local_files_only=False,
            )
        logger.info("[+] Successfully downloaded and cached model weights!")
    except Exception as e:
        logger.error(f"[-] Failed to download model weights: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
