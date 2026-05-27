#!/usr/bin/env python3
"""
merge_and_push.py - Model Merging & Hugging Face Upload Phase Entrypoint
"""

import os
import sys
import argparse
import logging

# Add the project root to sys.path to enable src.* imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.merge import merge_lora_weights
from src.models.hub import upload_to_hub

# Setup detailed logging
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "merge_and_push.log"), mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("MergeAndPush")

LORA_DIR = os.path.join(BASE_DIR, "outputs", "lora_adapters")
MERGED_DIR = os.path.join(BASE_DIR, "outputs", "merged_model")

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters and upload to Hugging Face Hub.")
    parser.add_argument(
        "--hf_repo",
        type=str,
        required=True,
        help="Target Hugging Face repository (e.g. 'username/gemma-4-esco')."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face Hub Write Token. Defaults to the HF_TOKEN environment variable."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Set the Hugging Face repository to private."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/gemma-4-E4B-it",
        help="Base model path or Hugging Face ID."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        merge_lora_weights(args.base_model, LORA_DIR, MERGED_DIR)
        upload_to_hub(args.hf_repo, args.hf_token, args.private, MERGED_DIR)
    except Exception as e:
        logger.error(f"[-] Model Merging and Upload failed: {str(e)}", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
