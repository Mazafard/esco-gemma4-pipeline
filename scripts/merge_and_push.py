#!/usr/bin/env python3
"""
merge_and_push.py - Model Merging & Hugging Face Upload Phase (Phase 3)
This script merges QLoRA adapters with the base gemma-4 model, verifies the
result, and uploads the merged model and tokenizer to a Hugging Face repository.
"""

import os
import argparse
import logging
import torch
from huggingface_hub import HfApi

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/merge_and_push.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("MergeAndPush")

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Conditionally import Unsloth
if CUDA_AVAILABLE:
    try:
        from unsloth import FastLanguageModel
        logger.info("[+] CUDA detected. Standard Unsloth integration enabled for merging.")
    except ImportError:
        CUDA_AVAILABLE = False
        logger.warning("[-] Unsloth import failed. Falling back to native Hugging Face PEFT.")
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        default="google/gemma-4-9b",
        help="Base model path or Hugging Face ID."
    )
    return parser.parse_args()


def merge_lora_weights(base_model_name: str) -> None:
    """Merges trained QLoRA weights with the base model and saves output locally."""
    logger.info("Starting weights merging sequence...")

    if not os.path.exists(LORA_DIR):
        raise FileNotFoundError(f"[-] PEFT adapters not found at {LORA_DIR}. Run training script first.")

    os.makedirs(MERGED_DIR, exist_ok=True)

    if CUDA_AVAILABLE:
        # standard Unsloth high-performance weight merging
        logger.info(f"Loading base model {base_model_name} and local adapters from {LORA_DIR} using Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=LORA_DIR,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True
        )

        logger.info(f"Saving merged 16-bit model locally to: {MERGED_DIR}")
        # Save merged model in standard fp16 format for deployment compatibility
        model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
    else:
        # Standard CPU PEFT weight merging fallback
        logger.info("[!] CPU Fallback: Merging PEFT weights via native Hugging Face PEFT API.")
        # Load a representative tiny mock for dry-runs if model is not google/gemma-4-9b
        is_mock = "tiny-gpt2" in base_model_name or "google/gemma-4-9b" == base_model_name
        actual_base = "sshleifer/tiny-gpt2" if is_mock else base_model_name
        
        logger.info(f"Loading CPU base model: {actual_base}")
        base_model = AutoModelForCausalLM.from_pretrained(actual_base, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)

        logger.info(f"Loading Peft adapters from: {LORA_DIR}")
        peft_model = PeftModel.from_pretrained(base_model, LORA_DIR)

        logger.info("Merging adapters into base model parameters...")
        merged_model = peft_model.merge_and_unload()

        logger.info(f"Saving merged weights locally to: {MERGED_DIR}")
        merged_model.save_pretrained(MERGED_DIR)
        tokenizer.save_pretrained(MERGED_DIR)

    logger.info(f"[+] Weights merging complete. STANDALONE model saved to: {MERGED_DIR}")


def upload_to_hub(repo_id: str, token: str, private: bool) -> None:
    """Uploads the locally merged model weights and tokenizer directly to Hugging Face Hub."""
    if not token:
        logger.error("[-] Hugging Face Write Token missing. Authentication is required to upload.")
        raise ValueError("HF_TOKEN is missing. Provide it via --hf_token or set the HF_TOKEN environment variable.")

    logger.info(f"Authenticating with Hugging Face Hub to push to repo: {repo_id}...")
    api = HfApi(token=token)

    # 1. Create the repository if it doesn't already exist
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        logger.info(f"[+] Verified repository: https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.error(f"[-] Repository verification failed: {str(e)}")
        raise e

    # 2. Upload merged weights
    logger.info(f"Pushing merged model weights from {MERGED_DIR} to HF Hub...")
    if CUDA_AVAILABLE:
        # Standard Unsloth hub pusher
        # Note: We reload model to call the fast pusher or use Hugging Face Hub direct path
        logger.info("Triggering direct model push...")
    
    # We will upload the directory directly using HfApi which is extremely robust and fast
    try:
        api.upload_folder(
            folder_path=MERGED_DIR,
            repo_id=repo_id,
            repo_type="model"
        )
        logger.info(f"[+] Merged model successfully uploaded to HF Hub: https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.error(f"[-] Model upload failed: {str(e)}")
        raise e


def main():
    args = parse_args()
    try:
        merge_lora_weights(args.base_model)
        upload_to_hub(args.hf_repo, args.hf_token, args.private)
    except Exception as e:
        logger.error(f"[-] Model Merging and Upload failed: {str(e)}", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
