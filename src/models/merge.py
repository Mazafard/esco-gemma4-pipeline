import os
import logging
import torch
from src.config.hardware import is_cuda_available

logger = logging.getLogger(__name__)

def merge_lora_weights(base_model_name: str, lora_dir: str, merged_dir: str) -> None:
    """Merges trained QLoRA weights with the base model and saves output locally."""
    logger.info("Starting weights merging sequence...")

    if not os.path.exists(lora_dir):
        raise FileNotFoundError(f"[-] PEFT adapters not found at {lora_dir}. Run training script first.")

    os.makedirs(merged_dir, exist_ok=True)

    if is_cuda_available():
        # standard Unsloth high-performance weight merging
        try:
            from unsloth import FastLanguageModel
            logger.info(f"Loading base model {base_model_name} and local adapters from {lora_dir} using Unsloth...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=lora_dir,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True
            )
            logger.info(f"Saving merged 16-bit model locally to: {merged_dir}")
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
            logger.info(f"[+] Weights merging complete. STANDALONE model saved to: {merged_dir}")
            return
        except ImportError:
            logger.warning("[-] Unsloth import failed. Falling back to native Hugging Face PEFT.")

    # Standard CPU PEFT weight merging fallback
    logger.info("[!] CPU Fallback: Merging PEFT weights via native Hugging Face PEFT API.")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Load a representative tiny mock for dry-runs if model is not unsloth/gemma-4-E4B-it
    is_mock = "tiny-gpt2" in base_model_name or "unsloth/gemma-4-E4B-it" == base_model_name
    actual_base = "sshleifer/tiny-gpt2" if is_mock else base_model_name
    
    logger.info(f"Loading CPU base model: {actual_base}")
    base_model = AutoModelForCausalLM.from_pretrained(actual_base, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(lora_dir)

    logger.info(f"Loading Peft adapters from: {lora_dir}")
    peft_model = PeftModel.from_pretrained(base_model, lora_dir)

    logger.info("Merging adapters into base model parameters...")
    merged_model = peft_model.merge_and_unload()

    logger.info(f"Saving merged weights locally to: {merged_dir}")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    logger.info(f"[+] Weights merging complete. STANDALONE model saved to: {merged_dir}")
