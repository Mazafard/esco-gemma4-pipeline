import logging
from typing import Optional
from src.config.hardware import is_cuda_available

logger = logging.getLogger(__name__)

def preload_model_weights(base_model: str) -> None:
    """
    Leverages Unsloth to accurately download and cache the precise quantized weights 
    so that the fine-tuning phase can start immediately without network operations.
    """
    if not is_cuda_available():
        logger.info("[-] No CUDA detected. Skipping Unsloth model caching, relying on CPU fallback.")
        return

    try:
        from unsloth import FastLanguageModel
        
        logger.info(f"--> Initiating pre-caching for model: {base_model} in 4-bit...")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
                local_files_only=True,
            )
            logger.info("[+] Loaded model directly from local cache!")
        except Exception:
            logger.info("--> Cache miss. Downloading model from Hugging Face...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
                local_files_only=False,
            )
        logger.info("[+] Successfully downloaded and cached model weights!")
    except Exception as e:
        logger.error(f"[-] Failed to download model weights: {e}")
        raise e
