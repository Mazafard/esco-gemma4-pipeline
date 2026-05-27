import os
import logging
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

def verify_or_create_repo(repo_id: str, token: str, private: bool) -> HfApi:
    """Authenticates and ensures the target repository exists."""
    if not token:
        logger.error("[-] Hugging Face Write Token missing. Authentication is required to upload.")
        raise ValueError("HF_TOKEN is missing. Provide it via --hf_token or set the HF_TOKEN environment variable.")

    logger.info(f"Authenticating with Hugging Face Hub to push to repo: {repo_id}...")
    api = HfApi(token=token)

    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        logger.info(f"[+] Verified repository: https://huggingface.co/{repo_id}")
        return api
    except Exception as e:
        logger.error(f"[-] Repository verification failed: {str(e)}")
        raise e

def upload_directory_to_hub(api: HfApi, folder_path: str, repo_id: str) -> None:
    """Uploads a local directory directly to the Hugging Face Hub."""
    logger.info(f"Pushing merged model weights from {folder_path} to HF Hub...")
    logger.info("Triggering direct model push...")
    
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model"
        )
        logger.info(f"[+] Merged model successfully uploaded to HF Hub: https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.error(f"[-] Model upload failed: {str(e)}")
        raise e

def generate_model_card(folder_path: str, repo_id: str) -> None:
    """Generates a Model Card (README.md) for the Hugging Face Hub."""
    readme_path = os.path.join(folder_path, "README.md")
    
    # Simple, professional markdown format for the model card
    card_content = f"""---
language:
- en
license: apache-2.0
tags:
- unsloth
- gemma-4
- classification
- esco
- occupations
---

# {repo_id.split('/')[-1] if '/' in repo_id else repo_id}

This model is a fine-tuned version of `gemma-4` designed to map job descriptions and user inputs to official ESCO (European Skills, Competences, Qualifications and Occupations) taxonomies. 

The model was trained efficiently utilizing the `unsloth` library and merged into 16-bit format for optimal inference performance.

## Usage (Inference)

You can load this model directly using `unsloth` or standard `transformers` for sequence generation. For best results, use the identical prompt formatting applied during fine-tuning (see the provided `inference.py` script).

### Loading with Unsloth

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{repo_id}",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

## Training Framework
- **Base Model**: Google Gemma 4 (4-bit)
- **Fine-Tuning**: LoRA / PEFT
- **Engine**: [Unsloth](https://github.com/unslothai/unsloth)
"""
    
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card_content)
    logger.info(f"[+] Generated HF Model Card at {readme_path}")


def upload_to_hub(repo_id: str, token: str, private: bool, folder_path: str) -> None:
    """Wrapper to authenticate and upload weights."""
    api = verify_or_create_repo(repo_id, token, private)
    generate_model_card(folder_path, repo_id)
    upload_directory_to_hub(api, folder_path, repo_id)
