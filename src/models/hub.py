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

def upload_to_hub(repo_id: str, token: str, private: bool, folder_path: str) -> None:
    """Wrapper to authenticate and upload weights."""
    api = verify_or_create_repo(repo_id, token, private)
    upload_directory_to_hub(api, folder_path, repo_id)
