import torch

def is_cuda_available() -> bool:
    return torch.cuda.is_available()

def is_bf16_supported() -> bool:
    return torch.cuda.is_bf16_supported() if is_cuda_available() else False

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
