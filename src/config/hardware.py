import torch

def is_cuda_available() -> bool:
    return torch.cuda.is_available()

def is_bf16_supported() -> bool:
    return torch.cuda.is_bf16_supported() if is_cuda_available() else False
