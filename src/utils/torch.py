import torch

def ensure_device() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("We would be using this device:", device)