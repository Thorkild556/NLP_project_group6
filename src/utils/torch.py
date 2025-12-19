import torch


def ensure_device() -> None:
    """
    checks which device does the torch could run our code (prints the device name)
    :return: None
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("We would be using this device:", device)
