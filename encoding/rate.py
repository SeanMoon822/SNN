import torch

def rate_encode_flat(x: torch.Tensor, time_steps: int) -> torch.Tensor:
    """
    Rate encoding
    - Take the flattened (normalized) image and repeat it for every timestep.

    Args:
        x: [B, 1, 28, 28] (normalized by Normalize(mean=0.5, std=0.5))
        time_steps: T
    """
    B = x.size(0)
    x_flat = x.view(B, -1)  # [B, 784]
    seq = x_flat.unsqueeze(0).repeat(time_steps, 1, 1)  # [T, B, 784]
    return seq

def denorm_to_unit(x: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized tensor (approx [-1,1]) back to [0,1].
    """
    x01 = 0.5 * x + 0.5
    return torch.clamp(x01, 0.0, 1.0)
