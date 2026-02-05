import torch

def _to_unit_interval(x:torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Ensure x is in [0, 1].
    If input is already normalized to [-1, 1] (common with Normalize(mean=0.5,std=0.5)),
    convert to [0, 1] via (x + 1) / 2.
    """
    # If values look like [-1,1], shift to [0,1]
    if x.min() < -eps:
        x = (x+1.0) / 2.0
    return x.clamp(0.0,1.0)

def phase_encode_flat(x:torch.Tensor, time_steps: int, flatten: bool = True, 
                      invert: bool = False, threshold: float = 0.0,
                      device = None,
                      dtype: torch.dtype = torch.float32,
                      ) -> torch.Tensor:
    """
    Convert static input x to phase-coded spike trains.

    Args:
        x: Input tensor.
           Common shapes:
             - [B, 1, H, W] (images)
             - [B, N] (already flattened)
        time_steps: Number of timesteps T.
        flatten: If True, output shape is [T, B, N].
                 If False and x is image-shaped, output [T, B, 1, H, W].
        invert: If True, invert intensity mapping (dark -> earlier spike).
        threshold: Values <= threshold produce no spike at all.
                   Use 0.0 to allow all nonzero pixels to spike.
        device: Optional device override.
        dtype: Output dtype.

    Returns:
        spikes: Binary spikes over time.
                Shape:
                  - if flatten=True:  [T, B, N]
                  - else (image input): [T, B, 1, H, W]
    """
    if time_steps <= 0:
        raise ValueError("time_steps must be > 0")

    if device is None:
        device = x.device

    x = x.to(device=device)
    x_unit = _to_unit_interval(x)

    if invert:
        x_unit = 1.0 - x_unit

    # Apply threshold mask (no spike for small values)
    fire_mask = x_unit > threshold

    # Flatten if requested or if input is already [B, N]
    original_shape = x_unit.shape
    if x_unit.ndim == 2:
        # [B, N]
        x_flat = x_unit
        mask_flat = fire_mask
        out_flat = True
        B, N = x_flat.shape
    elif x_unit.ndim == 4:
        # [B, 1, H, W]
        B = x_unit.shape[0]
        if flatten:
            x_flat = x_unit.view(B, -1)
            mask_flat = fire_mask.view(B, -1)
            out_flat = True
            N = x_flat.shape[1]
        else:
            out_flat = False
    else:
        raise ValueError(f"Unsupported input shape {original_shape}. Expected [B,N] or [B,1,H,W].")

    # Allocate output spikes
    if out_flat:
        spikes = torch.zeros((time_steps, B, N), device=device, dtype=dtype)

        # Map intensity in [0,1] -> spike time in [0, T-1]
        # High intensity -> early time.
        # t = floor((1 - x) * (T-1))
        # x=1 -> t=0, x=0 -> t=T-1
        t_float = (1.0 - x_flat) * (time_steps - 1)
        t_idx = torch.floor(t_float).long().clamp(0, time_steps - 1)

        # Only set spikes where mask is True
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(t_idx)
        n_idx = torch.arange(N, device=device).unsqueeze(0).expand_as(t_idx)

        # Can’t directly index with a 3D mask, so filter
        valid = mask_flat
        spikes[t_idx[valid], b_idx[valid], n_idx[valid]] = 1.0

        return spikes

    else:
        # Non-flattened output: [T, B, 1, H, W]
        _, C, H, W = x_unit.shape
        spikes = torch.zeros((time_steps, B, C, H, W), device=device, dtype=dtype)

        t_float = (1.0 - x_unit) * (time_steps - 1)
        t_idx = torch.floor(t_float).long().clamp(0, time_steps - 1)

        # Build indices for assignment
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand_as(t_idx)
        c_idx = torch.arange(C, device=device).view(1, C, 1, 1).expand_as(t_idx)
        h_idx = torch.arange(H, device=device).view(1, 1, H, 1).expand_as(t_idx)
        w_idx = torch.arange(W, device=device).view(1, 1, 1, W).expand_as(t_idx)

        valid = fire_mask
        spikes[t_idx[valid], b_idx[valid], c_idx[valid], h_idx[valid], w_idx[valid]] = 1.0

        return spikes
