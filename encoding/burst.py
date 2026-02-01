import torch

def _to_unit_interval(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """ 
    Ensure x is in [0, 1].
    If x appears to be normalized to [-1, 1] (common with Normalize(mean=0.5, std=0.5)),
    convert back to [0, 1] via (x + 1) / 2.
    """
    if x.min() < -eps:
        x = (x+1.0) / 2.0
    return x.clamp(0.0,1.0)

def burst_encode_flat( 
    x: torch.Tensor,
    time_steps: int,
    max_spikes: int | None = None,
    threshold: float = 0.0,
    early: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    """
     Burst encoding (flat): returns spike trains shaped [T, B, N].

        Interpretation:
        - Each pixel produces K spikes, where K increases with intensity.
        - K is capped by max_spikes (default: time_steps).
        - Spikes are placed as a short burst. If early=True, the burst starts at t=0.
        If early=False, the burst is centered in time.

        Args:
            x: [B, 1, H, W] or [B, N]
            time_steps: number of timesteps T
            max_spikes: maximum spikes per pixel (<= T). Default: T
            threshold: values <= threshold produce no spikes
            early: if True, burst occupies earliest timesteps; else centered
            device: optional device override
            dtype: output dtype

        Returns:
        spikes: [T, B, N] with 0/1 values
    """
    if time_steps <= 0:
        raise ValueError("time_steps must be > 0")

    if device is None:
        device = x.device

    if max_spikes is None:
        max_spikes = time_steps
    max_spikes = int(max_spikes)
    if max_spikes <= 0:
        raise ValueError("max_spikes must be > 0")
    if max_spikes > time_steps:
        raise ValueError("max_spikes must be <= time_steps")

    x = x.to(device=device)
    x_unit = _to_unit_interval(x)

    # Flatten to [B, N]
    if x_unit.ndim == 4:
        B = x_unit.shape[0]
        x_flat = x_unit.view(B, -1)
    elif x_unit.ndim == 2:
        x_flat = x_unit
        B = x_flat.shape[0]
    else:
        raise ValueError(f"Unsupported input shape {x_unit.shape}. Expected [B,N] or [B,1,H,W].")

    B, N = x_flat.shape
    spikes = torch.zeros((time_steps, B, N), device=device, dtype=dtype)

    # Determine burst count K per pixel: K in [0, max_spikes]
    # Use floor(x * max_spikes). x=1 -> max_spikes spikes, x=0 -> 0 spikes.
    K = torch.floor(x_flat * max_spikes).long()
    K = torch.clamp(K, 0, max_spikes)

    # Apply threshold: values <= threshold -> K=0
    K = torch.where(x_flat > threshold, K, torch.zeros_like(K))

    if not (K > 0).any():
        return spikes

    # Place spikes for each pixel for timesteps t < K (burst)
    if early:
        # Burst from t=0 to t=K-1
        for t in range(time_steps):
            spikes[t] = (K > t).to(dtype)
    else:
        # Center the burst around the middle
        # start = mid - K//2
        mid = time_steps // 2
        start = mid - (K // 2)  # [B,N]
        for t in range(time_steps):
            spikes[t] = ((t >= start) & (t < start + K) & (K > 0)).to(dtype)

    return spikes
