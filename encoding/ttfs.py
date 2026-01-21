import torch

def ttfs_encode_flat(x, time_steps):
    B = x.size(0)
    x_flat = x.view(B, -1)

    x_denorm = torch.clamp(0.5 * x_flat + 0.5, 0.0, 1.0)
    eps = 1e-3
    x_clamped = torch.clamp(x_denorm, eps, 1.0)

    spike_times = ((1.0 - x_clamped) * (time_steps - 1)).round().long()

    spikes = torch.zeros(time_steps, B, x_flat.size(1), device=x.device)
    for b in range(B):
        idx = torch.arange(x_flat.size(1), device=x.device)
        spikes[spike_times[b], b, idx] = 1.0

    return spikes