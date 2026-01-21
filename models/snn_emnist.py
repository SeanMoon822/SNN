import torch
import torch.nn as nn
from models.lif import LIFLayer
from encoding.ttfs import ttfs_encode_flat

class SNN_EMNIST(nn.Module):
    def __init__(self, time_steps=10, hidden_dim=256, num_classes=47,
                 coding="rate", tau_out=2.0):
        super().__init__()
        self.time_steps = time_steps
        self.coding = coding.lower()
        self.hidden_dim = hidden_dim

        self.lif = LIFLayer(28*28, hidden_dim)
        self.readout = nn.Linear(hidden_dim, num_classes)
        self.tau_out = tau_out

    def forward(self, x):
        B = x.size(0)
        v_out = torch.zeros(B, self.readout.out_features, device=x.device)
        v_hid = None

        if self.coding == "rate":
            x_flat = x.view(B, -1)
            for _ in range(self.time_steps):
                spikes, v_hid = self.lif(x_flat, v_hid)
                I = self.readout(spikes)
                v_out = v_out + (I - v_out) / self.tau_out

        elif self.coding == "ttfs":
            spike_seq = ttfs_encode_flat(x, self.time_steps)
            for t in range(self.time_steps):
                spikes, v_hid = self.lif(spike_seq[t], v_hid)
                I = self.readout(spikes)
                v_out = v_out + (I - v_out) / self.tau_out

        return v_out