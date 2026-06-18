"""
mlp_v2.py — BurnTrack MLP Corrector v2
Architecture according to official spec: 5->64->32->1, GELU, LayerNorm, Dropout(0.2)
Features: fuel_target_encoded, wind_speed, humidity, slope, ros_rothermel
"""
import torch
import torch.nn as nn


class BurnTrackMLP(nn.Module):
    """
    MLP minimaliste pour correction ROS.
    Input: 8 features (wind, humidity, slope, ros_rothermel, h_dead, sigma, m_live_woody, mx)
    Output: delta_ros (correction additive)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
