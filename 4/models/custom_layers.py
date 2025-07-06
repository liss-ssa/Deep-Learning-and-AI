import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomConv2d(nn.Module):
    """Кастомный сверточный слой с энергетической регуляризацией"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.energy_scale = nn.Parameter(torch.tensor(1.0))
        self.energy_bias = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        x = self.conv(x)
        if self.training:
            energy = x.pow(2).mean(dim=[1,2,3], keepdim=True)
            x = x * (self.energy_scale / (energy + 1e-6) + self.energy_bias)
        return x

class ChannelAttention(nn.Module):
    """Механизм внимания по каналам"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.mlp(self.avg_pool(x).view(b, c))
        max = self.mlp(self.max_pool(x).view(b, c))
        return x * (avg + max).view(b, c, 1, 1)

class Swish(nn.Module):
    """Функция активации Swish"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionAwarePooling(nn.Module):
    """Pooling с сохранением позиционной информации"""
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        
    def forward(self, x):
        output, indices = F.max_pool2d(
            x, self.kernel_size, self.stride,
            return_indices=True
        )
        
        b, c, h, w = output.shape
        pos_features = torch.zeros(b, c*3, h, w, device=x.device)
        
        for bi in range(b):
            for ci in range(c):
                for hi in range(h):
                    for wi in range(w):
                        idx = indices[bi, ci, hi, wi].item()
                        pos_h = idx // x.size(3)
                        pos_w = idx % x.size(3)
                        
                        pos_features[bi, ci, hi, wi] = output[bi, ci, hi, wi]
                        pos_features[bi, c + ci, hi, wi] = pos_h / x.size(2)
                        pos_features[bi, 2*c + ci, hi, wi] = pos_w / x.size(3)
        
        return pos_features

def test_layers():
    """Тестирование всех слоев"""
    print("Testing CustomConv2d...")
    conv = CustomConv2d(3, 16, 3)
    x = torch.randn(2, 3, 32, 32)
    print("Input shape:", x.shape)
    print("Output shape:", conv(x).shape)
    
    print("\nTesting ChannelAttention...")
    attn = ChannelAttention(64)
    x = torch.randn(4, 64, 32, 32)
    print("Input shape:", x.shape)
    print("Output shape:", attn(x).shape)

if __name__ == "__main__":
    test_layers()