import torch
import torch.nn as nn
import torch.nn.functional as F

class LargeUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, init_features=64):
        """
        Large UNet model for RAW to RGB conversion.
        
        Args:
            in_channels: Number of input channels (4 for packed Bayer RAW)
            out_channels: Number of output channels (3 for RGB)
            init_features: Number of initial features (controls model size)
        """
        super(LargeUNet, self).__init__()
        
        features = init_features
        self.encoder1 = LargeUNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = LargeUNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = LargeUNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = LargeUNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = LargeUNet._block(features * 8, features * 16, name="bottleneck")
        
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = LargeUNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = LargeUNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = LargeUNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = LargeUNet._block(features * 2, features, name="dec1")
        
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        
        # Additional large model components
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(features * 16) for _ in range(4)]
        )
        
        # Attention gates
        self.attention3 = AttentionGate(features * 4, features * 8)
        self.attention2 = AttentionGate(features * 2, features * 4)
        self.attention1 = AttentionGate(features, features * 2)
        
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck with residual blocks
        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = self.residual_blocks(bottleneck)
        
        # Decoder path with attention gates
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((self.attention3(enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((self.attention2(enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((self.attention1(enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.LeakyReLU(0.2, inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_l, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = LargeUNet(in_channels=4, out_channels=3, init_features=64)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test with sample input
    sample_input = torch.randn(2, 4, 256, 256)  # Batch of 2, 4-channel RAW, 256x256
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
